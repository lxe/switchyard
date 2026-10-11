"""CPU acceptance for the prerequisite gate and pinned native adapters."""

import ast
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PYTHON = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON))
import switchyard_vllm_hooks as hooks
import switchyard_vllm_startup as startup


class PolicyTests(unittest.TestCase):
    def test_default_is_legacy(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(startup.Policy.from_environment())

    def test_configuration_validation(self):
        for value in [[], {"unknown": 1}, {"schema_version": True},
                      {"worker_method": "forkserver"}, {"tokenizer_preload": 1},
                      {"cache_aliases": "/tmp/cache"}, {"compute_capability": "sm_89"},
                      {"strategy": "prepared", "worker_method": "guarded_fork"}]:
            with self.subTest(value=value), patch.dict(os.environ, {
                "VLLM_ENGINE_STARTUP_POLICY": json.dumps(value)
            }, clear=True), self.assertRaises((ValueError, TypeError)):
                startup.Policy.from_environment()

    def test_guarded_fork_requires_selected_baked_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = startup.Policy(strategy="prepared", worker_method="guarded_fork",
                                    fork_runtime_id="a" * 64, compute_capability="12.0", telemetry=False)
            receipt = {"source_patches": [{"descriptor_sha256": startup.NVML_PATCH,
                       "after_sha256": startup.NVML_RESULT, "target": startup.NVML_TARGET,
                       "package": {"name": "vllm", "version": "0.31.0+cu129"}}]}
            (root / "build-info.json").write_text(json.dumps(receipt))
            environment = {"SWITCHYARD_RUNTIME_ROOT": str(root),
                           "SWITCHYARD_RUNTIME_ARTIFACT_ID": "b" * 64}
            with patch.dict(os.environ, environment, clear=True), patch.object(
                startup.importlib.metadata, "version", return_value="0.31.0+cu129"
            ):
                with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                    startup.configure(policy)
                os.environ["SWITCHYARD_RUNTIME_ARTIFACT_ID"] = "a" * 64
                receipt["source_patches"][0]["after_sha256"] = "b" * 64
                (root / "build-info.json").write_text(json.dumps(receipt))
                with self.assertRaisesRegex(RuntimeError, "baked pinned"):
                    startup.configure(policy)
                receipt["source_patches"][0]["after_sha256"] = startup.NVML_RESULT
                (root / "build-info.json").write_text(json.dumps(receipt))
                startup.configure(policy)
                self.assertEqual(os.environ["VLLM_WORKER_MULTIPROC_METHOD"], "fork")
                self.assertEqual(os.environ["TOKENIZERS_PARALLELISM"], "false")
                self.assertEqual(os.environ["FLASHINFER_CUDA_ARCH_LIST"], "12.0")
                os.environ["FLASHINFER_CUDA_ARCH_LIST"] = "8.9"
                with self.assertRaisesRegex(RuntimeError, "architecture differs"):
                    startup.configure(policy)

    def test_required_flags_and_version_are_checked(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
            startup.importlib.metadata, "version", return_value="0.31.0"
        ):
            with self.assertRaisesRegex(RuntimeError, "strict AOT"):
                startup.configure(startup.Policy(cache_policy="required", telemetry=False))
            with self.assertRaisesRegex(RuntimeError, "startup-plan flag"):
                startup.configure(startup.Policy(memory_plan="required", telemetry=False))
        with patch.object(startup.importlib.metadata, "version", return_value="0.30.0"):
            with self.assertRaisesRegex(RuntimeError, "pinned vLLM"):
                startup.configure(startup.Policy())


class GateTests(unittest.TestCase):
    def test_import_consumer_waits_and_cancellation_wakes_it(self):
        for cancelled in [False, True]:
            finder = startup.PrerequisiteImports(startup.Policy(telemetry=False))
            entered = threading.Event()
            completed = threading.Event()
            errors = []

            def consumer():
                entered.set()
                try:
                    finder.find_spec("flashinfer.jit.env")
                except RuntimeError as error:
                    errors.append(error)
                finally:
                    completed.set()

            thread = threading.Thread(target=consumer)
            thread.start()
            self.assertTrue(entered.wait(1))
            self.assertFalse(completed.wait(0.02))
            finder.release(cancelled=cancelled)
            thread.join(1)
            self.assertFalse(thread.is_alive())
            self.assertEqual(bool(errors), cancelled)

    def test_bounded_gate_and_early_preparation_failure(self):
        for token, expected in [(startup.GATE.encode(), True), (b"", False),
                                (b"wrong\n", False), (b"x" * 1024, False)]:
            read_fd, write_fd = os.pipe()
            preparation = startup.Preparation(startup.Policy(telemetry=False))
            try:
                os.write(write_fd, token)
                os.close(write_fd)
                write_fd = None
                with os.fdopen(os.dup(read_fd), "r") as stdin, patch.object(sys, "stdin", stdin):
                    self.assertEqual(startup.wait_gate(preparation), expected)
            finally:
                os.close(read_fd)
                if write_fd is not None:
                    os.close(write_fd)
                preparation.close()
        read_fd, write_fd = os.pipe()
        preparation = startup.Preparation(startup.Policy(telemetry=False))
        try:
            with patch.object(startup.importlib, "import_module", side_effect=ImportError("missing")):
                preparation.thread.start()
                with os.fdopen(os.dup(read_fd), "r") as stdin, patch.object(sys, "stdin", stdin):
                    with self.assertRaises(ImportError):
                        startup.wait_gate(preparation)
        finally:
            preparation.thread.join(1)
            preparation.close()
            os.close(read_fd)
            os.close(write_fd)

    def test_prepared_launcher_imports_early_and_dispatches_after_gate(self):
        # Fake native modules exercise a real process and the import finder.
        # The consumer path starts absent and is published while imports wait.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ["vllm", "vllm/entrypoints", "vllm/entrypoints/cli", "vllm/v1", "vllm/v1/worker"]:
                package = root / name
                package.mkdir(parents=True, exist_ok=True)
                (package / "__init__.py").write_text("")
            (root / "torch.py").write_text("print('early native import', flush=True)\nclass cuda:\n @staticmethod\n def is_initialized(): return False\n")
            (root / "transformers.py").write_text("")
            (root / "vllm/entrypoints/cli/serve.py").write_text("import flashinfer_cubin\n")
            (root / "vllm/v1/worker/gpu_worker.py").write_text("")
            (root / "vllm/entrypoints/cli/main.py").write_text(
                "import os\ndef main():\n assert os.environ['SWITCHYARD_STARTUP_GATE_RELEASED'] == '1'\n print('native engine dispatched', flush=True)\n")
            (root / "switchyard_vllm_hooks.py").write_text("def install(policy): pass\n")
            launcher = root / "launcher.py"
            launcher.write_text("import switchyard_vllm_startup as s\ns.importlib.metadata.version = lambda _: '0.31.0'\ns.main()\n")
            late = root / "supplement"
            environment = {**os.environ, "PYTHONPATH": os.pathsep.join([str(root), str(PYTHON), str(late)]),
                           "VLLM_ENGINE_STARTUP_POLICY": json.dumps({"strategy": "prepared"}),
                           "PYTHONDONTWRITEBYTECODE": "1"}
            process = subprocess.Popen([sys.executable, str(launcher), "serve"], env=environment,
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                prefix = []
                for line in process.stdout:
                    prefix.append(line)
                    if "import_waiting_for_prerequisites" in line:
                        break
                self.assertIn("early native import", "".join(prefix))
                self.assertNotIn("native engine dispatched", "".join(prefix))
                late.mkdir()
                (late / "flashinfer_cubin.py").write_text("print('late cubin imported', flush=True)\n")
                stdout, stderr = process.communicate(startup.GATE, timeout=10)
                self.assertEqual(process.returncode, 0, stderr)
                self.assertIn("late cubin imported", stdout)
                self.assertIn("native engine dispatched", stdout)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()


class CacheTests(unittest.TestCase):
    def test_aliases_require_checked_sibling_ownership(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "selected"
            root.mkdir()
            alias = root.with_name("baked-path")
            policy = startup.Policy(cache_aliases=[str(alias)], telemetry=False)
            with patch.dict(os.environ, {"SWITCHYARD_CACHE_ROOT": str(root)}):
                startup.cache_aliases(policy)
                startup.cache_aliases(policy)
                self.assertEqual(alias.resolve(), root)
                alias.unlink()
                alias.mkdir()
                with self.assertRaisesRegex(RuntimeError, "already exists"):
                    startup.cache_aliases(policy)
                alias.rmdir()
                alias.symlink_to(root.parent)
                with self.assertRaisesRegex(RuntimeError, "another owner"):
                    startup.cache_aliases(policy)
                with self.assertRaisesRegex(ValueError, "normalized sibling"):
                    startup.cache_aliases(startup.Policy(cache_aliases=[str(root / "nested")]))

    def test_native_provider_inventory_is_refreshed_after_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider_dir = root / "provider"
            (provider_dir / "sampling").mkdir(parents=True)
            (provider_dir / "sampling/sampling.so").touch()
            cache = root / "torch_compile_cache/torch_aot_compile"
            cache.mkdir(parents=True)
            (cache / "complete").touch()
            provider = SimpleNamespace(modules=["sampling"], jit_cache_dir=provider_dir)
            module = SimpleNamespace(_get_aot_providers=lambda: [provider], FLASHINFER_AOT_DIR=root / "native")
            with patch.dict(os.environ, {"VLLM_CACHE_ROOT": str(root)}), patch.object(
                startup.importlib, "import_module", return_value=module
            ), patch.object(startup.importlib.metadata, "version", return_value="0.7.0.post1"):
                startup.refresh_providers(startup.Policy(cache_policy="required", telemetry=False))
                self.assertEqual(module.FLASHINFER_AOT_PROVIDERS, [provider])
                self.assertEqual(module.FLASHINFER_AOT_DIRS, (root / "native", provider_dir))
                (provider_dir / "sampling/sampling.so").unlink()
                with self.assertRaisesRegex(RuntimeError, "sampling provider is absent"):
                    startup.refresh_providers(startup.Policy(cache_policy="required", telemetry=False))


class TokenizerTests(unittest.TestCase):
    def test_exact_native_keys_and_concurrent_object_reuse(self):
        calls = []

        @lru_cache()
        def native(*args, **kwargs):
            calls.append((args, kwargs))
            time.sleep(0.01)
            return object()

        registry = SimpleNamespace(cached_get_tokenizer=native)
        hooks.tokenizer_cache(registry)
        base = {"runner_type": "generate", "tokenizer_mode": "auto", "revision": "v1", "trust_remote_code": False}
        with ThreadPoolExecutor(max_workers=8) as pool:
            objects = list(pool.map(lambda _: registry.cached_get_tokenizer("/model", **base), range(8)))
        self.assertTrue(all(value is objects[0] for value in objects))
        self.assertEqual(len(calls), 1)
        for key, value in [("runner_type", "pooling"), ("tokenizer_mode", "slow"),
                           ("revision", "v2"), ("trust_remote_code", True), ("padding_side", "left")]:
            result = registry.cached_get_tokenizer("/model", **{**base, key: value})
            self.assertIsNot(result, objects[0])
        self.assertIsNot(registry.cached_get_tokenizer("/other", **base), objects[0])
        self.assertIs(registry.cached_get_tokenizer._switchyard_native, native)
        self.assertEqual(registry.cached_get_tokenizer.cache_info().misses, 7)
        hooks.tokenizer_cache(registry)
        self.assertIs(registry.cached_get_tokenizer._switchyard_native, native)

    def test_native_config_skip_and_preload_errors(self):
        hooks._preloads = []
        called = []
        result = SimpleNamespace(skip_tokenizer_init=True)

        class EngineArgs:
            def create_model_config(self):
                return result

            def create_engine_config(self):
                self.create_model_config()
                return "native engine config"

        policy = startup.Policy(tokenizer_preload=True, telemetry=False)
        registry = SimpleNamespace(cached_tokenizer_from_config=lambda config: called.append(config))
        hooks.model_preparation(policy, EngineArgs, registry)
        with patch.dict(os.environ, {"SWITCHYARD_STARTUP_GATE_RELEASED": "1"}):
            self.assertEqual(EngineArgs().create_engine_config(), "native engine config")
            self.assertEqual(called, [])
            result.skip_tokenizer_init = False
            self.assertEqual(EngineArgs().create_engine_config(), "native engine config")
            self.assertEqual(called, [result])
            registry.cached_tokenizer_from_config = lambda _: (_ for _ in ()).throw(ValueError("invalid tokenizer"))
            with self.assertRaisesRegex(ValueError, "invalid tokenizer"):
                EngineArgs().create_engine_config()
        hooks._preloads = []

    def test_fork_tokenizer_has_no_owned_thread(self):
        registry = SimpleNamespace(cached_tokenizer_from_config=lambda _: None)
        preparation = hooks.TokenizerPreparation(startup.Policy(worker_method="guarded_fork", telemetry=False), object(), registry)
        with patch.dict(os.environ, {"SWITCHYARD_STARTUP_GATE_RELEASED": "1"}):
            preparation.start()
        self.assertIsNone(preparation.thread)

    def test_gate_blocks_native_config_and_memory(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(RuntimeError, "before metadata"):
            hooks.require_gate()


class NativeSourceTests(unittest.TestCase):
    """Exercise actual pinned guard functions with hardware observations stubbed."""

    @classmethod
    def setUpClass(cls):
        source = os.environ.get("SWITCHYARD_TEST_VLLM_SOURCE")
        if source is None:
            source = PYTHON.parents[2] / "results/vllm-export/runtime/venv/lib/python3.12/site-packages/vllm"
        cls.runtime = Path(source)
        if not cls.runtime.is_dir():
            raise unittest.SkipTest("pinned runtime source unavailable; set SWITCHYARD_TEST_VLLM_SOURCE")

    def functions(self, relative, names, environment):
        tree = ast.parse((self.runtime / relative).read_text())
        selected = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
        self.assertEqual(len(selected.body), len(names))
        exec(compile(selected, str(self.runtime / relative), "exec"), environment)
        return SimpleNamespace(**{name: environment[name] for name in names})

    def test_native_process_guards_retain_spawn_fallback(self):
        controls = {"cuda": False, "xpu": False, "ray": False, "wsl": False}
        environment = {"os": os, "sys": sys, "is_in_ray_actor": lambda: controls["ray"],
                       "cuda_is_initialized": lambda: controls["cuda"], "xpu_is_initialized": lambda: controls["xpu"],
                       "in_wsl": lambda: controls["wsl"], "logger": SimpleNamespace(warning=lambda *args: None)}
        system = self.functions("utils/system_utils.py", {"_maybe_force_spawn"}, environment)
        policy = startup.Policy(telemetry=False)
        hooks.process_method(policy, system)
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: controls["cuda"]))
        with patch.dict(sys.modules, {"torch": fake_torch}), patch.dict(os.environ, {"SWITCHYARD_STARTUP_GATE_RELEASED": "1"}):
            for trigger in [None, "cuda", "xpu", "wsl", "numa"]:
                os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"
                for name in controls:
                    controls[name] = name == trigger
                with patch.object(sys, "argv", ["vllm", "serve", *(["--numa-bind"] if trigger == "numa" else [])]):
                    system._maybe_force_spawn()
                self.assertEqual(os.environ["VLLM_WORKER_MULTIPROC_METHOD"], "fork" if trigger is None else "spawn")

    def test_native_memory_baseline_is_never_relaxed(self):
        state = {"plan": {"kv_cache_memory_bytes": 400, "free_memory_baseline": 900}}
        environment = {"envs": SimpleNamespace(VLLM_ENABLE_STARTUP_PLAN=True),
                       "compute_plan_fingerprint": lambda *args: "matching",
                       "_load_plan": lambda fingerprint: state["plan"],
                       "logger": SimpleNamespace(info=lambda *args: None)}
        module = self.functions("v1/worker/startup_plan.py", {"maybe_apply_startup_plan", "_applicable_kv_cache_memory_bytes"}, environment)
        hooks.memory_plan(startup.Policy(telemetry=False), module)
        worker = SimpleNamespace(cache_config=SimpleNamespace(kv_cache_memory_bytes=None),
                                 init_snapshot=SimpleNamespace(free_memory=899), vllm_config=object(), rank=0,
                                 parallel_config=SimpleNamespace(world_size=1))
        with patch.dict(os.environ, {"SWITCHYARD_STARTUP_GATE_RELEASED": "1"}):
            module.maybe_apply_startup_plan(worker)
            self.assertIsNone(worker.cache_config.kv_cache_memory_bytes)
            self.assertEqual(state["plan"]["free_memory_baseline"], 900)
            hooks.memory_plan(startup.Policy(memory_plan="required", telemetry=False), module)
            with self.assertRaisesRegex(RuntimeError, "not accepted"):
                module.maybe_apply_startup_plan(worker)
            worker.init_snapshot.free_memory = 900
            module.maybe_apply_startup_plan(worker)
            self.assertEqual(worker.cache_config.kv_cache_memory_bytes, 400)
            worker.cache_config.kv_cache_memory_bytes = None
            state["plan"] = None
            with self.assertRaisesRegex(RuntimeError, "not accepted"):
                module.maybe_apply_startup_plan(worker)

    def test_native_tokenizer_config_supplies_every_key(self):
        calls = []
        environment = {"_maybe_register_hf_config": lambda config: None,
                       "cached_get_tokenizer": lambda *args, **kwargs: calls.append((args, kwargs))}
        # The annotation belongs to native vLLM; stub only its import for this
        # source-level CPU check and keep the native call expression intact.
        environment["ModelConfig"] = object
        module = self.functions("tokenizers/registry.py", {"cached_tokenizer_from_config"}, environment)
        config = SimpleNamespace(skip_tokenizer_init=False, hf_config=object(), tokenizer="/selected",
                                 runner_type="generate", tokenizer_mode="slow", tokenizer_revision="revision",
                                 trust_remote_code=True)
        module.cached_tokenizer_from_config(config, padding_side="left")
        self.assertEqual(calls, [(("/selected",), {"runner_type": "generate", "tokenizer_mode": "slow",
                         "revision": "revision", "trust_remote_code": True, "padding_side": "left"})])
        config.skip_tokenizer_init = True
        module.cached_tokenizer_from_config(config)
        self.assertEqual(len(calls), 1)

    def test_plugin_observer_precedes_native_worker_device_initialization(self):
        source = ast.parse((PYTHON / "switchyard_extent_loader.py").read_text())
        register = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "register")
        install_line = next(node.lineno for node in ast.walk(register)
                            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "install")
        loader_line = next(node.lineno for node in ast.walk(register) if isinstance(node, ast.ClassDef))
        self.assertLess(install_line, loader_line)
        # Native WorkerWrapperBase loads general plugins before resolving the
        # GPU Worker class. The hook's import also updates a prepared alias.
        native_source = (self.runtime / "v1/worker/worker_base.py").read_text()
        self.assertLess(native_source.index("        load_general_plugins()"),
                        native_source.index("            worker_class: type[WorkerBase] = resolve_obj_by_qualname"))



if __name__ == "__main__":
    unittest.main()
