//! Native LoRA engines owned by the inference execution queue.
//!
//! Admission borrows an `Arc<AdapterHandle>` from the shared catalog and retains
//! it through queued and active inference. An adapter can only be removed when
//! the catalog is its sole owner. Engine construction and destruction run on the
//! same execution owner as inference; catalog locks never cover tensor transfer.
use crate::{
    Error, Result, config::Config, engine::Engine, models::lora::Adapter, telemetry::Telemetry,
};
use serde::Serialize;
use std::{
    collections::BTreeMap,
    sync::{Arc, RwLock},
    time::Instant,
};

/// A request lease identifies a particular loaded revision, never just its name.
#[derive(Debug, Serialize)]
pub struct AdapterHandle {
    pub id: u64,
    pub name: String,
    pub revision: String,
    pub rank: usize,
    /// Device-resident adapter tensor bytes, excluding shared base model weights.
    pub bytes: u64,
}

pub type SharedCatalog = Arc<RwLock<BTreeMap<String, Arc<AdapterHandle>>>>;
pub type RegistryResult<T> = std::result::Result<T, RegistryError>;

#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    #[error("Adapter {0} is not loaded")]
    NotFound(String),
    #[error("Adapter {0} has queued or active request leases")]
    InUse(String),
    #[error("Adapter {0} is already loaded")]
    AlreadyLoaded(String),
    #[error(transparent)]
    Invalid(#[from] Error),
}

/// Stores engines on one inference owner; it never allocates cloud hosts.
/// `max_loaded` counts adapters only. The immutable base always occupies ID zero.
pub struct EngineRegistry {
    engines: BTreeMap<u64, Engine>,
    catalog: SharedCatalog,
    config: Config,
    next_id: u64,
    resident_bytes: u64,
    telemetry: Telemetry,
}

impl EngineRegistry {
    pub fn new(base: Engine, catalog: SharedCatalog, config: Config) -> Result<Self> {
        let telemetry = base.telemetry();
        {
            let mut names = catalog
                .write()
                .map_err(|_| Error::Config("Adapter catalog lock was poisoned".into()))?;
            if names.is_empty() {
                names.insert(
                    config.model.clone(),
                    Arc::new(AdapterHandle {
                        id: 0,
                        name: config.model.clone(),
                        revision: "base".into(),
                        rank: 0,
                        bytes: 0,
                    }),
                );
            } else if names.len() != 1
                || names.get(&config.model).is_none_or(|handle| {
                    handle.id != 0
                        || handle.name != config.model
                        || handle.rank != 0
                        || handle.bytes != 0
                })
            {
                return Err(Error::Config(
                    "Adapter registry requires an empty or matching base-only catalog".into(),
                ));
            }
        }
        Ok(Self {
            engines: BTreeMap::from([(0, base)]),
            catalog,
            config,
            next_id: 1,
            resident_bytes: 0,
            telemetry,
        })
    }

    pub fn engine(&self, id: u64) -> Result<&Engine> {
        self.engines
            .get(&id)
            .ok_or_else(|| Error::Config(format!("No engine for adapter identity {id}")))
    }

    pub fn load(&mut self, name: String, adapter: &Adapter) -> RegistryResult<Arc<AdapterHandle>> {
        let began = Instant::now();
        if name.is_empty() || name.len() > 128 || name.chars().any(char::is_control) {
            return Err(Error::Config(
                "Adapter name must contain 1 to 128 bytes without control characters".into(),
            )
            .into());
        }
        let limits = self
            .config
            .lora
            .as_ref()
            .ok_or_else(|| Error::Config("Runtime LoRA loading is disabled".into()))?;
        if self
            .catalog
            .read()
            .map_err(|_| Error::Config("Adapter catalog lock was poisoned".into()))?
            .contains_key(&name)
        {
            return Err(RegistryError::AlreadyLoaded(name));
        }
        if self.engines.len().saturating_sub(1) >= limits.max_loaded {
            return Err(
                Error::Config("Loaded adapter count exceeds its configured bound".into()).into(),
            );
        }
        if adapter.base_model_name() != self.config.model
            && !limits
                .base_model_aliases
                .iter()
                .any(|alias| alias == adapter.base_model_name())
        {
            return Err(Error::Config(
                "Adapter base model does not match the served model or its configured aliases"
                    .into(),
            )
            .into());
        }
        if adapter.rank() == 0
            || adapter.rank() > limits.max_rank
            || adapter.bytes() > limits.max_upload_bytes
        {
            return Err(Error::Config(
                "Adapter rank or CPU tensor bytes exceed their configured bounds".into(),
            )
            .into());
        }
        let base = self.engine(0)?;
        let bytes = adapter.allocated_bytes(base.adapter_dtype())?;
        let resident = self
            .resident_bytes
            .checked_add(bytes)
            .filter(|&total| total <= limits.max_resident_bytes)
            .ok_or_else(|| {
                Error::Config("Adapter device-resident bytes exceed their configured bound".into())
            })?;
        let id = self.next_id;
        let next_id = id
            .checked_add(1)
            .ok_or_else(|| Error::Config("Adapter identity space is exhausted".into()))?;
        // All count, rank and memory bounds precede device tensor conversion.
        let validation_s = began.elapsed().as_secs_f64();
        let construction = Instant::now();
        let engine = base.with_lora(adapter)?;
        let construction_s = construction.elapsed().as_secs_f64();
        let handle = Arc::new(AdapterHandle {
            id,
            name: name.clone(),
            revision: adapter.revision().to_owned(),
            rank: adapter.rank(),
            bytes,
        });
        {
            let mut catalog = self
                .catalog
                .write()
                .map_err(|_| Error::Config("Adapter catalog lock was poisoned".into()))?;
            if catalog.contains_key(&name) {
                return Err(RegistryError::AlreadyLoaded(name));
            }
            self.engines.insert(id, engine);
            catalog.insert(name.clone(), handle.clone());
        }
        self.next_id = next_id;
        self.resident_bytes = resident;
        self.telemetry.event(
            "adapter_loaded",
            serde_json::json!({
                "id":id,"name":name,"revision":handle.revision,"rank":handle.rank,
                "cpu_tensor_bytes":adapter.bytes(),"device_tensor_bytes":bytes,
                "resident_adapter_bytes":resident,"validation_s":validation_s,"construction_s":construction_s,
                "elapsed_s":began.elapsed().as_secs_f64()
            }),
        );
        Ok(handle)
    }

    /// Resolve numeric identity on the execution owner, so a stale unload cannot
    /// remove a newly loaded revision that reused the same public name.
    pub fn unload_id(&mut self, id: u64) -> RegistryResult<()> {
        let name = self
            .catalog
            .read()
            .map_err(|_| Error::Config("Adapter catalog lock was poisoned".into()))?
            .values()
            .find(|handle| handle.id == id)
            .map(|handle| handle.name.clone())
            .ok_or_else(|| RegistryError::NotFound(id.to_string()))?;
        self.unload(&name)
    }

    pub fn unload(&mut self, name: &str) -> RegistryResult<()> {
        let began = Instant::now();
        let mut catalog = self
            .catalog
            .write()
            .map_err(|_| Error::Config("Adapter catalog lock was poisoned".into()))?;
        let handle = catalog
            .get(name)
            .ok_or_else(|| RegistryError::NotFound(name.into()))?;
        if handle.id == 0 {
            return Err(Error::Config("The base model cannot be unloaded".into()).into());
        }
        if Arc::strong_count(handle) != 1 {
            return Err(RegistryError::InUse(name.into()));
        }
        let id = handle.id;
        let bytes = handle.bytes;
        // Holding the write lock prevents a new admission lease between the
        // last-owner check and removal of the corresponding device engine.
        let engine = self
            .engines
            .remove(&id)
            .ok_or_else(|| Error::Config("Adapter catalog has no matching engine".into()))?;
        catalog.remove(name);
        drop(catalog);
        drop(engine);
        self.resident_bytes = self.resident_bytes.saturating_sub(bytes);
        self.telemetry.event("adapter_unloaded", serde_json::json!({
            "id":id,"name":name,"device_tensor_bytes":bytes,
            "resident_adapter_bytes":self.resident_bytes,"elapsed_s":began.elapsed().as_secs_f64()
        }));
        Ok(())
    }

    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{ModelFactory, ModelSession};
    use candle_core::{DType, Device, Tensor};
    use std::{
        collections::HashMap,
        path::PathBuf,
        sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    };

    #[derive(Clone)]
    struct Factory {
        device: Device,
        constructions: Arc<AtomicUsize>,
        fail: Arc<AtomicBool>,
    }
    impl ModelFactory for Factory {
        fn with_lora(&self, _: &Adapter) -> Result<Box<dyn ModelFactory>> {
            self.constructions.fetch_add(1, Ordering::Relaxed);
            if self.fail.load(Ordering::Relaxed) {
                return Err(Error::Config(
                    "Fixture model rejected the adapter target".into(),
                ));
            }
            Ok(Box::new(self.clone()))
        }
        fn session(&self) -> Box<dyn ModelSession> {
            Box::new(Session)
        }
        fn device(&self) -> &Device {
            &self.device
        }
        fn vocab_size(&self) -> usize {
            2
        }
        fn eos_token(&self) -> u32 {
            1
        }
    }
    struct Session;
    impl ModelSession for Session {
        fn forward(&mut self, tokens: &Tensor, _: usize) -> Result<Tensor> {
            Ok(Tensor::zeros(
                (tokens.dim(0)?, 2),
                DType::F32,
                &Device::Cpu,
            )?)
        }
    }

    struct Fixture(PathBuf);
    impl Fixture {
        fn new(base: &str) -> Self {
            static ID: AtomicU64 = AtomicU64::new(0);
            let directory = std::env::temp_dir().join(format!(
                "switchyard-adapter-registry-{}-{}",
                std::process::id(),
                ID.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&directory).unwrap();
            let config = serde_json::json!({"peft_type":"LORA","base_model_name_or_path":base,"r":1,"lora_alpha":1.0,"target_modules":["q_proj"],"bias":"none"});
            std::fs::write(
                directory.join("adapter_config.json"),
                serde_json::to_vec(&config).unwrap(),
            )
            .unwrap();
            let tensors = HashMap::from([
                (
                    "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight",
                    Tensor::new(&[[0.25_f32, -0.5]], &Device::Cpu).unwrap(),
                ),
                (
                    "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight",
                    Tensor::new(&[[0.75_f32], [1.0]], &Device::Cpu).unwrap(),
                ),
            ]);
            candle_core::safetensors::save(&tensors, directory.join("adapter_model.safetensors"))
                .unwrap();
            Self(directory)
        }
        fn read(&self) -> Adapter {
            Adapter::read(&self.0, 1 << 20, 16, "").unwrap()
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn config() -> Config {
        serde_json::from_value(serde_json::json!({
            "model":"served-base","model_root":"/fixture","device":{"kind":"cpu"},"dtype":"f32","max_context":16,"max_weight_bytes":1024,"max_batch_size":2,"queue_capacity":4,"batch_wait_ms":0,
            "lora":{"root":"/fixture/adapters","base_model_aliases":["fixture/base"],"max_loaded":2,"max_rank":8,"max_upload_bytes":1048576,"max_resident_bytes":1024,"transfer_timeout_s":30}
        })).unwrap()
    }
    fn registry(config: Config) -> (EngineRegistry, SharedCatalog, Factory) {
        let factory = Factory {
            device: Device::Cpu,
            constructions: Arc::default(),
            fail: Arc::default(),
        };
        let engine = Engine::new(
            config.clone(),
            Box::new(factory.clone()),
            Telemetry::default(),
        )
        .unwrap();
        let catalog = SharedCatalog::default();
        (
            EngineRegistry::new(engine, catalog.clone(), config).unwrap(),
            catalog,
            factory,
        )
    }

    #[test]
    fn admission_leases_prevent_unload_and_reloading_uses_a_new_identity() {
        let fixture = Fixture::new("fixture/base");
        let adapter = fixture.read();
        let (mut registry, catalog, _) = registry(config());
        assert!(registry.engine(0).is_ok());
        assert!(registry.unload("served-base").is_err());
        let loaded = registry.load("example".into(), &adapter).unwrap();
        let first_id = loaded.id;
        assert_eq!(loaded.rank, 1);
        assert_eq!(loaded.bytes, 16);
        assert_eq!(loaded.revision, adapter.revision());
        assert!(matches!(
            registry.unload("example"),
            Err(RegistryError::InUse(_))
        ));
        drop(loaded);
        let queued = catalog.read().unwrap()["example"].clone();
        assert!(matches!(
            registry.unload("example"),
            Err(RegistryError::InUse(_))
        ));
        assert!(registry.engine(first_id).is_ok());
        drop(queued);
        registry.unload("example").unwrap();
        assert!(registry.engine(first_id).is_err());
        assert_eq!(registry.resident_bytes(), 0);
        assert_eq!(catalog.read().unwrap().len(), 1);
        let replacement = registry.load("example".into(), &adapter).unwrap();
        assert!(replacement.id > first_id);
        assert!(matches!(
            registry.unload_id(first_id),
            Err(RegistryError::NotFound(_))
        ));
        assert!(registry.engine(replacement.id).is_ok());
    }

    #[test]
    fn duplicate_names_and_wrong_base_never_construct_a_device_adapter() {
        let fixture = Fixture::new("fixture/base");
        let adapter = fixture.read();
        let (mut registry, _, factory) = registry(config());
        let handle = registry.load("example".into(), &adapter).unwrap();
        assert!(matches!(
            registry.load("example".into(), &adapter),
            Err(RegistryError::AlreadyLoaded(_))
        ));
        assert!(matches!(
            registry.load("served-base".into(), &adapter),
            Err(RegistryError::AlreadyLoaded(_))
        ));
        let wrong = Fixture::new("unrelated/base");
        assert!(registry.load("wrong".into(), &wrong.read()).is_err());
        assert_eq!(factory.constructions.load(Ordering::Relaxed), 1);
        assert_eq!(registry.resident_bytes(), handle.bytes);
        assert!(matches!(
            registry.unload("missing"),
            Err(RegistryError::NotFound(_))
        ));
    }

    #[test]
    fn explicit_count_rank_and_memory_limits_precede_device_construction() {
        let fixture = Fixture::new("fixture/base");
        let adapter = fixture.read();
        for (field, bound) in [
            ("max_loaded", 0),
            ("max_rank", 0),
            ("max_upload_bytes", 15),
            ("max_resident_bytes", 15),
        ] {
            let mut value = serde_json::to_value(config()).unwrap();
            value["lora"][field] = serde_json::json!(bound);
            let (mut registry, catalog, factory) = registry(serde_json::from_value(value).unwrap());
            assert!(
                registry.load("example".into(), &adapter).is_err(),
                "{field}"
            );
            assert_eq!(factory.constructions.load(Ordering::Relaxed), 0, "{field}");
            assert_eq!(registry.resident_bytes(), 0);
            assert_eq!(catalog.read().unwrap().len(), 1);
        }
        let mut config = config();
        config.lora.as_mut().unwrap().max_resident_bytes = 24;
        let (mut registry, _, factory) = registry(config);
        let _first = registry.load("first".into(), &adapter).unwrap();
        assert!(registry.load("second".into(), &adapter).is_err());
        assert_eq!(factory.constructions.load(Ordering::Relaxed), 1);
        assert_eq!(registry.resident_bytes(), 16);
    }

    #[test]
    fn disabled_loading_and_failed_construction_leave_the_catalog_unchanged() {
        let fixture = Fixture::new("fixture/base");
        let adapter = fixture.read();
        let mut disabled = config();
        disabled.lora = None;
        let (mut registry, catalog, factory) = registry(disabled);
        assert!(registry.load("example".into(), &adapter).is_err());
        assert_eq!(factory.constructions.load(Ordering::Relaxed), 0);
        assert_eq!(catalog.read().unwrap().len(), 1);
        let (mut registry, catalog, factory) = self::registry(config());
        factory.fail.store(true, Ordering::Relaxed);
        assert!(registry.load("example".into(), &adapter).is_err());
        assert_eq!(registry.resident_bytes(), 0);
        assert_eq!(catalog.read().unwrap().len(), 1);
        factory.fail.store(false, Ordering::Relaxed);
        let handle = registry.load("example".into(), &adapter).unwrap();
        assert_eq!(handle.id, 1);
    }

    #[test]
    fn resident_bytes_charge_f32_adapter_matrices_with_a_bf16_base() {
        let fixture = Fixture::new("fixture/base");
        let adapter = fixture.read();
        let mut config = config();
        config.dtype = crate::config::Precision::Bf16;
        config.lora.as_mut().unwrap().max_resident_bytes = 16;
        let (mut registry, _, _) = registry(config);
        let loaded = registry.load("example".into(), &adapter).unwrap();
        assert_eq!(loaded.bytes, 16);
        assert_eq!(registry.resident_bytes(), 16);
    }
}
