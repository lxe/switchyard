"""Prepare mixed-length traces and compare merged, independent and published F32 probes."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def prepare(args):
    source = json.loads(args.source.read_text())
    config = json.loads(args.config.read_text())
    if len(source) < 3 or config['dtype'] != 'f32' or config['max_context'] < 112:
        raise ValueError('Need three source traces and F32 context capacity at least 112')
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    traces = []
    # Last three rows repeat the same prompt histories. The longest histories
    # finish first, forcing KV cropping while shorter rows continue decoding.
    for row, (length, count) in enumerate(zip([8, 32, 96, 8, 32, 96], [16, 12, 4, 10, 8, 2])):
        trace = source[row % 3]
        prompt, continuation = trace['prompt'], trace['continuation']
        if not prompt or not continuation:
            raise ValueError('Empty source history')
        traces.append({
            'prompt': (prompt * ((length + len(prompt) - 1) // len(prompt)))[:length],
            'continuation': (continuation * ((count + len(continuation) - 1) // len(continuation)))[:count],
        })
    if config['max_batch_size'] < len(traces):
        raise ValueError('Gate requires all six rows in one admitted batch')
    (output / 'traces.json').write_text(json.dumps(traces, indent=2) + '\n')
    requests = [{'prompt': t['prompt'], 'max_tokens': len(t['continuation']),
                 'ignore_eos': True} for t in traces]
    (output / 'requests.json').write_text(json.dumps(requests, indent=2) + '\n')
    (output / 'f32.config.json').write_text(json.dumps(config, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'prompt_lengths': [len(t['prompt']) for t in traces],
                      'continuation_lengths': [len(t['continuation']) for t in traces]}))


def compare(args):
    args.output.mkdir(parents=True, exist_ok=True)
    candidate = json.loads(args.candidate.read_text())
    independent = json.loads(args.independent.read_text())
    traces_path = args.traces or args.candidate.with_name('traces.json')
    traces = json.loads(traces_path.read_text())
    if len(candidate) != len(independent) or len(candidate) != len(traces) or not candidate:
        raise ValueError('Probe row counts differ or are empty')
    reference = []
    for row, (actual, expected, trace) in enumerate(zip(candidate, independent, traces)):
        if len(actual) != len(expected) or len(actual) != len(trace['continuation']) or not actual:
            raise ValueError('Probe step counts differ or are empty')
        for step, (a, e) in enumerate(zip(actual, expected)):
            position = 0 if step == 0 else len(trace['prompt']) + step - 1
            if (a['position'] != e['position'] or a['expected'] != e['expected']
                    or a['position'] != position or a['expected'] != trace['continuation'][step]):
                raise ValueError('Independent and merged histories differ')
            reference.append({'row': row, 'step': step, 'expected': e['expected'],
                              'argmax': e['top'][0]['token'],
                              'expected_logprob': e['expected_logprob'], 'top': e['top']})
    native = args.output / 'independent.jsonl'
    native.write_text(''.join(json.dumps(row) + '\n' for row in reference))
    comparator = Path(__file__).with_name('compare_teacher.py')
    reports = {}
    for name, path in [('independent', native), ('published', args.reference)]:
        report = args.output / (name + '-comparison.json')
        report.unlink(missing_ok=True)
        process = subprocess.run([sys.executable, str(comparator), '--candidate', str(args.candidate),
                        '--reference', str(path), '--output', str(report), '--require-argmax',
                        '--max-logprob-error', '0.0001'], check=False)
        if process.returncode not in (0, 1) or not report.exists():
            raise RuntimeError('Comparison failed before writing a validated report')
        reports[name] = json.loads(report.read_text())
    requests = [{'prompt': t['prompt'], 'max_tokens': len(t['continuation']),
                 'ignore_eos': True} for t in traces]
    result = {'passed': all(r['passed'] is True for r in reports.values()), 'comparisons': reports,
              'requests': requests, 'inputs_sha256': {
                  name: hashlib.sha256(path.read_bytes()).hexdigest()
                  for name, path in [('traces', traces_path), ('merged', args.candidate),
                                     ('independent', args.independent), ('published', args.reference)]}}
    (args.output / 'gate.json').write_text(json.dumps(result, indent=2) + '\n')
    raise SystemExit(0 if result['passed'] else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--source', type=Path, required=True)
    prep.add_argument('--config', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    check = commands.add_parser('compare')
    check.add_argument('--candidate', type=Path, required=True)
    check.add_argument('--independent', type=Path, required=True)
    check.add_argument('--reference', type=Path, required=True)
    check.add_argument('--traces', type=Path, help='Defaults to traces.json beside candidate')
    check.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
    else:
        compare(args)


if __name__ == '__main__':
    main()
