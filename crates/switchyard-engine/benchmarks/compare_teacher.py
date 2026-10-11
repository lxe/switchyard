"""Compare next-token choices and log probabilities under an identical history."""
import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-logprob-error', type=float)
    parser.add_argument('--require-argmax', action='store_true')
    args = parser.parse_args()
    candidate = json.loads(args.candidate.read_text())
    reference = [json.loads(line) for line in args.reference.read_text().splitlines()]
    seen, errors, mismatches = set(), [], []
    for expected in reference:
        row, step = expected['row'], expected['step']
        key = (row, step)
        if key in seen:
            raise ValueError('Duplicate reference position')
        seen.add(key)
        actual = candidate[row][step]
        if actual['expected'] != expected['expected']:
            raise ValueError('Teacher histories differ')
        error = abs(actual['expected_logprob'] - expected['expected_logprob'])
        if not math.isfinite(error):
            raise ValueError('Nonfinite log-probability error')
        errors.append(error)
        if actual['top'][0]['token'] != expected['argmax']:
            mismatches.append({'row':row, 'step':step, 'reference':expected['argmax'],
                'candidate':actual['top'][0]['token'], 'reference_top_margin':
                    expected['top'][0]['logit']-expected['top'][1]['logit'],
                'logprob_error':error})
    positions = {(row, step) for row, values in enumerate(candidate) for step in range(len(values))}
    if not seen or seen != positions:
        raise ValueError('Reference and candidate positions must exactly agree')
    maximum = max(errors)
    passed = None
    if args.require_argmax or args.max_logprob_error is not None:
        passed = (not args.require_argmax or not mismatches) and (
            args.max_logprob_error is None or maximum <= args.max_logprob_error)
    result = {'positions':len(seen), 'argmax_matches':len(seen)-len(mismatches),
        'mean_abs_expected_logprob_error':sum(errors)/len(errors),
        'max_abs_expected_logprob_error':maximum, 'mismatches':mismatches,
        'contract':{'require_argmax':args.require_argmax, 'max_logprob_error':args.max_logprob_error},
        'passed':passed}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'mismatches'}))
    raise SystemExit(1 if passed is False else 0)


if __name__ == '__main__':
    main()
