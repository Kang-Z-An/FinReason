"""Binary FinQA execution reward with bounded DSL validation, no generated Python."""
import hashlib
import math
from pathlib import Path

from .analyze_run import official_functions
from .executor import execute

OFFICIAL_SHA = '845cd131cab843eceff256cf6d392978cc470a7da4a80107beb56027fdca5c13'
SWIFT_EMPTY_THINK_PREFIX = '<think>\n\n</think>\n\n'


def normalize_swift_completion(completion):
    """Undo exactly one known response prefix reattached by Swift's decoder.

    The core DSL scorer stays strict. Nonempty thinking, other wrappers and a
    second prefix are not removed. The generation prompt already contains this
    empty prefix; Swift decode_generate_ids prepends it to generated token text.
    """
    if not isinstance(completion, str):
        raise ValueError('Completion must be text')
    removed = completion.startswith(SWIFT_EMPTY_THINK_PREFIX)
    return (completion[len(SWIFT_EMPTY_THINK_PREFIX):] if removed else completion), removed


class FinQAReward:
    def __init__(self, official_path):
        if hashlib.sha256(Path(official_path).read_bytes()).hexdigest() != OFFICIAL_SHA:
            raise ValueError('Official evaluator changed; review before use')
        self.official = official_functions(official_path)

    def evaluate(self, program, table, gold_answer):
        if not isinstance(gold_answer, (float, int, str)) or isinstance(gold_answer, bool):
            raise ValueError('Invalid reward-only gold answer')
        if isinstance(gold_answer, str):
            if gold_answer not in ('yes', 'no'):
                raise ValueError('Unknown categorical gold answer')
        elif not math.isfinite(gold_answer):
            raise ValueError('Nonfinite gold answer')
        # Independent bounded parser blocks nonfinite operands, huge exponent,
        # unknown operators, forward references and non-DSL text before upstream.
        safe = execute(program, table)
        if not safe.valid:
            return {'reward': 0.0, 'safe_valid': False, 'reason': safe.error,
                    'official_valid': None, 'official_answer': None}
        try:
            invalid, answer = self.official['eval_program'](
                self.official['program_tokenization'](program), table)
            finite = isinstance(answer, str) and answer in ('yes', 'no') or \
                isinstance(answer, (float, int)) and math.isfinite(answer)
            valid = invalid == 0 and finite
            return {'reward': float(valid and answer == gold_answer), 'safe_valid': True,
                    'official_valid': valid, 'official_answer': answer if finite else None,
                    'reason': '' if valid else 'official_invalid_or_nonfinite'}
        except Exception as exc:
            return {'reward': 0.0, 'safe_valid': True, 'official_valid': False,
                    'official_answer': None, 'reason': type(exc).__name__}


def score_batch(reward, completions, ids, registry):
    if len(completions) != len(ids):
        raise ValueError('Completion/id length mismatch')
    results = []
    for completion, id_ in zip(completions, ids):
        if id_ not in registry:
            raise ValueError('Missing reward-only training metadata: ' + str(id_))
        row = registry[id_]
        results.append(reward.evaluate(completion, row['table'], row['gold_answer']))
    return results
