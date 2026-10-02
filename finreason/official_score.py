"""Unmodified v3 official execution scoring function."""
def score_one(program, table, gold_answer, official):
    try:
        invalid, answer = official['eval_program'](official['program_tokenization'](program), table)
        valid = invalid == 0
        return {'valid': valid, 'correct': valid and answer == gold_answer,
                'error': '' if valid else 'official_invalid'}
    except Exception as exc:
        return {'valid': False, 'correct': False, 'error': type(exc).__name__}
