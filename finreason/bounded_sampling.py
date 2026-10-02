"""Bounded online group selection; never drops the final allowed retry."""


def count_response_tokens(token_ids):
    # Single-turn Transformers output is flat despite GRPOSample's nested annotation.
    if all(isinstance(x, int) for x in token_ids):
        return len(token_ids)
    if all(isinstance(x, list) and all(isinstance(t, int) for t in x) for x in token_ids):
        return sum(map(len, token_ids))
    raise ValueError('Unknown response token representation')


def select_group(initial, sample_next, is_mixed, max_resamples):
    if not isinstance(max_resamples, int) or isinstance(max_resamples, bool) or max_resamples < 0:
        raise ValueError('max_resamples must be a nonnegative integer')
    current = initial
    for attempt in range(max_resamples + 1):
        if is_mixed(current):
            return current, {'resamples': attempt, 'fallback_zero_signal': False,
                             'groups_generated': attempt + 1}
        if attempt < max_resamples:
            current = sample_next()
    # Keep the original zero-variance batch for framework shape/step semantics.
    # Its task advantage is zero; optimizer momentum can still move parameters.
    return initial, {'resamples': max_resamples, 'fallback_zero_signal': True,
                     'groups_generated': max_resamples + 1}
