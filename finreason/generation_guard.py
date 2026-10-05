"""Audit the resolver reached by an actual Transformers generate call."""


def generation_policy(config, tokenizer, model_bos_token_id):
    return dict(use_model_defaults=False, do_sample=config['do_sample'],
        temperature=config['temperature'], top_p=config['top_p'], top_k=config['top_k'],
        max_new_tokens=config['max_new_tokens'], num_return_sequences=1, num_beams=1,
        repetition_penalty=1.0, use_cache=True,
        eos_token_id=list(dict.fromkeys([tokenizer.eos_token_id, tokenizer.pad_token_id])),
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id if tokenizer.bos_token_id is not None else model_bos_token_id)


def install_guard(base_model, expected):
    original = base_model._prepare_generation_config
    calls = []

    def guarded(generation_config, use_model_defaults=None, **kwargs):
        if use_model_defaults is not False:
            raise ValueError('Model default fallback must be disabled')
        effective, remaining = original(generation_config, use_model_defaults=False, **kwargs)
        actual = {k: getattr(effective, k) for k in expected if k != 'use_model_defaults'}
        actual['use_model_defaults'] = False
        if actual != expected:
            raise ValueError('Effective generation policy changed')
        calls.append(actual)
        return effective, remaining

    base_model._prepare_generation_config = guarded
    return calls
