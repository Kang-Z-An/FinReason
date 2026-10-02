"""Original v3 prediction-boundary checks."""
from .infer import item_seed

def validate_predictions(rows, run, progress, ids, config, entry):
    if run['stage'] != 'rlvr_v3_prospective_final_fp16_inference' or run['mode'] != entry['tag']:
        raise ValueError('Unexpected final run identity')
    if run['config'] != config or run['labels_on_remote'] is not False:
        raise ValueError('Generation configuration or label boundary changed')
    if (run['variant'], run['training_seed']) != (entry['variant'], entry['training_seed']):
        raise ValueError('Reward or training seed differs')
    if progress['state'] != 'complete' or progress['completed_total'] != len(ids) or \
            progress['dataset_total'] != len(ids) or progress['failed_id'] is not None:
        raise ValueError('Final predictions incomplete')
    if [r['id'] for r in rows] != ids or len(set(ids)) != len(ids):
        raise ValueError('Final prediction IDs, count or order changed')
    if any(r['seed'] != item_seed(config['seed'], r['id']) for r in rows):
        raise ValueError('Per-item random seeds changed')
