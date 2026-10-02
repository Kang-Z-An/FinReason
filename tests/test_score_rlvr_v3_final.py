import copy
import unittest

from finreason.infer import item_seed
from finreason.statistics import cluster_intervals, paired_counts
from finreason.prediction_contract import validate_predictions


class FinalScorerTest(unittest.TestCase):
    def test_paired_gains_include_regressions(self):
        rows = [dict(a=dict(correct=a, valid=True), b=dict(correct=b, valid=True))
                for a, b in [(True, False), (False, True), (True, True)]]
        result = paired_counts(rows, 'a', 'b')
        self.assertEqual(result['correct']['left_only'], 1)
        self.assertEqual(result['correct']['right_only'], 1)
        self.assertEqual(result['correct']['net'], 0)

    def test_cluster_bootstrap_preserves_unequal_document_sizes_and_pairing(self):
        rows = [{'id': 'A/2020/page_1.pdf-1',
                 'a': dict(correct=True, valid=True), 'b': dict(correct=False, valid=True)}]
        rows += [{'id': f'B/2021/page_2.pdf-{i}',
                  'a': dict(correct=False, valid=True), 'b': dict(correct=False, valid=True)}
                 for i in range(3)]
        result = cluster_intervals(rows, ['a', 'b'], {'gain': ('a', 'b'), 'self': ('a', 'a')}, 1000, 42)
        self.assertEqual(result['documents'], 2)
        self.assertEqual(result['paired_delta_percentage_points']['self']['correct'], [0, 0])
        self.assertEqual(result['paired_delta_percentage_points']['gain']['valid'], [0, 0])
        # Possible document draws produce 0%, 25%, 100%, rather than an unweighted 50%.
        self.assertEqual(result['rates_percent']['a']['correct'], [0, 100])

    def test_reject_incomplete_reordered_and_seed_changed_predictions(self):
        ids = ['A/2020/page_1.pdf-1', 'B/2021/page_1.pdf-1']
        config = {'seed': 42}
        entry = {'tag': 'sft', 'variant': None, 'training_seed': None}
        run = dict(stage='rlvr_v3_prospective_final_fp16_inference', mode='sft', config=config,
                   labels_on_remote=False, variant=None, training_seed=None)
        progress = dict(state='complete', completed_total=2, dataset_total=2, failed_id=None)
        rows = [dict(id=i, seed=item_seed(42, i)) for i in ids]
        validate_predictions(rows, run, progress, ids, config, entry)
        with self.assertRaises(ValueError):
            validate_predictions(rows[::-1], run, progress, ids, config, entry)
        incomplete = dict(progress, completed_total=1)
        with self.assertRaises(ValueError):
            validate_predictions(rows, run, incomplete, ids, config, entry)
        altered = copy.deepcopy(rows)
        altered[0]['seed'] += 1
        with self.assertRaises(ValueError):
            validate_predictions(altered, run, progress, ids, config, entry)


if __name__ == '__main__':
    unittest.main()
