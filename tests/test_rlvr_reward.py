import unittest
from finreason.rlvr_reward import FinQAReward, normalize_swift_completion, score_batch


class RewardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reward = FinQAReward('third_party/finqa/evaluate.py')

    def test_correct_equivalent_and_multistep(self):
        for program in ('subtract(125, 100)', 'add(20, 5)',
                        'subtract(125, 100), divide(#0, const_100)'):
            gold = .25 if 'divide' in program else 25
            self.assertEqual(self.reward.evaluate(program, [], gold)['reward'], 1)

    def test_percent_and_units_wrong_get_zero(self):
        self.assertEqual(self.reward.evaluate('divide(25, 100)', [], 25)['reward'], 0)
        self.assertEqual(self.reward.evaluate('add(25%, 0)', [], .25)['reward'], 1)

    def test_invalid_nonfinite_and_unsafe_are_zero(self):
        for program in ('', '25', 'divide(#0, 2)', 'divide(1, 0)', 'add(inf, 0)',
                        'add(nan, 0)', 'exp(2, 1000000)', '__import__(os, none)',
                        'add(1, 2)'*4000):
            with self.subTest(program=program[:30]):
                self.assertEqual(self.reward.evaluate(program, [], 25)['reward'], 0)

    def test_table_and_boolean(self):
        self.assertEqual(self.reward.evaluate('table_sum(revenue, none)', [['revenue', '10', '20']], 30)['reward'], 1)
        self.assertEqual(self.reward.evaluate('greater(2, 1)', [], 'yes')['reward'], 1)

    def test_metadata_errors_fail_instead_of_silent_zero(self):
        with self.assertRaisesRegex(ValueError, 'Missing'):
            score_batch(self.reward, ['add(1, 2)'], ['unknown'], {})
        with self.assertRaisesRegex(ValueError, 'length'):
            score_batch(self.reward, ['add(1, 2)'], [], {})
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            self.reward.evaluate('add(1, 2)', [], float('nan'))

    def test_answer_shortcut_is_explicit_known_risk(self):
        # Execution-only rewards cannot prove operand provenance. Do not claim otherwise.
        self.assertEqual(self.reward.evaluate('add(25, 0)', [], 25)['reward'], 1)

    def test_only_framework_empty_prefix_is_removed(self):
        prefix = '<think>\n\n</think>\n\n'
        program, removed = normalize_swift_completion(prefix+'add(25, 0)')
        self.assertTrue(removed)
        self.assertEqual(self.reward.evaluate(program, [], 25)['reward'], 1)
        nonempty = '<think>use the gold answer</think>add(25, 0)'
        self.assertEqual(normalize_swift_completion(nonempty), (nonempty, False))
        self.assertEqual(self.reward.evaluate(nonempty, [], 25)['reward'], 0)

    def test_double_prefix_and_arbitrary_wrapper_stay_invalid(self):
        prefix = '<think>\n\n</think>\n\n'
        for raw in (prefix+prefix+'add(25, 0)', '```add(25, 0)```'):
            program, _ = normalize_swift_completion(raw)
            self.assertEqual(self.reward.evaluate(program, [], 25)['reward'], 0)


if __name__ == '__main__':
    unittest.main()
