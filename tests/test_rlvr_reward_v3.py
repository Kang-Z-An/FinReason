import unittest

from finreason.rlvr_reward_v3 import FinQAV3Reward, reward_for_diagnostic


class V3RewardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = 'third_party/finqa/evaluate.py'
        cls.binary = FinQAV3Reward(path, 'binary')
        cls.validity = FinQAV3Reward(path, 'validity_0p1')

    def test_valid_wrong_only_gets_small_reward_in_validity_variant(self):
        program = 'add(1, 2)'
        self.assertEqual(self.binary.evaluate(program, [], 4)['reward'], 0.0)
        diagnostic = self.validity.evaluate(program, [], 4)
        self.assertEqual(diagnostic['reward'], 0.1)
        self.assertEqual(diagnostic['binary_reward'], 0.0)

    def test_correct_and_invalid_bounds(self):
        for variant in (self.binary, self.validity):
            self.assertEqual(variant.evaluate('add(1, 2)', [], 3)['reward'], 1.0)
            self.assertEqual(variant.evaluate('add(1, #0)', [], 3)['reward'], 0.0)
            self.assertEqual(variant.evaluate('divide(1, 0)', [], 3)['reward'], 0.0)

    def test_unknown_variant_fails(self):
        with self.assertRaises(ValueError):
            reward_for_diagnostic({'reward': 0, 'official_valid': True}, 'other')


if __name__ == '__main__':
    unittest.main()
