import json
from pathlib import Path
import subprocess
import sys
import unittest

from scripts.verify_results import verify


class ReleaseTests(unittest.TestCase):
    def test_all_five_results_and_paired_counts(self):
        result = verify(with_intervals=False)
        self.assertEqual((result['examples'], result['models'], result['paired_comparisons']), (551, 5, 6))

    def test_train_plans_preserve_precision_and_grouping(self):
        for stage in ('sft', 'grpo'):
            command = [sys.executable, '-m', 'scripts.train', '--stage', stage,
                       '--model-dir', 'models/base', '--output', 'outputs/test']
            if stage == 'grpo':
                command += ['--adapter', 'models/adapter', '--variant', 'binary', '--seed', '1234']
            result = json.loads(subprocess.check_output(command, text=True))
            self.assertFalse(result['execute'])
            options = result['command']
            self.assertEqual(options[options.index('--torch_dtype')+1], 'float32' if stage == 'sft' else 'bfloat16')
            if stage == 'grpo':
                self.assertEqual(options[options.index('--num_generations')+1], '8')
                self.assertEqual(options[options.index('--dynamic_sample')+1], 'false')
                self.assertEqual(options[options.index('--seed')+1], '1234')


if __name__ == '__main__':
    unittest.main()
