"""Frozen v3 FinQA GRPO reward variants; both use the official executor."""

from .rlvr_reward import FinQAReward


VARIANTS = ('binary', 'validity_0p1')


def reward_for_diagnostic(diagnostic, variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown frozen reward variant: ' + str(variant))
    if diagnostic['reward'] == 1.0:
        return 1.0
    if variant == 'validity_0p1' and diagnostic['official_valid'] is True:
        return 0.1
    return 0.0


class FinQAV3Reward(FinQAReward):
    def __init__(self, official_path, variant):
        if variant not in VARIANTS:
            raise ValueError('Unknown frozen reward variant: ' + str(variant))
        super().__init__(official_path)
        self.variant = variant

    def evaluate(self, program, table, gold_answer):
        diagnostic = super().evaluate(program, table, gold_answer)
        binary = diagnostic['reward']
        diagnostic['binary_reward'] = binary
        diagnostic['reward'] = reward_for_diagnostic(diagnostic, self.variant)
        diagnostic['reward_variant'] = self.variant
        return diagnostic
