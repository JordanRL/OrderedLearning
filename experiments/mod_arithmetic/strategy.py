from framework import SimpleTrainStep

class StrideStrategy(SimpleTrainStep):
    @property
    def name(self) -> str:
        return "Stride"

class TargetStrategy(SimpleTrainStep):
    @property
    def name(self) -> str:
        return "Target"

class RandomStrategy(SimpleTrainStep):
    @property
    def name(self) -> str:
        return "Random"

class FixedRandomStrategy(SimpleTrainStep):
    @property
    def name(self) -> str:
        return "Fixed Random"