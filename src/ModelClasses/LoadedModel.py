from enum import Enum, auto

from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer


class MissionTypes(Enum):
    Regression: auto()
    Classification: auto()


class LoadedModel:
    model: nn.Module
    optimizer: Optimizer
    missionType: MissionTypes
    loss: _Loss

    def __init__(self, model, optimizer, missionType, loss):
        self.model = model
        self.optimizer = optimizer
        self.missionType = missionType
        self.loss = loss