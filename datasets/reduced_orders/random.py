import torch
from .base import ReducedModel

class RandomReducedModel(ReducedModel):
    def __init__(self, eigen_value_fraction, eigen_vector_count):
        self.eigen_value_fraction = eigen_value_fraction
        self.eigen_vector_count = eigen_vector_count

    def get_direction(self, simulator, x):
        return torch.randn((simulator.domain, self.eigen_count(simulator)))

    def eigen_count(self, simulator):
        if self.eigen_vector_count is not None:
            return self.eigen_vector_count
        return int(simulator.domain * self.eigen_value_fraction)
