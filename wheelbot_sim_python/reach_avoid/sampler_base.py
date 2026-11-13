from abc import ABC, abstractmethod
import jax.numpy as jp
from .util import ReachAvoidSeq

class ReachAvoidSampler(ABC):
    @abstractmethod
    def sample_eval(self, obs: jp.ndarray, rng: jp.ndarray) -> ReachAvoidSeq:
        pass

    @abstractmethod
    def sample_training(self, obs: jp.ndarray, rng: jp.ndarray) -> ReachAvoidSeq:
        pass


class ExplicitReachAvoidSampler(ReachAvoidSampler):

    def __init__(self, batched_reach_avoid):
        self.batched_reach_avoid = batched_reach_avoid
        self.num_sequences = self.batched_reach_avoid['reach']['seq'].shape[0]
        self.sequence_size = self.batched_reach_avoid['reach']['seq'].shape[1]

    @abstractmethod
    def sample_eval(self, obs: jp.ndarray, rng: jp.ndarray) -> ReachAvoidSeq:
        pass

    @abstractmethod
    def sample_training(self, obs: jp.ndarray, rng: jp.ndarray) -> ReachAvoidSeq:
        pass

class ExplicitZoneSampler(ReachAvoidSampler):

    def __init__(self, sequence_size, reach_sets: jp.ndarray, avoid_sets: jp.ndarray):
        self.sequence_size = sequence_size

        reach_set_size = jp.array([jp.sum(reach_sets != 0)])
        avoid_set_size = jp.array([jp.sum(avoid_sets != 0)])

        reach = {'seq': reach_sets, 'set_sizes': reach_set_size}
        avoid = {'seq': avoid_sets, 'set_sizes': avoid_set_size}
        self.reach_avoid = {'reach': reach, 'avoid': avoid, 'length': jp.array([sequence_size])}

    def sample_eval(self, obs: jp.ndarray, rng: jp.ndarray) -> ReachAvoidSeq:
        return self.reach_avoid
    
    def sample_training(self, obs: jp.ndarray, rng: jp.ndarray) -> ReachAvoidSeq:
        return self.reach_avoid