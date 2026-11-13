from .sampler_base import ExplicitReachAvoidSampler
import jax.numpy as jp
from .util import ReachAvoidSeq, select_tree, get_prefix
from jax import lax
import jax
from ..envs.wheelbot_setpoint import SetpointContext
from ..automata.formulae import *

class SetpointSampler(ExplicitReachAvoidSampler):
    def __init__(self):
        dba = SetpointContext.formula_to_dba(turn_either, seq_size=4)
        seqs = dba.get_all_reach_avoid_at(0)
        super().__init__(seqs)

    def sample_eval(self, obs: jp.ndarray, rng: jp.ndarray) -> ReachAvoidSeq:
        setpoint_yaw = obs[12]
        seq_index = lax.cond(setpoint_yaw < 0., lambda: 0 , lambda: 1)              
        return select_tree(self.batched_reach_avoid, seq_index)
    
    def sample_training(self, obs: jp.ndarray, rng: jp.ndarray) -> ReachAvoidSeq:
        setpoint_yaw = obs[12]
        seq_index = lax.cond(setpoint_yaw < 0.,
                            lambda: jax.random.choice(rng, jp.array([0,1]), p=jp.array([0.96, 0.04])),
                            lambda: jax.random.choice(rng, jp.array([1,0]), p=jp.array([0.96, 0.04])),
                            )
        prefix_length =  jax.lax.cond(jp.logical_or(
                                        jp.logical_and(setpoint_yaw < 0, seq_index == 1),
                                        jp.logical_and(setpoint_yaw > 0, seq_index == 0),
                                    ),  
                                    lambda: 4,
                                    lambda: jax.random.randint(rng, minval=2, maxval=5, shape=()))
        
        seq = select_tree(self.batched_reach_avoid, seq_index)
        prefix = get_prefix(seq, prefix_length)
        return prefix 

class LeftSampler(ExplicitReachAvoidSampler):
    def __init__(self):
        dba = SetpointContext.formula_to_dba(turn_left, seq_size=3)
        seq = dba.get_all_reach_avoid_at(0)
        super().__init__(seq)

    def sample_eval(self, obs: jp.ndarray, rng: jp.ndarray) -> ReachAvoidSeq:
        return select_tree(self.batched_reach_avoid, 0)
    
    def sample_training(self, obs: jp.ndarray, rng: jp.ndarray) -> ReachAvoidSeq:
        seq = select_tree(self.batched_reach_avoid, 0)
        prefix_length = jax.random.randint(rng, minval=1, maxval=4, shape=())
        prefix = get_prefix(seq, prefix_length)
        return prefix