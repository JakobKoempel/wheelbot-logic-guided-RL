
import jax.numpy as jp
from ..reach_avoid import util
from ..reach_avoid.util import select_tree
from typing import Optional

class DBA():

    def __init__(self, transitions: jp.ndarray, accepting_reach_avoid_seq, accepting_reach_avoid_mask,
                sink_state: Optional[int] = None, accepting_state: Optional[int] = None,
                init_state: int = 0):
        """
        transitions: shape [num_states, num_assignments]
        accepting_reach_avoid_seq: batched reached_avoid_sequenecs with batch dimensions [num_states, max_num_accepting_paths]
        accepting_reach_avoid_mask: shape [num_states, max_num_accepting_paths]
        """
        self.transitions = transitions
        self.num_states = self.transitions.shape[0]
        self.num_assignments = self.transitions.shape[1] #includes padding and eps
        self.accepting_reach_avoid_seq = accepting_reach_avoid_seq
        self.accepting_reach_avoid_mask = accepting_reach_avoid_mask
        self.sink_state = sink_state
        self.accepting_state = accepting_state
        self.init_state = init_state   

    @property
    def sequence_size(self):
        return self.accepting_reach_avoid_seq['reach']['seq'].shape[2]
    
    @property
    def max_num_accepting_paths(self):
        return self.accepting_reach_avoid_seq['reach']['seq'].shape[1]
    
    def get_reach_avoid_at(self, state: int, index: int):
        return select_tree(select_tree(self.accepting_reach_avoid_seq, state), index)

    def step(self, state: int, assignment: int) -> int:
        return self.transitions[state, assignment]
    
    def get_all_reach_avoid_at(self, state: int):
        """
        returns reach-avoid sequences for all accepting paths starting from state
        """
        return util.select_tree(self.accepting_reach_avoid_seq, state)
    
    def is_sink_state(self, state: int) -> bool:
        return state == self.sink_state
    
    def is_accepting_state(self, state: int) -> bool:
        return state == self.accepting_state


if __name__ == "__main__":
    from ..envs.wheelbot_setpoint import SetpointContext
    from ..envs.wheelbot_zone_env import ZoneEnvContext
    from .formulae import *
    dba = ZoneEnvContext.formula_to_dba("(F red) | F (green & F blue)", seq_size=2, strict_avoid=False, simplify_labels=True, draw=True)
    print(dba.get_reach_avoid_at(0, 0))
    print(dba.get_reach_avoid_at(0, 1))
    print(dba.get_reach_avoid_at(1, 0))
    print(dba.get_reach_avoid_at(1, 1))
    print(dba.get_reach_avoid_at(2, 0))
    print(dba.get_reach_avoid_at(2, 1))

 