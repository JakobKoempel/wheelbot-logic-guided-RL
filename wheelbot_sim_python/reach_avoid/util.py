import jax
from jax import numpy as jp
from flax import struct
from jax import numpy as jp
from typing import Tuple, Any
from enum import IntEnum
from jax import jit, lax, debug
from functools import partial

ReachAvoidSeq = Any

@jit
def select_tree(batched_pytree, index):
    return jax.tree.map(lambda x: x[index], batched_pytree)

@jit
def batch_trees(*pytrees):
    return jax.tree.map(lambda *leaves: jp.stack(leaves), *pytrees)

class EvalStatus(IntEnum):
    FAIL = -1
    ONGOING = 0
    SUCCESS = 1

@struct.dataclass
class AssignmentSet:
    elements: jp.ndarray

    @jit
    def contains(self, assignment_index) -> bool:
        return jp.any(self.elements == assignment_index) 

@struct.dataclass
class ReachAvoidState:
    reach_avoid: ReachAvoidSeq
    status: EvalStatus
    progress: bool

    @property
    def is_success(self) -> bool:
        return self.status == EvalStatus.SUCCESS
    
    @property
    def is_fail(self) -> bool:
        return self.status == EvalStatus.FAIL
    
    @property
    def is_ongoing(self) -> bool:
        return self.status == EvalStatus.ONGOING

def create_reach_avoid_state(reach_avoid: ReachAvoidSeq) -> ReachAvoidState:
    return ReachAvoidState(reach_avoid=reach_avoid,
                            status=EvalStatus.ONGOING,
                            progress=False)

def get_prefix(reach_avoid: ReachAvoidSeq, prefix_length: int) -> ReachAvoidSeq:
    """
    Returns a reach-avoid sequence of given prefix length
    """
    reach_avoid['length'] = jp.ravel(prefix_length)
    return reach_avoid

@jit 
def flatten_reach_avoid(reach_avoid: ReachAvoidSeq) -> jp.ndarray:
    """
    Returns flattened reach-avoid sequence
    """
    reach_seq_flat = jp.ravel(reach_avoid['reach']['seq'])
    reach_set_sizes_flat = jp.ravel(reach_avoid['reach']['set_sizes'])
    avoid_seq_flat = jp.ravel(reach_avoid['avoid']['seq'])
    avoid_set_sizes_flat = jp.ravel(reach_avoid['avoid']['set_sizes'])
    length = reach_avoid['length'][0]
    return jp.array([*reach_seq_flat, *reach_set_sizes_flat, *avoid_seq_flat, *avoid_set_sizes_flat, length])

def unflatten_reach_avoid(flat: jp.ndarray, max_length: int) -> ReachAvoidSeq:
    """
    Returns reach-avoid sequence as pytree
    """
    length = jp.array([flat[-1]])
    flat = flat[:-1]
    mid = flat.shape[0] // 2
    reach = flat[:mid]
    avoid = flat[mid:]
    reach_seq_flat = reach[:-max_length]
    reach_set_sizes = reach[-max_length:]
    avoid_seq_flat = avoid[:-max_length]
    avoid_set_sizes = avoid[-max_length:]
    reach_seq = reach_seq_flat.reshape(max_length, -1)
    avoid_seq = avoid_seq_flat.reshape(max_length, -1)

    return {
        'reach': {
            'seq': reach_seq,
            'set_sizes': reach_set_sizes
        },
        'avoid': {
            'seq': avoid_seq,
            'set_sizes': avoid_set_sizes
        },
        'length': length
    }

@jit
def reverse_reach_avoid(reach_avoid: ReachAvoidSeq) -> ReachAvoidSeq:
    return jax.tree_util.tree_map(lambda x: jp.flip(x, axis=0), reach_avoid)

@jit
def get_active_reach_avoid_sets(reach_avoid: ReachAvoidSeq) -> Tuple[AssignmentSet, AssignmentSet]:
    current_reach_set = reach_avoid['reach']['seq'][0]
    current_avoid_set = reach_avoid['avoid']['seq'][0]
    length = reach_avoid['length'][0]

    """lax.cond(
        length <= 0,
        lambda _: debug.print("Warning: length {} <=  0 in get_active_reach_avoid_sets", length),
        lambda _: None,
        operand=None
    )"""

    return AssignmentSet(current_reach_set), AssignmentSet(current_avoid_set)

@jit
def update_reach_avoid(state: ReachAvoidState, assignment_index: int) -> 'ReachAvoidState':

    """lax.cond(
        jp.logical_or(state.is_success, state.is_fail),
        lambda _: debug.print("Warning: evaluation of reach-avoid sequence is already finished but update_reach_avoid() is called again"),
        lambda _: None,
        operand=None
    )"""

    reach_avoid = state.reach_avoid
    reach_seq = reach_avoid['reach']['seq']
    reach_set_sizes = reach_avoid['reach']['set_sizes']
    avoid_seq = reach_avoid['avoid']['seq']
    avoid_set_sizes =reach_avoid['avoid']['set_sizes']
    status = state.status
    length = reach_avoid['length']
    reach_set, avoid_set = get_active_reach_avoid_sets(reach_avoid)

    def case_in_reach(_):
        padding_set = jp.zeros_like(reach_seq[0])
        updated_reach_seq = jp.array([*reach_seq[1:], padding_set])
        updated_reach_set_sizes = jp.array([*reach_set_sizes[1:], 0])
        updated_avoid_seq = jp.array([*avoid_seq[1:], padding_set])
        updated_avoid_set_sizes = jp.array([*avoid_set_sizes[1:], 0])
        updated_length = length - 1
        return {
            'reach': {
                'seq': updated_reach_seq,
                'set_sizes': updated_reach_set_sizes
            },
            'avoid': {
                'seq': updated_avoid_seq,
                'set_sizes': updated_avoid_set_sizes
            },
            'length': updated_length
        }, True

    def case_not_in_reach(_):
        return reach_avoid, False
    
    reach_avoid, progress = lax.cond(
        reach_set.contains(assignment_index),
        case_in_reach,
        case_not_in_reach,
        operand=None
    )

    status = lax.cond(
        avoid_set.contains(assignment_index),
        lambda _: EvalStatus.FAIL,
        lambda _: status,
        operand=None
    )

    status = lax.cond(
        reach_avoid['length'][0] == 0,
        lambda _: EvalStatus.SUCCESS,
        lambda _: status,
        operand=None
    )

    return state.replace(reach_avoid=reach_avoid, status=status, progress=progress)

def _pytree_allclose(tree1, tree2, atol=1e-6, rtol=1e-6) -> bool:
    # Check structure and shapes first
    try:
        jax.tree_util.tree_structure(tree1) == jax.tree_util.tree_structure(tree2)
    except Exception:
        return False
    # Compare leaves elementwise
    leaves1, _ = jax.tree_util.tree_flatten(tree1)
    leaves2, _ = jax.tree_util.tree_flatten(tree2)
    if len(leaves1) != len(leaves2):
        return False
    for l1, l2 in zip(leaves1, leaves2):
        if isinstance(l1, jp.ndarray) or isinstance(l2, jp.ndarray):
            if not jp.allclose(l1, l2, atol=atol, rtol=rtol):
                return False
        else:
            if l1 != l2:
                return False
    return True

if __name__ == "__main__":
    reach_set1 = jp.array([1.0, 2.0, 0.0, 0.0])
    reach_set2 = jp.array([3.0, 2.0, 0.0, 0.0])
    reach_set3 = jp.array([4.0, 1.0, 2.0, 3.0])

    avoid_set1 = jp.array([3.0, 4.0, 5.0, 0.0])
    avoid_set2 = jp.array([1.0, 0.0, 0.0, 0.0])
    avoid_set3 = jp.array([6.0, 0.0, 0.0, 0.0])
    reach_set_sizes = jp.array([2, 2, 4])
    avoid_set_sizes = jp.array([3, 1, 1])
    reach = {'seq': jp.array([reach_set1, reach_set2, reach_set3]), 'set_sizes': reach_set_sizes}
    avoid = {'seq': jp.array([avoid_set1, avoid_set2, avoid_set3]), 'set_sizes': avoid_set_sizes}
    reach_avoid = {'reach': reach, 'avoid': avoid, 'length': jp.array([3])}

    #test prefix generation
    batched_prefixes = get_all_prefixes(reach_avoid)
    assert _pytree_allclose(select_tree(batched_prefixes, 2), reach_avoid)

    #test flattening and unflattening
    flat = flatten_reach_avoid(reach_avoid)
    unflatten = jit(partial(unflatten_reach_avoid, max_length=3))
    reconstructed = unflatten(flat)
    assert _pytree_allclose(reach_avoid, reconstructed)

    #test ReachAvoidState creation and updates
    state = create_reach_avoid_state(reach_avoid)
    assert state.is_ongoing
    state = update_reach_avoid(state, 1)
    assert state.is_ongoing and state.reach_avoid['length'] == 2
    state = update_reach_avoid(state, 2)
    assert state.is_ongoing and state.reach_avoid['length'] == 1

    success_state = update_reach_avoid(state, 3)
    assert success_state.is_success and success_state.reach_avoid['length'] == 0
    
    #Test whether warning is printed
    warning_state = update_reach_avoid(success_state, 5)

    fail_state = update_reach_avoid(state, 6)
    assert fail_state.is_fail and fail_state.reach_avoid['length'] == 1

    ongoing_state = update_reach_avoid(state, 7)
    assert ongoing_state.is_ongoing and ongoing_state.reach_avoid['length'] == 1

    print('Test passed')