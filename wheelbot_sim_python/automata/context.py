import jax.numpy as jp
from .ltl.automata.ldba import LDBA
from .dba import DBA
from .ltl.logic.assignment import Assignment, FrozenAssignment
from typing import Sequence
from ..reach_avoid.util import ReachAvoidSeq, batch_trees, select_tree
import jax
from .sequence.search.bfs import BFS
from .ltl.automata import LDBA, ltl2ldba
from dataclasses import dataclass
from .visualize import draw_ldba

OxReachAvoid = Sequence[tuple[Sequence[FrozenAssignment], Sequence[FrozenAssignment]]]

@dataclass(frozen=True)
class Context:
    ap_labels: list[str]
    possible_assignments: jp.ndarray

    @property
    def num_assignments(self) -> int:
        return self.possible_assignments.shape[0] + 2  # 0: padding, 1: epsilon
    
    @property
    def num_proper_assignments(self) -> int:
        return self.possible_assignments.shape[0]
    
    def assignment_to_index(self, assignment: jp.ndarray) -> int:
        matches = jp.all(self.possible_assignments == assignment, axis=1)
        return jp.argmax(matches) + 2 # 0 : padding, 1 : epsilon
    
    def index_to_assignment(self, index: int) -> jp.ndarray:
        return self.possible_assignments[index - 2, :] # 0 : padding, 1 : epsilon

    def assignment_array_to_assignment(self, assignment: jp.ndarray) -> Assignment:
        dict = {}
        for i, prop in enumerate(self.ap_labels):
            dict[prop] = assignment[i].item()

        return Assignment(dict)
    
    def assignment_indices_to_frozen_assignment_set(self, set: jp.ndarray, set_size: int) -> set[Assignment]:
        res = []
        for i in range(set_size):
            assignment_array = self.index_to_assignment(set[i])
            assignment = self. assignment_array_to_assignment(assignment_array)
            res.append(FrozenAssignment(assignment))
        return res

    def frozen_assignment_to_assignment_array(self, frozen: FrozenAssignment) -> jp.ndarray:
        assignment_dict = dict(frozen.assignment)
        assignment_list = [assignment_dict.get(p, False) for p in self.ap_labels]
        return jp.array(assignment_list, dtype=bool)

    def frozen_assignments_to_assignment_indices(self, frozen_assignments: Sequence[FrozenAssignment], set_size: int) -> jp.ndarray:
        assignment_indices = []
        for frozen in frozen_assignments:
            assignment_array = self.frozen_assignment_to_assignment_array(frozen)
            assignment_index = self.assignment_to_index(assignment_array)
            assignment_indices.append(assignment_index)
        assert set_size >= len(assignment_indices)
        padding = [0] * (set_size - len(assignment_indices))
        assignment_indices += padding
        return jp.array(assignment_indices)

    def ox_reach_avoid_to_jax_reach_avoid(self, ox_seq: OxReachAvoid, seq_size: int, set_size: int) -> ReachAvoidSeq:
        length = jp.array([min(len(ox_seq), seq_size)])
        reach_seq = []
        avoid_seq = []
        for frozen_reach, frozen_avoid in ox_seq:
            reach_indices = self.frozen_assignments_to_assignment_indices(frozen_reach, set_size)
            reach_seq.append(reach_indices)
            avoid_indices = self.frozen_assignments_to_assignment_indices(frozen_avoid, set_size)
            avoid_seq.append(avoid_indices)
        
        reach_seq = jp.array(reach_seq)
        avoid_seq = jp.array(avoid_seq)

        if reach_seq.shape[0] < seq_size:
            padding_sets = jp.zeros((seq_size - reach_seq.shape[0], reach_seq.shape[1]))
            reach_seq = jp.concatenate([reach_seq, padding_sets], axis=0)
            avoid_seq = jp.concatenate([avoid_seq, padding_sets], axis=0)
        reach_set_sizes = jp.sum(reach_seq != 0, axis=1)
        avoid_set_sizes = jp.sum(avoid_seq != 0, axis=1)

        reach = {'seq': reach_seq, 'set_sizes': reach_set_sizes}
        reach = jax.tree.map(lambda x: x[:seq_size, ...], reach)
        avoid = {'seq': avoid_seq, 'set_sizes': avoid_set_sizes}
        avoid = jax.tree.map(lambda x: x[:seq_size, ...], avoid)
        return {'reach': reach, 'avoid': avoid, 'length': length}
        

    def ox_ldba_to_jax_dba(self, ldba: LDBA, seq_size: int, strict_avoid: bool) -> DBA:
        transitions = [[-1 for _ in range(self.num_assignments)] for _ in range(ldba.num_states)]
        search = BFS(None, set(self.ap_labels), strict_avoid)
        state_to_seqs = {}
        #create transition matrix
        for source in range(ldba.num_states):
            for trans in ldba.state_to_transitions[source]:
                target = trans.target
                frozen_assignments = trans.valid_assignments
                assignment_indices = self.frozen_assignments_to_assignment_indices(frozen_assignments, self.num_assignments)
                for index in assignment_indices.tolist():
                    if index == 0:
                        continue
                    if index == 1:
                        raise ValueError("Epsilon transitions are not supported in the DBA")
                    transitions[source][index] = target
            seqs = search.bfs(ldba, source, None)
            state_to_seqs[source] = seqs
        transitions = jp.array(transitions)

        #create reach-avoid sequences
        state_to_jax_seqs = {}
        max_seq_length = max(len(seq) for _, seqs in state_to_seqs.items() for seq in seqs)
        seq_size = min(max_seq_length, seq_size) # if resulting jax sequences should not be longer than all ldba sequences
        for state, seqs in state_to_seqs.items():
            jax_seqs = []
            for seq in seqs:
                jax_seq = self.ox_reach_avoid_to_jax_reach_avoid(seq, seq_size, self.num_assignments)
                jax_seqs.append(jax_seq)
            state_to_jax_seqs[state] = jax_seqs

        #create padding reach-avoid sequence
        some_seq = None
        for jax_seqs in state_to_jax_seqs.values():
            if jax_seqs:                
                some_seq = jax_seqs[0]
                break

        if some_seq == None:
            raise ValueError("LDBA does not contain any accepting path")
        padding_seq = jax.tree.map(lambda x: jp.zeros_like(x), some_seq)

        #fill in padding sequences and make padding mask
        max_num_accepting_paths = max(len(seqs) for seqs in state_to_seqs.values())
        padding_mask = {}
        for state, jax_seqs in state_to_jax_seqs.items():
            padding_mask[state] = jp.array([1] * len(jax_seqs) + [0] * (max_num_accepting_paths - len(jax_seqs)))
            if len(jax_seqs) < max_num_accepting_paths:
                padding = [padding_seq] * (max_num_accepting_paths - len(jax_seqs))
                jax_seqs += padding
            state_to_jax_seqs[state] = batch_trees(*jax_seqs)

        batched_reach_avoid_seqs = batch_trees(*[state_to_jax_seqs[s] for s in sorted(state_to_jax_seqs.keys())])
        padding_masks = jp.array([padding_mask[s] for s in sorted(padding_mask.keys())])
        #compute accepting states
        ldba.compute_accepting_state()
        accepting_state = ldba.accepting_state
        sink_state = ldba.sink_state
        return DBA(transitions, batched_reach_avoid_seqs, padding_masks, sink_state, accepting_state)
    
    def formula_to_dba(self, formula: str, seq_size: int, strict_avoid: bool= True, simplify_labels: bool = False, draw: bool = False, positive_labels: bool = False) -> DBA:
        ldba = ltl2ldba(formula, self.ap_labels, simplify_labels=simplify_labels)
        if len(self.possible_assignments) < 2 ** len(self.ap_labels):
            possible_assignments_list = [self.assignment_array_to_assignment(a) for a in self.possible_assignments]
            ldba.prune(possible_assignments_list)
        ldba.complete_sink_state()
        ldba.compute_accepting_state()
        ldba.compute_sccs()
        ldba.prune_eps()
        assert ldba.check_valid()
        if draw:
            draw_ldba(ldba, 'automata', fmt='png', view=False, positive_label=positive_labels, self_loops=True)
        return self.ox_ldba_to_jax_dba(ldba, seq_size, strict_avoid)
    
if __name__ == '__main__':
    import itertools
    ap_labels = ['a', 'b', 'c']
    possible_assignments = jp.array(list(itertools.product([False, True], repeat=len(ap_labels))))
    print(possible_assignments)
    context = Context(ap_labels, possible_assignments)

    #test assignment array -> Assignment -> FrozenAssignment -> assignment array
    assignment_array = jp.array([True, False, True])
    frozen_assingment = FrozenAssignment(context.assignment_array_to_assignment(assignment_array))
    assignment_array_1 = context.frozen_assignment_to_assignment_array(frozen_assingment)
    assert jp.all(assignment_array == assignment_array_1)
    print('All assignment tests passed')

    #test Seq[FrozenAssignment] -> assignment indices
    asignment_array_1 = jp.array([True, False, True])
    asignment_array_2 = jp.array([False, False, True])
    asignment_array_3 = jp.array([True, False, False])
    frozen_assignment_1 = FrozenAssignment(context.assignment_array_to_assignment(asignment_array_1))
    frozen_assignment_2 = FrozenAssignment(context.assignment_array_to_assignment(asignment_array_2))
    frozen_assignment_3 = FrozenAssignment(context.assignment_array_to_assignment(asignment_array_3))
    frozen_assignemnts = [frozen_assignment_1, frozen_assignment_2, frozen_assignment_3]
    assignment_indices = context.frozen_assignments_to_assignment_indices(frozen_assignemnts, 3)

    assignment_indices_golden = [3, 6, 7] # add 2 for offset caused by eps and padding assignment
    assert sorted(assignment_indices) == assignment_indices_golden
    print('All assignment list tests passed')

    #test ox reach-avoid to jax reach-avoid
    def construt_ldba(formula: str):
        ldba = ltl2ldba(formula, propositions=ap_labels)
        assert ldba.check_valid()
        ldba.complete_sink_state()
        ldba.compute_sccs()
        return ldba
    
    ldba = construt_ldba('a U b')
    search = BFS(None, set(ap_labels))
    seqs = search.bfs(ldba, ldba.initial_state, None)
    seq = seqs[0]
    reach_avoid = context.ox_reach_avoid_to_jax_reach_avoid(seq, seq_size=3, set_size=8)
    assert jp.all(jp.sort(reach_avoid['reach']['seq'][0, :4]) == jp.array([4, 5, 8, 9]))
    assert jp.all(jp.sort(reach_avoid['avoid']['seq'][0, :2]) == jp.array([2, 3]))
    print('All reach_avoid tests passed')
    
    #test formula -> ox ldba -> jax dba
    dba = context.formula_to_dba('(a U b) | (b U c)', seq_size=2, draw=False)
    assert dba.num_states == 4
    assert dba.num_assignments == 10
    assert dba.init_state == 0
    assert dba.sink_state == 3
    assert dba.step(0, 2) == dba.sink_state
    assert dba.step(0, 6) == 2
    assert dba.step(0, 3) == 1
    assert dba.step(2, 4) == 1
    assert dba.is_accepting_state(1)

    seqs = [[], [], [], []]
    for i in range(dba.num_states):
        seqs[i].append(dba.get_reach_avoid_at(i, 0))
        seqs[i].append(dba.get_reach_avoid_at(i, 1))
    
    def pytree_equal(p1, p2):
        # Compare structure and values
        return jax.tree_util.tree_all(
            jax.tree.map(lambda x, y: jp.array_equal(x, y), p1, p2)
        )

    #test padding pytrees
    padding_seq = jax.tree.map(lambda x: jp.zeros_like(x), seqs[1][0])
    assert pytree_equal(seqs[1][0], padding_seq)
    assert pytree_equal(seqs[1][0], seqs[1][1]) and pytree_equal(seqs[1][0], seqs[3][0]) and pytree_equal(seqs[1][0], seqs[3][1]) and pytree_equal(seqs[1][0], seqs[2][1])

    #test reach sequences
    seq = seqs[0][1]
    assert seq['length'].item() == 2
    reach_seq = seq['reach']['seq']
    assert jp.all(jp.sort(reach_seq[0, :]) == jp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 6]))
    assert jp.all(jp.sort(reach_seq[1, :]) == jp.array([0, 0, 0, 0, 0, 0, 4, 5, 8, 9]))

    seq = seqs[0][0]
    assert seq['length'].item() == 1
    reach_seq = seq['reach']['seq']
    assert jp.all(jp.sort(reach_seq[0, :]) == jp.array([0, 0, 0, 0, 3, 4, 5, 7, 8, 9]))

    #test pruning of impossible transitions
    ap_labels = ['a', 'b', 'c']
    possible_assignments = jp.array([[False, False, False],
                                     [True, False, False],
                                     [False, True, False],
                                     [False, False, True],])
    context = Context(ap_labels, possible_assignments)
    dba = context.formula_to_dba('(F (a)) | ((b) U (c))', seq_size=2, draw=False, positive_labels=False)
    assert dba.num_states == 3
    assert dba.num_assignments == 6
    assert dba.init_state == 0
    assert dba.sink_state is None
    assert dba.step(0, 2) == 1
    assert dba.step(0, 3) == 2
    assert dba.step(0, 4) == 0
    assert dba.step(0, 5) == 2
    assert dba.step(1, 2) == 1
    assert dba.step(1, 3) == 2
    assert dba.step(1, 4) == 1
    assert dba.step(1, 5) == 1
    assert dba.step(2, 2) == 2
    assert dba.step(2, 3) == 2
    assert dba.step(2, 4) == 2
    assert dba.step(2, 5) == 2
    assert dba.is_accepting_state(2)

    seqs = [[], [], [], []]
    for i in range(dba.num_states):
        seqs[i].append(dba.get_reach_avoid_at(i, 0))
        seqs[i].append(dba.get_reach_avoid_at(i, 1))

    seq = seqs[0][0]
    assert seq['length'].item() == 1
    reach_seq = seq['reach']['seq']
    avoid_seq = seq['avoid']['seq']
    assert jp.all(jp.sort(reach_seq[0, :]) == jp.array([0, 0, 0, 0, 3, 5]))
    assert jp.all(jp.sort(reach_seq[1, :]) == jp.array([0, 0, 0, 0, 0, 0]))
    assert jp.all(jp.sort(avoid_seq[0, :]) == jp.array([0, 0, 0, 0, 0, 0]))
    assert jp.all(jp.sort(avoid_seq[1, :]) == jp.array([0, 0, 0, 0, 0, 0]))

    seq = seqs[0][1]
    assert seq['length'].item() == 2
    reach_seq = seq['reach']['seq']
    avoid_seq = seq['avoid']['seq']
    assert jp.all(jp.sort(reach_seq[0, :]) == jp.array([0, 0, 0, 0, 0, 2]))
    assert jp.all(jp.sort(reach_seq[1, :]) == jp.array([0, 0, 0, 0, 0, 3]))
    assert jp.all(jp.sort(avoid_seq[0, :]) == jp.array([0, 0, 0, 0, 0, 0]))
    assert jp.all(jp.sort(avoid_seq[1, :]) == jp.array([0, 0, 0, 0, 0, 0]))

    print('All automata tests passed')