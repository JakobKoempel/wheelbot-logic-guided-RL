from dataclasses import dataclass

from torch import nn
from typing import Tuple
from ...ltl.automata import LDBA, LDBASequence, LDBATransition
from . import SequenceSearch


@dataclass(eq=True, frozen=True)
class SearchNode:
    ldba_state: int
    sequence: LDBASequence
    visited_seq: list[Tuple[int, bool]]

    def has_accepting_cycle(self, trans: LDBATransition) -> bool:
        visited_states = set(s for s, _ in self.visited_seq)

        #check whether there is a cycle
        if trans.target in visited_states:
            #check whether the cycle is accepting
            cycle_start_index = next(i for i, (n, _) in enumerate(self.visited_seq) if n == trans.target)
            cycle = self.visited_seq[cycle_start_index:]
            cycle.append((trans.target, trans.accepting))
            return any(accepting for _, accepting in cycle) and len(cycle) > 1
        
        else:
            return False


class BFS(SequenceSearch):
    def __init__(self, model: nn.Module, propositions: set[str], strict_avoid: bool = True):
        self.strict_avoid = strict_avoid
        super().__init__(model, propositions)

    def __call__(self, ldba: LDBA, ldba_state: int, obs) -> LDBASequence:
        seqs = self.bfs(ldba, ldba_state, obs)
        seq = max(seqs, key=lambda s: self.get_value(s, obs))
        seq = self.augment_sequence(ldba, ldba_state, seq)
        return seq
    
    def is_true_transition(self, ldba: LDBA, transition: LDBATransition) -> bool:
        return len(ldba.state_to_transitions[transition.source]) == 1 and transition.source == transition.target

    def bfs(self, ldba: LDBA, ldba_state: int, obs) -> list[LDBASequence]:
        queue = [SearchNode(ldba_state, (), [])]
        sequences = []
        while queue:
            node = queue.pop(0)
            visited_states = set(s for s, _ in node.visited_seq)
            for trans in ldba.state_to_transitions[node.ldba_state]:
                if self.strict_avoid:
                    avoid_transitions = self.collect_avoid_transitions_strict(ldba, node.ldba_state, trans)
                else:
                    avoid_transitions = self.collect_avoid_transitions_relaxed(ldba, node.ldba_state, visited_states)
                avoid = [a.valid_assignments for a in avoid_transitions]
                avoid = set() if not avoid else set.union(*avoid)
                if trans.target == trans.source:
                    if trans.accepting:
                        if not self.is_true_transition(ldba, trans):
                            new_sequence = node.sequence + ((frozenset(trans.valid_assignments), frozenset(avoid)),)
                            sequences.append(new_sequence)
                        elif node.sequence:
                            sequences.append(node.sequence)
                    continue

                scc = ldba.state_to_scc[trans.target]
                if scc.bottom and not scc.accepting:
                    continue
                    
                new_sequence = node.sequence + ((frozenset(trans.valid_assignments), frozenset(avoid)),)
                if trans.target in visited_states:
                    if node.has_accepting_cycle(trans):
                        sequences.append(new_sequence)
                    continue     
                
                new_node = SearchNode(trans.target, new_sequence, node.visited_seq + [(node.ldba_state, trans.accepting)])
                queue.append(new_node)
        return sequences
