import enum

from graphviz import Source

from .ltl.automata import LDBA, ltl2ldba, ltl2nba
from .ltl.logic import Assignment
from .sequence.search import ExhaustiveSearch, BFS
import jax.numpy as jp


class Color(enum.Enum):
    SINK = 'tomato'
    ACCEPTING = 'lightskyblue'
    ROOT = '#ffcc00'

    def __str__(self):
        return self.value


def draw_ldba(ldba: LDBA, filename='ldba', fmt='pdf', view=True, positive_label=False, self_loops=True) -> None:
    """Draw an LDBA as a graph using Graphviz."""
    dot = 'digraph "" {\n'
    dot += 'rankdir=LR\n'
    dot += 'labelloc="t"\n'
    dot += 'node [shape="circle"]\n'
    dot += 'I [label="", style=invis, width=0]\n'
    dot += f'I -> {ldba.initial_state}\n'
    for state, transitions in ldba.state_to_transitions.items():
        dot += f'{state} [label="{state}" fontname="helvetica"'
        if state == ldba.sink_state:
            dot += f' color="{Color.SINK}" style="filled"'
        dot += ']\n'
        for transition in transitions:
            if not self_loops and transition.target == state:
                continue
            dot += f'{state} -> {transition.target} [label="{transition.label if not positive_label else (transition.positive_label if not "|" in transition.positive_label else transition.positive_label.split("|")[0])}"'
            if transition.accepting:
                dot += f' color="{Color.ACCEPTING}"'
            dot += ' fontname="helvetica"'
            dot += ']\n'
    dot += '}'
    s = Source(dot, filename=filename, format=fmt)
    s.render(view=view, cleanup=True)
    s.render(filename='automata', format='png', cleanup=True, )

def construct_ldba(formula: str, simplify_labels: bool = False, prune: list[Assignment] = [], ldba: bool = True) -> LDBA:
    fun = ltl2ldba if ldba else ltl2nba
    ldba = fun(formula, simplify_labels=simplify_labels)
    print('Constructed LDBA.')
    
    if prune:
        ldba.prune(prune)
        print('Pruned impossible transitions.')
    #ldba.prune_eps()
    #assert ldba.check_valid()
    ldba.complete_sink_state()
    print('Added sink state.')
    ldba.compute_sccs()
    return ldba


def ldba_to_dfa(ldba):
    dfa = {}
    for state, transitions in ldba.state_to_transitions.items():
        dfa[state] = {}
        for transition in transitions:
            if transition.target is state:
                continue
            assert len(transition.positive_label) == 1
            dfa[state][transition.positive_label] = transition.target
    return dfa


#'(a&!(b|c|d)) U ((b|c|d) & X((b & !(c|d)) U ((c|d) & X(c U d))))' 
if __name__ == '__main__':
    from ..envs.wheelbot_setpoint import SetpointContext
    from ..envs.wheelbot_zone_env import ZoneEnvContext
    from .formulae import *
    props = SetpointContext.ap_labels
    f = turn_left + '|' + turn_right
    possible_assignments = Assignment.one_proposition(set(props))
    ldba = construct_ldba(f, simplify_labels=True, prune=None, ldba=True)
    ldba.compute_accepting_state()
    print('accepting_states ', ldba.accepting_state)
    #print(f'Finite: {ldba.is_finite_specification()}')
    #print(f'number of states: {ldba.num_states}')
    # dfa = ldba_to_dfa(ldba)
    # print(repr(dfa))
    draw_ldba(ldba, filename='automata', fmt='png', positive_label=False, view=False, self_loops=True)
    #search = ExhaustiveSearch(None, props, num_loops=0, value_threshold=0.0)
    #seqs = search.all_sequences(ldba, ldba.initial_state)
    search = BFS(None, set(props), strict_avoid=True)
    seqs = search.bfs(ldba, ldba.initial_state, None)
    for seq in seqs:
        print(seq)
    
    print(1)
    seqs = search.bfs(ldba, 1, None)
    for seq in seqs:
        print(seq)
