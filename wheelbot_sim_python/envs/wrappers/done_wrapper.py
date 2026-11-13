from brax.envs.base import State, Wrapper
import jax
from jax import numpy as jp

class DoneCountWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset_to_start(self, track_seed: int):
        state = self.env.reset_to_start(track_seed)
        metrics = {"done_at": -1, 'step': 0}
        return state.replace(metrics=metrics)
    
    def reset(self, rng: jp.ndarray) -> State:
        state = self.env.reset(rng)
        metrics = {"done_at": -1, 'step': 0}
        return state.replace(metrics=metrics)

    def step(self, state, action) -> State:

        def f_done(operand):
            state = operand
            return state

        def f_not_done(operand):
            state = operand
            state = self.env.step(state, action)
            step = state.metrics['step']
            done_at = jax.lax.cond(
                state.done, lambda _: step, lambda _: -1, operand=None
            )
            metrics = {"done_at": done_at, 'step': step + 1}
            return state.replace(metrics=metrics)
        
        return jax.lax.cond(state.done, f_done, f_not_done, operand=state)
    
class DoneStopWrapper(Wrapper):
    """Stops batched Brax envs that are done"""
    def __init__(self, env, stop_at_fail=True):
        super().__init__(env)
        self.stop_at_fail = stop_at_fail

    def reset(self, rng: jp.ndarray) -> State:
        return self.env.reset(rng)

    def step(self, state: State, action: jp.ndarray) -> State:
        info = dict(state.info)
        metrics = dict(state.metrics)
        state_next = self.env.step(state, action)

        def where_done(x, y):
            done = state.done
                
            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return jp.where(done, x, y)
        
        state_next = jax.tree_util.tree_map(where_done, 
                                            state.replace(reward=jp.zeros_like(state.reward), info=info, metrics=metrics), 
                                            state_next)
        return state_next