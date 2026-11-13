from typing import Callable, Optional, Tuple

from brax.base import System
from brax.envs.base import Env, State, Wrapper
import jax
from jax import numpy as jp
from brax.envs.wrappers.training import (
    VmapWrapper,
    EpisodeWrapper,
    DomainRandomizationVmapWrapper,
)
from .auto_reset_wrapper import AutoResetWrapper


class DoneStopWrapper(Wrapper):
    """Stops batched Brax envs that are done"""
    def __init__(self, env):
        super().__init__(env)

    def reset(self, rng: jp.ndarray) -> State:
        return self.env.reset(rng)

    def step(self, state: State, action: jp.ndarray) -> State:
        state_next = self.env.step(state, action)

        def where_done(x, y):
            done = state.done

            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return jp.where(done, x, y)

        state_next = jax.tree.map(where_done, state, state_next)
        return state_next


def train_wrap(
    env: Env,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[Callable[[System], Tuple[System, System]]] = None,
) -> Wrapper:
    if randomization_fn is None:
        env = VmapWrapper(env)
    else:
        env = DomainRandomizationVmapWrapper(env, randomization_fn)
    env = EpisodeWrapper(env, episode_length, action_repeat)
    env = AutoResetWrapper(env)
    return env


def eval_wrap(
    env: Env,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[Callable[[System], Tuple[System, System]]] = None,
) -> Wrapper:
    if randomization_fn is None:
        env = VmapWrapper(env)
    else:
        env = DomainRandomizationVmapWrapper(env, randomization_fn)
    env = DoneStopWrapper(env)
    env = EpisodeWrapper(env, episode_length, action_repeat)
    return env
