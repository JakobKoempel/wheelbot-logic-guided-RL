from typing import Callable, Optional, Tuple

from brax.base import System
from brax.envs.base import Env, Wrapper
from jax import numpy as jp
from brax.envs.wrappers.training import (
    VmapWrapper,
    DomainRandomizationVmapWrapper,
)
from .auto_reset_wrapper import AutoResetWrapper
from .done_wrapper import DoneStopWrapper
from .episode_wrapper import EpisodeWrapper


def train_wrap(
    env: Env,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[Callable[[System], Tuple[System, System]]] = None,
) -> Wrapper:
    """Reimplementation of brax.envs.wrappers.training.wrap that uses the custom AutoRestWrapper for DeepLTL.

    Args:
      env: environment to be wrapped
      episode_length: length of episode
      action_repeat: how many repeated actions to take per step
      randomization_fn: randomization function that produces a vectorized system
        and in_axes to vmap over

    Returns:
      An environment that is wrapped with Episode and AutoReset wrappers.  If the
      environment did not already have batch dimensions, it is additional Vmap
      wrapped.
    """
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
) -> Wrapper:
    """
    Args:
      env: environment to be wrapped
      episode_length: length of episode
      action_repeat: how many repeated actions to take per step
    """
    env = VmapWrapper(env)
    env = DoneStopWrapper(env)
    env = EpisodeWrapper(env, episode_length, action_repeat)
    return env