from brax.envs.base import State, Wrapper, Env
import jax
from jax import numpy as jp
from ...reach_avoid.util import EvalStatus
from typing import Optional


class RepeatWrapper(Wrapper):
    def __init__(self, env: Env, num_repeat: Optional[int] = None, eval: bool = False, deploy: bool = False):
        """
        Wrapper that divides one episode into several subepiodes. A new subepisode is started with
        an env.repeat called instead of an env.reset call. This enables the resetting of reach-avoid sequences
        or DBAs without resetting the Wheelbots internal state.
        In deploy mode, repeats are unlimited.
        """
        assert hasattr(env, 'repeat')
        assert num_repeat >= 0
        assert not (eval and deploy)
        super().__init__(env)
        self.num_repeat = num_repeat
        self.eval = eval
        self.deploy = deploy
        self.reset_keys = {'num_repeat'} | self.env.reset_keys

    @property
    def env_observation_size(self) -> int:
        return self.env.env_observation_size + 1 #Added num_repeat to the observation
    
    @property
    def observation_size(self) -> int:
        return self.env.observation_size + 1 #Added num_repeat to the observation

    def reset(self, rng: jp.ndarray) -> State:
        reset_rng, repeat_rng, num_repeat_rng = jax.random.split(rng, 3)
        state = self.env.reset(reset_rng)

        if self.eval or self.deploy: 
            state.info['num_repeat'] = self.num_repeat - 1
        else:
            state.info['num_repeat'] = jax.random.randint(num_repeat_rng, shape=(), minval=0, maxval=self.num_repeat)
        state.info['repeat_rng'] = repeat_rng
        state.info['repeat_done'] = 0.0
        state.info['num_successes'] = 0.0
        state.info['num_failures'] = 0.0
        obs = jp.array([*state.obs[:self.env.env_observation_size], state.info['num_repeat'], *state.obs[self.env.env_observation_size:]])
        return state.replace(obs=obs)
    
    def repeat(self, state: State) -> State:
        new_repeat_rng, repeat_rng = jax.random.split(state.info['repeat_rng'], 2)
        reward = state.reward
        metrics = state.metrics
        info = state.info

        state = self.env.repeat(state, repeat_rng)
        info.update(state.info)
        info['repeat_rng'] = new_repeat_rng
        info['num_repeat'] = info['num_repeat'] - 1
        if self.deploy:
            info['num_repeat'] = jp.max(jp.array([info['num_repeat'], 1]))
        info['repeat_done'] = 1.0
        metrics.update(state.metrics)
        obs = jp.array([*state.obs[:self.env.env_observation_size], info['num_repeat'], *state.obs[self.env.env_observation_size:]])
        return state.replace(obs=obs, reward=reward, done=0.0, info=info, metrics=metrics)
    
    def reset_reach_avoid(self, state: State) -> State:
        reward = state.reward
        metrics = state.metrics
        info = state.info

        state = self.env.reset_reach_avoid(state)
        info.update(state.info)
        metrics.update(state.metrics)
        
        obs = jp.array([*state.obs[:self.env.env_observation_size], info['num_repeat'], *state.obs[self.env.env_observation_size:]])
        return state.replace(obs=obs, reward=reward, done=0.0, info=info, metrics=metrics)
    
    def step(self, state, action) -> State:
        num_repeat = state.info['num_repeat']
        state = self.env.step(state, action)
        state.info['repeat_done'] = 0.0
        success = state.info['eval_status'] == EvalStatus.SUCCESS
        fail = state.info['eval_status'] == EvalStatus.FAIL
        state.info['num_successes'] = state.info['num_successes'] + success
        state.info['num_failures'] = state.info['num_failures'] + fail
        reward = 1. * success * (num_repeat == 0) - 1. * fail * (1 - self.eval or self.deploy)
        if self.deploy:
            done = 0.0
        else:
            done = state.done
        obs = jp.array([*state.obs[:self.env.env_observation_size], num_repeat, *state.obs[self.env.env_observation_size:]])
        state = state.replace(obs=obs, reward=reward, done=done)
        
        state = jax.lax.cond(jp.logical_and(success, num_repeat > 0),
                            lambda s: self.repeat(s),
                            lambda s: s,
                            state)
        if self.deploy:
            state = jax.lax.cond(fail, 
                                lambda s: self.reset_reach_avoid(s), 
                                lambda s: s, 
                                state)

        if self.deploy:
            # deploy mode never ends on its own, except when the env reports that all its setpoints are done
            state = state.replace(done=state.info.get('all_setpoints_reached', 0.0))

        return state