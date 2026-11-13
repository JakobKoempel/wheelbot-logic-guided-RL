from brax.envs.base import State, Wrapper, Env
import jax
from jax import vmap
from jax import numpy as jp
from ...reach_avoid import util as reach_avoid_util
from ...automata.dba import DBA
from typing import Callable

Observation = jp.ndarray
EnvObservation = jp.ndarray
Action = jp.ndarray
Value = jp.ndarray


class DBAWrapper(Wrapper):
    def __init__(self, env: Env, dba: DBA):
        super().__init__(env)
        self.dba = dba
        self.value_inference_fn = lambda x: 0 #dummy inference function to initialize observation size
        dummy_state = self.reset(rng=jax.random.PRNGKey(0))
        self._observation_size = dummy_state.obs.size
        self.env_observation_size = self.env.observation_size
        self.reach_avoid_observation_size = self._observation_size - self.env_observation_size
        self.sequence_size = self.dba.sequence_size
        self.value_function_set = False

    def set_value_inference_fn(self, value_inference_fn: Callable[[Observation], Value]):
        self.value_function_set = True
        self.value_inference_fn = value_inference_fn

    @property
    def observation_size(self) -> int:
        return self._observation_size

    def add_reach_avoid(self, env_obs: EnvObservation, dba_state: int) -> Observation:
        accepting_sequences = self.dba.get_all_reach_avoid_at(dba_state)
        batch_dim = self.dba.max_num_accepting_paths
        accepting_sequences_flat = vmap(reach_avoid_util.flatten_reach_avoid)(accepting_sequences)
        #env_obs = jp.array([*env_obs, 0.]) #DIRTY QUICK FIX! DBA wrapper and repeat wrapper should be merge into one wrapper
        env_obs = jp.broadcast_to(env_obs, (batch_dim, env_obs.shape[0]))
        batched_obs = jp.concatenate([env_obs, accepting_sequences_flat], axis=1)
        values = vmap(self.value_inference_fn)(batched_obs)
        accepting_seqs_mask = self.dba.accepting_reach_avoid_mask[dba_state, :]
        values_masked = jp.where(accepting_seqs_mask == 0., -100., values)
        max_index = jp.argmax(values_masked)
        res = batched_obs[max_index, :]
        #res = jp.concatenate([res[:14], res[15:]]) #Remove added num_repeat obs because repeat wrapper will add it again
        return res, max_index
 
    def reset(self, rng: jp.ndarray) -> State:
        state = self.env.reset(rng)
        env_obs = state.obs
        obs, path_index = self.add_reach_avoid(env_obs, self.dba.init_state)

        info = state.info
        info['dba_state'] = self.dba.init_state
        info['prev_dba_state'] = -1
        info['eval_status'] = reach_avoid_util.EvalStatus.ONGOING
        info['path_index'] = path_index
        done = 0.
        reward = 0.

        return state.replace(obs=obs, info=info, done=done, reward=reward)
    
    def reset_reach_avoid(self, state: State) -> State:
        env_obs = state.obs[:self.env_observation_size]
        obs, path_index = self.add_reach_avoid(env_obs, self.dba.init_state)

        info = state.info
        info['dba_state'] = self.dba.init_state
        info['prev_dba_state'] = -1
        info['eval_status'] = reach_avoid_util.EvalStatus.ONGOING
        info['path_index'] = path_index
        done = 0.
        reward = 0.

        return state.replace(obs=obs, info=info, done=done, reward=reward)
    
    def repeat(self, state: State, rng: jp.ndarray) -> State:
        state = self.env.repeat(state, rng)
        return self.reset_reach_avoid(state)
    
    def step(self, state, action) -> State:
        env_obs = state.obs[:self.env.observation_size]
        reach_avoid_seq_flat = state.obs[-self.reach_avoid_observation_size:]
        state = state.replace(obs=env_obs)
        state = self.env.step(state, action)
        env_obs = state.obs

        info = state.info
        active_assignment = info['active_assignment']
        dba_state = info['dba_state']
        next_dba_state = self.dba.step(dba_state, active_assignment)
        info['prev_dba_state'] = dba_state
        info['dba_state'] = next_dba_state

        progress = dba_state != next_dba_state
        
        obs, path_index = jax.lax.cond(progress,
                            lambda: self.add_reach_avoid(env_obs, next_dba_state),
                            lambda: (jp.array([*env_obs, *reach_avoid_seq_flat]), info['path_index']))

        success = self.dba.is_accepting_state(next_dba_state)
        fail = self.dba.is_sink_state(next_dba_state)
    
        info['eval_status'] = jp.where(success, reach_avoid_util.EvalStatus.SUCCESS,
                                      jp.where(fail, reach_avoid_util.EvalStatus.FAIL,
                                               reach_avoid_util.EvalStatus.ONGOING))
        info['path_index'] = path_index
        
        done = jp.where(jp.logical_or(success, fail), 1.0, 0.0)
        
        return state.replace(obs=obs, info=info, done=done)