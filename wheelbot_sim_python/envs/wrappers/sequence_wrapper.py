from brax.envs.base import State, Wrapper, Env
import jax
from jax import numpy as jp
from ...reach_avoid.sampler_base import ReachAvoidSampler
from ...reach_avoid import util as reach_avoid_util



class SequenceWrapper(Wrapper):
    def __init__(self, env: Env, sampler: ReachAvoidSampler,
                  eval: bool = False, deploy: bool = False, partial_rewards=False):
        """
        Wrapper that samples and evaluates reach-avoid sequences. 
        The flattened reach-avoid sequence is added to the observation.
        In eval mode and deploy mode, only reach avoid sequences of full length are sampled
        In deploy mode a new reach-avoid sequence is sampled after a violation occurs
        """
        assert not (eval and deploy)
        super().__init__(env)
        self.reset_keys = {'reach_avoid_state', 'eval_status'} | self.env.reset_keys
        self.sampler = sampler
        self.eval = eval
        self.deploy = deploy
        self.partial_rewards = partial_rewards
        self.env_observation_size = self.env.observation_size
        self.sequence_size = self.sampler.sequence_size
        self.num_assignments = self.env.context.num_assignments
        self.dummy_state = self.reset(rng=jax.random.PRNGKey(0))
        self._observation_size = self.dummy_state.obs.size
        self.extra_fields = ('reach_avoid_state', 'eval_status',) + self.env.extra_fields

        
    @property
    def observation_size(self) -> int:
        return self._observation_size
    
    def add_reach_avoid(self, state: State, rng) -> State:
        env_obs = state.obs[:self.env_observation_size]
        if self.eval or self.deploy:
            reach_avoid_seq = self.sampler.sample_eval(env_obs, rng)
        else:
            reach_avoid_seq = self.sampler.sample_training(env_obs, rng)

        reach_avoid_state = reach_avoid_util.create_reach_avoid_state(reach_avoid_seq)
        reach_avoid_seq_flat = reach_avoid_util.flatten_reach_avoid(reach_avoid_seq)
        info = state.info
        info['reach_avoid_state'] = reach_avoid_state
        obs = jp.array([*env_obs, *reach_avoid_seq_flat])
        return state.replace(obs=obs, info=info)
    
    @staticmethod
    def update_sequence_state(state: State, partial_rewards: bool = False) -> State:
        active_assignment = state.info['active_assignment']
        reach_avoid_state = state.info['reach_avoid_state']
        reach_avoid_state = reach_avoid_util.update_reach_avoid(reach_avoid_state, active_assignment)
        info = state.info
        info['reach_avoid_state'] = reach_avoid_state
        info['eval_status'] = reach_avoid_state.status
        reach_avoid_seq_flat = reach_avoid_util.flatten_reach_avoid(reach_avoid_state.reach_avoid)
        obs = jp.array([*state.obs, *reach_avoid_seq_flat])
        fail = reach_avoid_state.is_fail
        success = reach_avoid_state.is_success
        progress = info['reach_avoid_state'].progress
        done = jp.where(jp.logical_or(state.done, jp.logical_or(success, fail)), 1.0, 0.0)
        if partial_rewards:
            reward = 1. * progress - 1. * fail
        else:
            reward = 1. * success - 1. * fail
        return state.replace(obs=obs, reward=reward, done=done, info=info)

    def reset(self, rng: jp.ndarray) -> State:
        reset_rng, reach_avoid_rng = jax.random.split(rng, 2)
        state = self.env.reset(reset_rng)
        state.info['eval_status'] = reach_avoid_util.EvalStatus.ONGOING
        state.info['progress'] = 0.0
        return self.add_reach_avoid(state, reach_avoid_rng)

    def reset_reach_avoid(self, state: State) -> State:
        state = self.add_reach_avoid(state, None)

        info = state.info
        info['eval_status'] = reach_avoid_util.EvalStatus.ONGOING
        info['progress'] = 0.0
        done = 0.
        reward = 0.

        return state.replace(info=info, done=done, reward=reward)
    
    def repeat(self, state: State, rng: jp.ndarray) -> State:
        repeat_rng, reach_avoid_rng = jax.random.split(rng, 2)
        state = self.env.repeat(state, repeat_rng)
        return self.add_reach_avoid(state, reach_avoid_rng)


    def step(self, state, action) -> State:
        env_obs = state.obs[:self.env.observation_size]
        state = state.replace(obs=env_obs)
        state = self.env.step(state, action)
        state = self.update_sequence_state(state, self.partial_rewards)
        reach_avoid_state = state.info['reach_avoid_state']

        if self.deploy:
            state = jax.lax.cond(reach_avoid_state.is_fail,
                                lambda s: self.add_reach_avoid(s, None),
                                lambda s: s,
                                state)
        
        return state