import jax.numpy as jp
from brax.training.types import PRNGKey 
from typing import Tuple
import jax
from jax import flatten_util
from brax.training.replay_buffers import ReplayBufferState, ReplayBuffer
from brax.training.types import Transition
from brax.envs.base import Env
import hydra
import omegaconf

class EnvReplayBuffer():

    def __init__(
        self,
        env: Env, 
        max_replay_size: int, 
        dummy_data_sample: Transition, 
        max_episode_length: int,
        hindsight_experience_replay: bool = False
        ):
        self._max_episode_length = max_episode_length
        self._flatten_fn = jax.vmap(lambda x: flatten_util.ravel_pytree(x)[0])
        dummy_flatten, self._unflatten_fn = flatten_util.ravel_pytree(
            dummy_data_sample
        )
        self._unflatten_fn = jax.vmap(self._unflatten_fn)
        data_size = len(dummy_flatten)
        self._data_shape = (max_replay_size + max_episode_length, data_size)
        self._max_replay_size = max_replay_size
        self._data_dtype = dummy_flatten.dtype
    
        if hindsight_experience_replay:
            self._create_her_episode = env.create_her_episode
            self._her_applicable = env.her_applicable

    def init(self, key: PRNGKey) -> ReplayBufferState:
        return ReplayBufferState(
            data=jp.zeros(self._data_shape, self._data_dtype),
            sample_position=jp.zeros((), jp.int32),
            insert_position=jp.zeros((), jp.int32),
            key=key,
        )
    
    def insert(self, buffer_state: ReplayBufferState, samples: Transition, num_samples: int) -> ReplayBufferState:
        """Insert samples[num_samples:] in the replay buffer.
            samples: Sample to insert with a leading batch size.
            num_samples: Number of samples that is actually inserted 
        """
        update = self._flatten_fn(samples)
        if len(update) > self._max_episode_length:
            raise ValueError(f'Cannot insert more samples than there are in an episode')
        data = buffer_state.data
        if data.shape != self._data_shape:
            raise ValueError(
                f'buffer_state.data.shape ({buffer_state.data.shape}) '
                f"doesn't match the expected value ({self._data_shape})"
        )
        num_samples = jp.minimum(num_samples, self._max_episode_length)
        # If needed, roll the buffer to make sure there's enough space to fit
        # `update` after the current position.
        position = buffer_state.insert_position
        roll = jp.minimum(0, self._max_replay_size - position - num_samples)
        data = jax.lax.cond(
            roll, lambda: jp.roll(data, roll, axis=0), lambda: data
        )
        position = position + roll

        # Update the buffer and the control numbers.
        data = jax.lax.dynamic_update_slice_in_dim(data, update, position, axis=0)
        position = (position + num_samples) % (self._max_replay_size + 1)
        sample_position = jp.maximum(0, buffer_state.sample_position + roll)

        return buffer_state.replace(
            data=data,
            insert_position=position,
            sample_position=sample_position,
        )
    
    def size(self, buffer_state: ReplayBufferState) -> int:
        return buffer_state.insert_position - buffer_state.sample_position
    
    def add_hindsight_experience(self, buffer_state: ReplayBufferState) -> ReplayBufferState:
        unflattened_data =  self._unflatten_fn(buffer_state.data)
        last_transition = jax.tree.map(lambda x: x[buffer_state.insert_position - 1], unflattened_data)

        def apply_her(buffer_state: ReplayBufferState):
            #compute last and second to last done to find the last episode in the buffer
            data = buffer_state.data
            dones = 1 - self._unflatten_fn(data).discount
            done_idx = jp.arange(dones.shape[0])
            buffer_range = (done_idx >= buffer_state.sample_position) & (done_idx < buffer_state.insert_position)
            dones = jp.where(buffer_range, dones, 0)
            idxs = jp.nonzero(dones, size=dones.shape[0], fill_value=-1)[0]
            num_valid = jp.sum(idxs != -1)
            second_to_last_done = jp.where(num_valid >= 2, idxs[num_valid - 2], buffer_state.sample_position - 1)
            last_done = jp.where(num_valid >= 1, idxs[num_valid - 1], buffer_state.sample_position - 1)
            
            #take the last episode from the buffer
            last_episode_start = second_to_last_done + 1
            last_episode_length = last_done - second_to_last_done
            last_episode = jax.lax.dynamic_slice_in_dim(data, start_index=last_episode_start, slice_size=self._max_episode_length)
            
            #change last episode to hindsight experience replay episode
            unflattened_last_episode = self._unflatten_fn(last_episode)
            unflattened_her_episode = self._create_her_episode(unflattened_last_episode, last_episode_length)

            #compute first done in hindsight experience replay episode
            her_dones = 1 - unflattened_her_episode.discount
            her_done_idx = jp.arange(her_dones.shape[0])
            her_episode_range = (her_done_idx < last_episode_length)
            her_dones = jp.where(her_episode_range, her_dones, 0)
            idxs = jp.nonzero(her_dones, size=her_dones.shape[0], fill_value=-1)[0]
            num_valid = jp.sum(idxs != -1)
            first_done = jp.where(num_valid >= 1, idxs[0], last_episode_length - 1)
        
            #add hindsight experience replay to buffer
            her_episode_length = first_done + 1
            buffer_state = self.insert(buffer_state, unflattened_her_episode, her_episode_length)
            return buffer_state

        return jax.lax.cond(self._her_applicable(last_transition), apply_her, lambda bs: bs, buffer_state)


class BatchedEnvsReplayBuffer(ReplayBuffer[ReplayBufferState, Transition]):

    def __init__(
            self, 
            env: Env, 
            max_replay_size: int,
            num_envs: int,
            dummy_data_sample: Transition,
            sample_batch_size: int,
            max_episode_length: int,
            hindsight_experience_replay: bool = False,
        ):
        max_replay_size = ((max_replay_size - 1) // num_envs + 1) * num_envs # round max_replay_size to the next biggest number divisible by num_envs
        self._max_replay_size_per_env_buffer = max_replay_size // num_envs
        if hindsight_experience_replay:
            assert self._max_replay_size_per_env_buffer >= max_episode_length, 'replay size of each environment buffer must at least store one episode'
        self._max_replay_size = max_replay_size
        self._num_envs = num_envs
        self._sample_batch_size = sample_batch_size
        self._max_episode_length = max_episode_length
        self.hindsight_experience_replay = hindsight_experience_replay

        self._flatten_fn = jax.vmap(lambda x: jax.flatten_util.ravel_pytree(x)[0])
        dummy_flatten, self._unflatten_fn = jax.flatten_util.ravel_pytree(dummy_data_sample)
        self._unflatten_fn = jax.vmap(self._unflatten_fn)
        self._data_dtype = dummy_flatten.dtype

        env_replay_buffer = EnvReplayBuffer(
            env=env,
            max_replay_size=self._max_replay_size_per_env_buffer,
            dummy_data_sample=dummy_data_sample,
            max_episode_length=max_episode_length,
            hindsight_experience_replay=hindsight_experience_replay)
        
        self._data_shape = (num_envs,) + env_replay_buffer._data_shape
        self.vmap_env_buffer_init = jax.vmap(env_replay_buffer.init)
        self.vmap_env_buffer_insert = jax.vmap(env_replay_buffer.insert)
        self.vmap_env_buffer_add_hindsight_exp = jax.vmap(env_replay_buffer.add_hindsight_experience)
        

    def init(self, rng: PRNGKey) -> ReplayBufferState:
        env_keys = jax.random.split(rng, self._num_envs)
        return self.vmap_env_buffer_init(env_keys)
    
    def insert_internal(self, buffer_state: ReplayBufferState, samples: Transition) -> ReplayBufferState:
        """Insert data in the replay buffer.
            samples: shape [num_envs, data_size]
        """
        if buffer_state.data.shape != self._data_shape:
            raise ValueError(
                f'buffer_state.data.shape ({buffer_state.data.shape}) '
                f"doesn't match the expected value ({self._data_shape})"
            )

        # only one sample is inserted per env but EnvReplayBuffer can only insert batched samples
        samples = jax.tree.map(lambda x: x[:, None, ...], samples)
        ones = jp.ones(shape=(self._num_envs,), dtype=jp.int32)
        buffer_state = self.vmap_env_buffer_insert(buffer_state, samples, ones)
        
        if self.hindsight_experience_replay:
            return self.vmap_env_buffer_add_hindsight_exp(buffer_state)
        else:
            return buffer_state

    def sample_internal(self, buffer_state: ReplayBufferState) -> Tuple[ReplayBufferState, Transition]:
        """Sample uniformly over all environments"""

        rng = buffer_state.key[0]
        rng, env_rng, id_rng = jax.random.split(rng, 3)
        id_rng = jax.random.split(id_rng, self._sample_batch_size)
        size_per_env = buffer_state.insert_position - buffer_state.sample_position
        total_size = jp.sum(size_per_env)
        prob_per_env = size_per_env / total_size
        env_idx = jax.random.choice(env_rng, self._num_envs, shape=(self._sample_batch_size,), p=prob_per_env)
        
        def sample_env_element(env_idx: int, id_rng: PRNGKey):
            return jax.random.randint(id_rng, shape=(), minval=buffer_state.sample_position[env_idx], maxval=buffer_state.insert_position[env_idx])

        idx = jax.vmap(sample_env_element)(env_idx, id_rng)
        sample = buffer_state.data[env_idx, idx, :] 

        n_key = buffer_state.key.at[0].set(rng)
        return buffer_state.replace(key=n_key), self._unflatten_fn(sample)

    def size(self, buffer_state: ReplayBufferState) -> int:
        size_per_env = buffer_state.insert_position - buffer_state.sample_position
        return jp.sum(size_per_env)


@hydra.main(config_path="../../config", config_name="main", version_base=None)
def main(cfg: omegaconf.DictConfig) -> None:
    import mediapy as media
    from ...reach_avoid.zone_env_sampler import ZoneReachSampler
    from ...envs.wrappers.sequence_wrapper import SequenceWrapper
    from brax.envs.wrappers.training import VmapWrapper
    from ...envs.wheelbot_zone_env import WheelbotZoneEnv
    from brax.envs.wrappers.training import EpisodeWrapper
    from ...envs.wrappers.auto_reset_wrapper import AutoResetWrapper

    def actor_step(env_state, action, extra_fields):
        nstate = env.step(env_state, action)
        state_extras = {x: nstate.info[x] for x in extra_fields}
        return nstate, Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
            observation=env_state.obs,
            action=action,
            reward=nstate.reward,
            discount=1 - nstate.done,
            next_observation=nstate.obs,
            extras={'state_extras': state_extras},
    )

    num_envs = 1
    max_episode_length = 2
    env = WheelbotZoneEnv(num_repeat=3, episode_length=650, raw_actions=False, jitable_step=True, cfg=cfg)
    sampler = ZoneReachSampler(sequence_size=1)
    env = SequenceWrapper(env, sampler, eval=True)
    env = VmapWrapper(env)
    env = EpisodeWrapper(env, episode_length=max_episode_length, action_repeat=1)
    env = AutoResetWrapper(env)

    
    dummy_transition = Transition(
        observation=jp.zeros(env.observation_size),
        action=jp.zeros(env.action_size),
        reward=0.0,
        discount=1.0,
        next_observation=jp.zeros(env.observation_size),
        extras={'state_extras' : {x: env.dummy_state.info[x] for x in env.extra_fields} | {'truncation': 0.}})
    
    buffer = BatchedEnvsReplayBuffer(
        env, 
        max_replay_size=5, 
        num_envs=num_envs, 
        dummy_data_sample=dummy_transition,
        sample_batch_size=5,
        max_episode_length=max_episode_length,
        hindsight_experience_replay=True)
    
    rng = jax.random.PRNGKey(2)
    buffer_rng, rng = jax.random.split(rng)
    buffer_state = buffer.init(buffer_rng)
    reset_rng = jax.random.split(rng, num_envs)
    state_0 = env.reset(reset_rng)
    print(state_0.info['reach_avoid_state'].reach_avoid['reach']['seq'])
    state_1, transition_1 = actor_step(state_0, jp.array([[1, 3.14]]), env.extra_fields + ('truncation',))
    buffer_state = buffer.insert(buffer_state, transition_1)
    state_2, transition_2 = actor_step(state_1, jp.array([[1.25, 0]]), env.extra_fields + ('truncation',))
    buffer_state = buffer.insert(buffer_state, transition_2)
    
    env_buffer_state_0 = buffer._unflatten_fn(buffer_state.data[0])
    env_buffer_state_1 = buffer._unflatten_fn(buffer_state.data[1])
    env_buffer_state_2 = buffer._unflatten_fn(buffer_state.data[2])

    print('Finished')

if __name__ == '__main__':
    main()