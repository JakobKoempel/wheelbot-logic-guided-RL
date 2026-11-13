from brax.training import distribution
from brax.training import types
from brax.training.agents.sac.networks import make_inference_fn
from brax.training.agents.sac.networks import SACNetworks
import jax.numpy as jp
from brax.training.networks import FeedForwardNetwork
from .deep_ltl_nets import create_deep_ltl_policy_net, create_deep_ltl_q_net
import jax


def make_deep_ltl_sac_networks(
    observation_size: int,
    env_observation_size: int,
    action_size: int,
    sequence_size: int,
    num_assignments: int,
    cfg,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
) -> SACNetworks:
    """Make SAC networks."""
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    policy_network = make_deep_ltl_policy_network(
        observation_size=observation_size,
        env_observation_size=env_observation_size,
        sequence_size=sequence_size,
        num_assignments=num_assignments,
        param_size=parametric_action_distribution.param_size,
        cfg=cfg,
        preprocess_observations_fn=preprocess_observations_fn
    )
    q_network = make_deep_ltl_q_network(
        observation_size=observation_size,
        env_observation_size=env_observation_size,
        sequence_size=sequence_size,
        num_assignments=num_assignments,
        action_size=action_size,
        cfg=cfg,
        preprocess_observations_fn=preprocess_observations_fn
    )
    return SACNetworks(
        policy_network=policy_network,
        q_network=q_network,
        parametric_action_distribution=parametric_action_distribution,
    )

def make_deep_ltl_policy_network(
    observation_size: int,
    env_observation_size: int,
    sequence_size: int,
    num_assignments: int,
    param_size: int,
    cfg,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
) -> FeedForwardNetwork:
    """Creates a DeepLTL policy network."""

    policy_module = create_deep_ltl_policy_net(
        sequence_size=sequence_size,
        num_assignments=num_assignments,
        distr_param_size=param_size,
        cfg=cfg
    )

    def apply(processor_params, policy_params, obs):
        #exclude the reach_avoid sequence in the normalization to not destroy its encoding
        if(obs.ndim == 1):
            obs = obs[None, :]
        env_obs = obs[:, :env_observation_size]
        reach_avoid_obs = obs[:, env_observation_size:].astype(jp.int32)
        env_obs = preprocess_observations_fn(env_obs, processor_params)
        return policy_module.apply(policy_params, env_obs, reach_avoid_obs)
    
    def init(key):
        dummy_obs = jp.zeros((1, observation_size))
        dummy_env_obs = dummy_obs[:, :env_observation_size]
        dummy_reach_avoid_obs = dummy_obs[:, env_observation_size:].astype(jp.int32)
        return policy_module.init(key, dummy_env_obs, dummy_reach_avoid_obs)
    
    return FeedForwardNetwork(
        init=init, apply=apply
    )

def make_deep_ltl_q_network(
    observation_size: int,
    env_observation_size: int,
    sequence_size: int,
    num_assignments: int,
    action_size: int,
    cfg,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
) -> FeedForwardNetwork:
    """Creates a DeepLTL Q network with two Q modules"""

    q_module = create_deep_ltl_q_net(
        sequence_size=sequence_size,
        num_assignments=num_assignments,
        cfg=cfg
    )

    def apply(processor_params, q_params, obs, actions):
        #exclude the reach_avoid sequence in the normalization to not destroy its encoding
        if(obs.ndim == 1):
            obs = obs[None, :]
        env_obs = obs[:, :env_observation_size]
        reach_avoid_obs = obs[:, env_observation_size:].astype(jp.int32)
        env_obs = preprocess_observations_fn(env_obs, processor_params)
        return q_module.apply(q_params, env_obs, reach_avoid_obs, actions)
    
    def init(key):
        dummy_obs = jp.zeros((1, observation_size))
        dummy_env_obs = dummy_obs[:, :env_observation_size]
        dummy_reach_avoid_obs = dummy_obs[:, env_observation_size:].astype(jp.int32)
        dummy_action = jp.zeros((1, action_size))
        return q_module.init(key, dummy_env_obs, dummy_reach_avoid_obs,dummy_action)

    return FeedForwardNetwork(
        init=init, apply=apply
    )

def make_value_inference_fn(sac_network: SACNetworks):

    def make_value_fn(processor_params, policy_params, q_params) -> float:
        make_policy_fn = make_inference_fn(sac_network)
        policy = make_policy_fn((processor_params, policy_params), deterministic=True)
        def value_fn(obs: jp.ndarray):
            action, _ = policy(obs, None)
            value = sac_network.q_network.apply(processor_params, q_params, obs, action)
            value = jp.squeeze(value)
            min_value = jp.min(value)
            return min_value
        
        return value_fn

    return make_value_fn


if __name__ == "__main__":
    from ..reach_avoid.util import flatten_reach_avoid, batch_trees
    from omegaconf import OmegaConf
    from pathlib import Path
    from .deep_ltl_nets import update_shared_parameters

    config_path = Path(__file__).parent / ".." / "config" / "deep_ltl" / "deep_ltl_net.yaml"
    cfg = OmegaConf.load(config_path)

    #create batched reach avoid sequences
    padding = 0
    eps = 1
    assignment_1 = 2
    assignment_2 = 3
    assignment_3 = 4

    set_1 = jp.array([assignment_1, assignment_2, padding, padding])
    padding_set = jp.array([padding, padding, padding, padding])

    seq_1 = jp.array([set_1, padding_set, padding_set])
    seq_2 = jp.array([padding_set, set_1, padding_set])
    seq_3 = jp.array([padding_set, padding_set, set_1])
    seq = jp.array([padding_set, padding_set, padding_set])

    reach_1 = {'seq': seq_1, 'set_sizes': jp.array([2, 0, 0])}
    reach_2 = {'seq': seq_2, 'set_sizes': jp.array([0, 2, 0])}
    reach_3 = {'seq': seq_3, 'set_sizes': jp.array([0, 0, 2])}
    avoid = {'seq': seq, 'set_sizes': jp.array([0, 0, 0])}
    reach_avoid_1 = {'reach': reach_1, 'avoid': avoid, 'length': jp.array([1])}
    reach_avoid_2 = {'reach': reach_2, 'avoid': avoid, 'length': jp.array([2])}
    reach_avoid_3 = {'reach': reach_3, 'avoid': avoid, 'length': jp.array([3])}
    batched_reach_avoid = batch_trees(reach_avoid_1, reach_avoid_2, reach_avoid_3)

    #flatten reach avoid sequences
    flatten_reach_avoid_batch = jax.vmap(flatten_reach_avoid)
    batched_reach_avoid_flat = flatten_reach_avoid_batch(batched_reach_avoid)

    env_obs = jp.array([[1,2,3],
                        [1,1,1],
                        [0,0,0]])
    
    obs = jp.concatenate([env_obs, batched_reach_avoid_flat], axis=-1)

    deep_ltl_sac_networks = make_deep_ltl_sac_networks(
        observation_size=obs.shape[-1],
        env_observation_size=3,
        action_size=3,
        sequence_size=3,
        num_assignments=5,
        cfg=cfg
    )

    policy_init = jax.jit(deep_ltl_sac_networks.policy_network.init)
    q_init = jax.jit(deep_ltl_sac_networks.q_network.init)
    key = jax.random.PRNGKey(0)
    policy_params = policy_init(key)
    q_params = q_init(key)

    policy_params = update_shared_parameters(policy_params, q_params)

    def pytree_equal(pytree1, pytree2):
        leaves1, _ = jax.tree_util.tree_flatten(pytree1)
        leaves2, _ = jax.tree_util.tree_flatten(pytree2)
        return all(jp.array_equal(a, b) for a, b in zip(leaves1, leaves2))
    
    assert pytree_equal(policy_params['params']['env_net'], q_params['params']['env_net'])
    assert pytree_equal(policy_params['params']['reach_avoid_net'], q_params['params']['reach_avoid_net'])
    print('All tests passed')
