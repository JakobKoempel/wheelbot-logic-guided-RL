import jax
import jax.numpy as jp
from flax import linen as nn
from brax.training.networks import MLP
from .reach_avoid_net import ReachAvoidNet
    
class DeepLTLPolicyNet(nn.Module):
    """DeepLTL policy network"""
    env_net: nn.Module
    reach_avoid_net: ReachAvoidNet
    head_net: MLP

    @nn.compact
    def __call__(self, env_obs: jp.ndarray, reach_avoid_obs: jp.ndarray) -> jp.ndarray:
        env_embeddings = self.env_net(env_obs)
        reach_avoid_embeddings, _ = self.reach_avoid_net(reach_avoid_obs)
        obs_embeddings = jp.concatenate([env_embeddings, reach_avoid_embeddings], axis=-1)
        out = self.head_net(obs_embeddings)
        return out
    
class DeepLTLQNet(nn.Module):
    env_net: nn.Module
    reach_avoid_net: ReachAvoidNet
    head_net_0: MLP
    head_net_1: MLP
    """"DeepLTL Q network with shared env and reach-avoid layer and multiple heads for critics."""

    @nn.compact
    def __call__(self, env_obs: jp.ndarray, reach_avoid_obs: jp.ndarray, action: jp.ndarray,) -> jp.ndarray:
        env_embeddings = self.env_net(env_obs)
        reach_avoid_embeddings, _ = self.reach_avoid_net(reach_avoid_obs)
        obs_embeddings = jp.concatenate([env_embeddings, reach_avoid_embeddings, action], axis=-1)
        q_0 = self.head_net_0(obs_embeddings)
        q_1 = self.head_net_1(obs_embeddings)
        out = jp.concatenate([q_0, q_1], axis=-1)
        return out
    
def update_shared_parameters(target_params, source_params):
    target_params['params']['env_net'] = source_params['params']['env_net']
    target_params['params']['reach_avoid_net'] = source_params['params']['reach_avoid_net']
    return target_params
    
    
def create_deep_ltl_policy_net(
                    sequence_size: int,
                    num_assignments: int,
                    distr_param_size: int,
                    cfg) -> DeepLTLPolicyNet:
    
    env_net = MLP(layer_sizes=cfg.env_net.layer_sizes, 
                  activate_final=cfg.env_net.activate_final)
    
    assignment_net = MLP(layer_sizes=cfg.reach_avoid_net.assignment_net.layer_sizes, 
                         activate_final=cfg.reach_avoid_net.assignment_net.activate_final)
    
    set_net = MLP(layer_sizes=cfg.reach_avoid_net.set_net.layer_sizes, 
                  activate_final=cfg.reach_avoid_net.set_net.activate_final)
    
    reach_avoid_net = ReachAvoidNet(
                            assignment_embedding_size=cfg.reach_avoid_net.assignment_embedding_size,
                            sequence_size=sequence_size,
                            num_assignments=num_assignments,
                            assignment_net=assignment_net,
                            set_net=set_net,
                            aggr_mode=cfg.reach_avoid_net.aggr_mode,
                            hidden_size=cfg.reach_avoid_net.hidden_size)
    
    head_net = MLP(layer_sizes=list(cfg.head_net.layer_sizes) + [distr_param_size],
                   activate_final=cfg.head_net.activate_final)
    
    deep_ltl_policy_net = DeepLTLPolicyNet(env_net=env_net,
                                reach_avoid_net=reach_avoid_net,
                                head_net=head_net)
    
    return deep_ltl_policy_net

def create_deep_ltl_q_net(
                    sequence_size: int,
                    num_assignments: int,
                    cfg) -> DeepLTLQNet:
    
    env_net = MLP(layer_sizes=cfg.env_net.layer_sizes, 
                  activate_final=cfg.env_net.activate_final)
    
    assignment_net = MLP(layer_sizes=cfg.reach_avoid_net.assignment_net.layer_sizes, 
                         activate_final=cfg.reach_avoid_net.assignment_net.activate_final)
    
    set_net = MLP(layer_sizes=cfg.reach_avoid_net.set_net.layer_sizes, 
                  activate_final=cfg.reach_avoid_net.set_net.activate_final)
    
    reach_avoid_net = ReachAvoidNet(
                            assignment_embedding_size=cfg.reach_avoid_net.assignment_embedding_size,
                            sequence_size=sequence_size,
                            num_assignments=num_assignments,
                            assignment_net=assignment_net,
                            set_net=set_net,
                            aggr_mode=cfg.reach_avoid_net.aggr_mode,
                            hidden_size=cfg.reach_avoid_net.hidden_size)
    
    head_net_0 = MLP(layer_sizes=list(cfg.head_net.layer_sizes) + [1],
                   activate_final=cfg.head_net.activate_final)
    
    head_net_1 = MLP(layer_sizes=list(cfg.head_net.layer_sizes) + [1],
                   activate_final=cfg.head_net.activate_final)
    
    deep_ltl_q_net = DeepLTLQNet(env_net=env_net,
                                reach_avoid_net=reach_avoid_net,
                                head_net_0=head_net_0,
                                head_net_1=head_net_1)
    
    return deep_ltl_q_net
    

if __name__ == "__main__":
    from ..reach_avoid.util import flatten_reach_avoid, batch_trees
    from omegaconf import OmegaConf
    from pathlib import Path
    config_path = Path(__file__).parent / ".." / "config" / "deep_ltl" / "deep_ltl_net.yaml"
    cfg = OmegaConf.load(config_path)

    #create batched reach avoid sequences
    padding = 0
    eps = 1
    assignment_1 = 2
    assignment_2 = 3
    assignment_3 = 4

    set_1 = jp.array([assignment_1, assignment_2, padding, padding])
    set_2 = jp.array([assignment_2, assignment_3, eps, padding])
    set_3 = jp.array([assignment_1, assignment_2, eps, padding])
    padding_set = jp.array([padding, padding, padding, padding])

    seq_1 = jp.array([set_1, padding_set, padding_set])
    seq_2 = jp.array([set_1, set_2, padding_set])
    seq_3 = jp.array([set_1, set_2, set_3])
    seq = jp.array([padding_set, padding_set, padding_set])

    reach_1 = {'seq': seq_1, 'set_sizes': jp.array([2, 0, 0])}
    reach_2 = {'seq': seq_2, 'set_sizes': jp.array([2, 3, 0])}
    reach_3 = {'seq': seq_3, 'set_sizes': jp.array([2, 3, 3])}
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
    
    action = jp.array([[3, 1],
                       [1, 0],
                       [2, 1]])

    data = [env_obs, batched_reach_avoid_flat, action]
    #create module instance
    deep_ltl_net = create_deep_ltl_q_net(
                            sequence_size=3,
                            num_assignments=5,
                            cfg=cfg)
    
    #random number key
    rng = jax.random.PRNGKey(0)

    #initialize parameters
    init = jax.jit(deep_ltl_net.init)
    apply = jax.jit(deep_ltl_net.apply)
    params = init(rng, *data)
    #test forward pass
    out = apply(params, *data)
    print(out)

    #test backwards pass
    def test_loss_fn(params, env_obs, batched_reach_avoid_flat, action):
        output = deep_ltl_net.apply(params, env_obs, batched_reach_avoid_flat, action)
        return jp.mean(output[:,1]) # dummy scalar dependent on params
    
    grads = jax.grad(test_loss_fn)(params, *data)
    assert jp.all(grads['params']['head_net_0']['hidden_0']['bias'] == 0.0)
    assert jp.all(grads['params']['head_net_0']['hidden_0']['kernel'] == 0.0)
    assert jp.all(grads['params']['head_net_0']['hidden_1']['bias'] == 0.0)
    assert jp.all(grads['params']['head_net_0']['hidden_1']['kernel'] == 0.0)
    assert not jp.all(grads['params']['head_net_1']['hidden_0']['bias'] == 0.0)
    assert not jp.all(grads['params']['head_net_1']['hidden_0']['kernel'] == 0.0)
    assert not jp.all(grads['params']['head_net_1']['hidden_1']['bias'] == 0.0)
    assert not jp.all(grads['params']['head_net_1']['hidden_1']['kernel'] == 0.0)

    def all_zeros(pytree):
        leaves, _ = jax.tree_util.tree_flatten(pytree)
        return all(jp.all(leaf == 0) for leaf in leaves)
    
    assert not all_zeros(grads['params']['reach_avoid_net'])
    print("All Tests passed")
