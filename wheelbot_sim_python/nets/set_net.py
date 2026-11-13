import jax
import jax.numpy as jp
from flax import linen as nn
from typing import Literal
from brax.training.networks import MLP
from flax.linen.linear import default_embed_init


class AssignmentSetNet(nn.Module):
    phi_net: MLP
    rho_net: MLP
    aggr_mode: Literal['sum', 'max', 'mean'] = 'sum'

    @nn.compact
    def __call__(self, data: jp.ndarray, mask: jp.ndarray) -> jp.ndarray:
        """
        data: [batch_size, set_size, feature_size]
        mask: [batch_size, set_size] #zeros out features that represent assignment padding
        """

        embedding = self.phi_net(data) * mask[:, :, None]
        if self.aggr_mode == 'sum':
            aggregate = jp.sum(embedding, axis=1)
        elif self.aggr_mode == 'mean':
            aggregate = jp.mean(embedding, axis=1)
        elif self.aggr_mode == 'max':
            aggregate = jp.max(embedding, axis=1, initial=0)

        # if all assignments in a set are padding, replace aggregate with an empty embedding
        is_empty = jp.all(aggregate == 0, axis=1)
        embed_dim = embedding.shape[-1]
        empty_embedding = self.param('empty_embedding', default_embed_init, (1, embed_dim))

        aggregate = jp.where(is_empty[:, None], empty_embedding, aggregate)

        out = self.rho_net(aggregate)
        return out
    


if __name__ == "__main__":
    def test_loss_fn(params, inputs, mask):
        output = model.apply(params, inputs, mask)
        return jp.mean(output[3]) - 2  # dummy scalar dependent on params

    #dummy input: shape (batch_size, set_size, feature_dim)
    dummy_input = jp.array([
        [[1., 2., 3., 1],
         [1., 2., 4., 1],
         [1., 1., 1., 2]],

        [[1., 2., 3., 1],
         [1., 5., 4., 2],
         [-1., -1., -1., -1.]],

        [[1., 5., 4., 2],
          [1., 2., 3., 1],
          [-5., -5., -5., -5.]],
          
        [[1., 2., 3., 1],
         [1., 2., 4., 1],
         [1., 1., 1., 2]],

        [[1., 2., 4., 1],
         [1., 1., 1., 2],
         [1., 2., 3., 1]],

        [[1., 2., 4., 1],
         [1., 1., 1., 2],
         [1., 2., 3., 1]]
    ])

    mask = jp.array([
        [1, 1, 1],
        [1, 1, 0],
        [1, 1, 0],
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0]
    ])

    #create module instance
    phi_net = MLP(layer_sizes=[], activate_final=True)
    rho_net = MLP(layer_sizes=[8, 8], activate_final=False)
    model = AssignmentSetNet(phi_net=phi_net, rho_net=rho_net, aggr_mode="sum")

    #random number key
    rng = jax.random.PRNGKey(0)

    #initialize parameters
    init = jax.jit(model.init)
    apply = jax.jit(model.apply)
    params = init(rng, dummy_input, mask)
   
    #test forward pass
    output = apply(params, dummy_input, mask)
    print(output)
    #test permutation invariance
    assert jp.allclose(output[1], output[2])
    assert jp.allclose(output[0], output[4])
    assert jp.allclose(output[3], output[5])

    #test backwards pass
    grads = jax.grad(test_loss_fn)(params, dummy_input, mask)
    
    print('All tests passed')