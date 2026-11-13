import jax
import jax.numpy as jp
from flax import linen as nn
from typing import Literal
from dataclasses import field
from ..reach_avoid.util import unflatten_reach_avoid, reverse_reach_avoid
from .set_net import AssignmentSetNet
from .sequence_net import SequenceNet
from brax.training.networks import MLP


class ReachAvoidNet(nn.Module):
    sequence_size: int # max length of the sequences.
    num_assignments: int # includes the padding and eps assignment
    assignment_embedding_size: int
    assignment_net: MLP
    set_net: MLP
    aggr_mode: Literal['sum', 'max', 'mean']
    hidden_size: int
    

    def create_set_mask(self, set_sizes: jp.ndarray, max_set_size: int):
        range_arr = jp.arange(max_set_size)[None, None, :]
        result = range_arr < set_sizes[..., None]
        return result.astype(jp.float32).reshape(-1, max_set_size)

    @nn.compact
    def __call__(self, batched_reach_avoid_flat: jp.ndarray) -> jp.ndarray:
        """
        batched_reach_avoid_flat: [batch_size, flat_size] 
        """
        #unflatten the reach avoid sequences
        unflatten_batch = jax.vmap(unflatten_reach_avoid, in_axes=(0, None))
        batched_reach_avoid = unflatten_batch(batched_reach_avoid_flat, self.sequence_size) #[batch_size, sequence_size, set_size]

        #reverse the reach avoid sequences
        reverse_batch = jax.vmap(reverse_reach_avoid)
        batched_reach_avoid = reverse_batch(batched_reach_avoid)

        #create assignment embeddings
        assignment_embedding_layer = nn.Embed(self.num_assignments, self.assignment_embedding_size, name='assignment_embedding')
        reach_assignments_embedding = assignment_embedding_layer(batched_reach_avoid['reach']['seq']) #[batch_size, sequence_size, set_size, assignment_feature_size]
        avoid_assignments_embedding = assignment_embedding_layer(batched_reach_avoid['avoid']['seq'])

        #create set embeddings
        batch_size, _, set_size, _ = reach_assignments_embedding.shape
        set_feature_size = self.set_net.layer_sizes[-1]
        reach_assignments_embedding = reach_assignments_embedding.reshape(-1, set_size, self.assignment_embedding_size) #[batch_size * sequence_size, set_size, assignment_feature_size]
        avoid_assignments_embedding = avoid_assignments_embedding.reshape(-1, set_size, self.assignment_embedding_size)
        
        set_layer = AssignmentSetNet(phi_net=self.assignment_net,
                                    rho_net=self.set_net,
                                    aggr_mode=self.aggr_mode,
                                    name='assignment_set_net')
        
        reach_set_sizes = batched_reach_avoid['reach']['set_sizes']
        reach_mask = self.create_set_mask(reach_set_sizes, set_size)
        reach_sets_embeddings = set_layer(reach_assignments_embedding, reach_mask).reshape(batch_size, self.sequence_size, set_feature_size)
        avoid_set_sizes = batched_reach_avoid['avoid']['set_sizes']
        avoid_mask = self.create_set_mask(avoid_set_sizes, set_size)
        avoid_sets_embeddings = set_layer(avoid_assignments_embedding, avoid_mask).reshape(batch_size, self.sequence_size, set_feature_size)

        #concatentate reach and avoid set embeddings
        sets_embeddings = jp.concatenate([reach_sets_embeddings, avoid_sets_embeddings], axis=-1) #[batch_size, sequence_size, 2 * set_feature_size]
        
        #create sequence embeddings
        sequence_layer = SequenceNet(hidden_size=self.hidden_size, name='sequence_net')
        seq_lengths = batched_reach_avoid['length']
        range_row = jp.arange(self.sequence_size)
        sequence_mask = (range_row >= (self.sequence_size - seq_lengths)).astype(jp.int32) # Note: Unlike an assignment mask, a sequence mask has the form [0, ..., 0, 1, ..., 1] because the sequences were reversed
        final_hidden, outputs = sequence_layer(sets_embeddings, sequence_mask)
        return final_hidden, outputs #return outputs for testing 



if __name__ == "__main__":
    from ..reach_avoid.util import flatten_reach_avoid, _pytree_allclose, batch_trees
    def test_loss_fn(params, input):
        seq_embeddings, _ = model.apply(params, input)
        return jp.mean(seq_embeddings)  
    
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

    #create module instance
    assignment_net = MLP(layer_sizes=[16], activate_final=True)
    set_net = MLP(layer_sizes=[16, 16], activate_final=False)
    model = ReachAvoidNet(assignment_embedding_size=16,
                            sequence_size=3,
                            num_assignments=5,
                            assignment_net=assignment_net,
                            set_net=set_net,
                            aggr_mode='sum',
                            hidden_size=16)

    #random number key
    rng = jax.random.PRNGKey(0)

    #initialize parameters
    init = jax.jit(model.init)
    apply = jax.jit(model.apply)
    params = init(rng, batched_reach_avoid_flat)
    #test forward pass
    final_hidden, outputs = apply(params, batched_reach_avoid_flat)
    assert jp.allclose(outputs[0][2], outputs[1][1]) and jp.allclose(outputs[0][2], outputs[2][0])

    #test backwards pass
    grads = jax.grad(test_loss_fn)(params, batched_reach_avoid_flat)

    assert jp.all(grads['params']['assignment_embedding']['embedding'][0] == 0.0)
    assert jp.all(grads['params']['assignment_embedding']['embedding'][1] == 0.0)
    assert jp.all(grads['params']['assignment_embedding']['embedding'][2] != 0.0)
    assert jp.all(grads['params']['assignment_embedding']['embedding'][3] != 0.0)
    assert jp.all(grads['params']['assignment_embedding']['embedding'][4] == 0.0)
    
    print("All tests passed")