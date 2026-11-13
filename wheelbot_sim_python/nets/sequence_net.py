from flax.linen.recurrent import GRUCell
import flax.linen as nn
import jax
from jax import numpy as jp

class SequenceNet(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, data: jp.ndarray, padding_masks: jp.ndarray) -> jp.ndarray:
        """
        data: [batch_size, seq_size, feature_size]
        padding_mask: [batch_size, seq_size]
        """
        
        class GRUCellMasked(GRUCell):
            @nn.compact
            def __call__(self, carry: jp.ndarray, inputs: jp.ndarray, padding_mask: jp.ndarray) -> jp.ndarray:
                """
                inputs: [batch_size, feature_size]
                carry: [batch_size, hidden_size]
                padding_mask: [batch_size, ]
                """
                padding_mask = padding_mask[:, None] #[batch_size, 1] for broadcasting
                hidden, hidden = super().__call__(carry, inputs)
                hidden = hidden * padding_mask
                return hidden, hidden
            
        init_carry = jp.zeros((data.shape[0], self.hidden_size))
        gru = nn.scan(
            GRUCellMasked,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1
        )(features=self.hidden_size, name="gru_cell_masked")
        

        final_hidden, outputs = gru(init_carry, data, padding_masks)
        return final_hidden, outputs
    


if __name__ == "__main__":
    def test_loss_fn(params, input, padding_masks):
        final_hidden, outputs = model.apply(params, input, padding_masks)
        return jp.mean(final_hidden)  

    hidden_size = 3
    padding_masks = jp.array([
                    [0, 0, 1, 1],
                    [0, 1, 1, 0],
                    [1, 1, 0, 0]])
    
    dummy_input = jp.array([[
                [1,0,0,-1,0],
                [1,-3,0,0,0],
                [3,1,0,5,7],
                [1,0,5,2,5]],

                [[1,0,-1,0,0],
                [3,1,0,5,7],
                [1,0,5,2,5],
                [1,0,0,-1,0]],
                
                [[3,1,0,5,7],
                [1,0,5,2,5],
                [1,0,0,0,-1],
                [1,0,0,-2,0]]
                ])
    
    model = SequenceNet(hidden_size)

    #random number key
    rng = jax.random.PRNGKey(0)

    #initialize parameters
    init = jax.jit(model.init)
    apply = jax.jit(model.apply)
    params = init(rng, dummy_input, padding_masks)
   
    #test forward pass
    final_hidden, outputs = apply(params, dummy_input, padding_masks)
    print(outputs)
    print(final_hidden)
    assert jp.allclose(outputs[0, 2], outputs[1, 1]) and jp.allclose(outputs[0, 2], outputs[2, 0])
    assert jp.allclose(outputs[0, 3], outputs[1, 2]) and jp.allclose(outputs[0, 3], outputs[2, 1])
    assert jp.all(outputs[0, 0] == 0)
    assert jp.all(outputs[0, 1] == 0)
    assert jp.all(outputs[1, 0] == 0)
    assert jp.all(outputs[1, 3] == 0)
    assert jp.all(outputs[2, 2] == 0)
    assert jp.all(outputs[2, 3] == 0)
    
    #test backwards pass
    grads = jax.grad(test_loss_fn)(params, dummy_input, padding_masks)
    print(grads)
    print("All tests passed")