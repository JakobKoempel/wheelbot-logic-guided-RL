from jax import numpy as jp
import jax
from .sampler_base import ReachAvoidSampler
from ..envs.wheelbot_zone_env import ZoneEnvContext
from .util import ReachAvoidSeq

class ZoneSampler(ReachAvoidSampler):

    def __init__(self, sequence_size: int = 1, sample_avoid: bool = False):
        self.sequence_size = sequence_size
        self.max_set_size = ZoneEnvContext.num_assignments
        self.sample_avoid = sample_avoid

    @staticmethod
    def overlap(set1, set2):
            set1 = jp.where(set1 == 0, -1, set1)
            return jp.any(jp.isin(set1, set2))

    def sample_reach_avoid_sets(self, rng: jp.ndarray):
        assignments = jp.arange(3, self.max_set_size) # all assignment indices except the first which represents agent is in no zone
        
        num_rng, perm_rng, reach_rng = jax.random.split(rng, 3)
        perm_assignments = jax.random.permutation(perm_rng, assignments)

        num_used_assignments = jax.random.choice(num_rng, 5, shape=(), p=jp.array([0., 0.1, 0.1, 0.4, 0.4])) # - 2 to exclude padding and eps index
        reach_set_size = jax.random.randint(reach_rng, shape=(), minval=1, maxval=num_used_assignments + 1)
        
        reach_set_mask = jp.arange(1, self.max_set_size - 2) <= reach_set_size
        reach_set = perm_assignments * reach_set_mask
        reach_set = jp.pad(reach_set, (0, self.max_set_size - reach_set.size))

        avoid_set_size = num_used_assignments - reach_set_size
        idx = jp.arange(1, self.max_set_size - 2)
        avoid_set_mask = (idx > reach_set_size) & (idx <= reach_set_size + avoid_set_size)
        avoid_set = perm_assignments * avoid_set_mask
        avoid_set = jp.roll(avoid_set, -reach_set_size)
        avoid_set = jp.pad(avoid_set, (0, self.max_set_size - avoid_set.size))

        return reach_set, avoid_set
    
    def is_proper(self, reach_seq: jp.ndarray, avoid_seq: jp.ndarray) -> bool:
        """
        Checks whether the following conditions and returns true if all are false:
        1. reach sets at step i and i + 1 overlap
        2. reach sets at step i and avoid set at step i overlap
        3. reach sets at step i and avoid set at step i + 1 overlap
        """
        def body(i, has_overlap):
            reach = reach_seq[i]
            avoid = avoid_seq[i]
            return jp.logical_or(has_overlap, self.overlap(reach, avoid))

        has_overlap = jax.lax.fori_loop(0, self.sequence_size, body, False)

        def body(i, has_overlap):
            reach = reach_seq[i]
            avoid = avoid_seq[i + 1]
            return jp.logical_or(has_overlap, self.overlap(reach, avoid))

        has_overlap = jax.lax.fori_loop(0, self.sequence_size - 1, body, has_overlap)

        def body(i, has_overlap):
            current_reach = reach_seq[i]
            nxt_reach = reach_seq[i + 1]
            return jp.logical_or(has_overlap, self.overlap(current_reach, nxt_reach))

        has_overlap = jax.lax.fori_loop(0, self.sequence_size - 1, body, has_overlap)

        return jp.logical_not(has_overlap)
    
    
    def sample(self, rng: jp.ndarray, length: int) -> ReachAvoidSeq:
        length = jp.min(jp.array([self.sequence_size, length]))

        def sample_once(rng):
            """Generate a single reach_seq (used inside while_loop)."""
            reach_seq = jp.zeros((self.sequence_size, self.max_set_size), dtype=jp.int32)
            avoid_seq = jp.zeros_like(reach_seq)

            def sample_sets(i, val):
                rng, reach_seq, avoid_seq = val
                rng, reach_set_rng = jax.random.split(rng)
                reach_set, avoid_set = self.sample_reach_avoid_sets(reach_set_rng)

                reach_seq = reach_seq.at[i].set(reach_set)
                if self.sample_avoid:
                    avoid_seq = avoid_seq.at[i].set(avoid_set)
                else:
                    avoid_seq = avoid_seq.at[i].set(jp.zeros_like(avoid_set))
                return rng, reach_seq, avoid_seq

            rng, reach_seq, avoid_seq = jax.lax.fori_loop(0, length, sample_sets, (rng, reach_seq, avoid_seq))

            return rng, reach_seq, avoid_seq

        def cond_fun(carry):
            _, reach_seq, avoid_seq = carry
            return jp.logical_not(self.is_proper(reach_seq, avoid_seq))

        def body_fun(carry):
            rng, _, _ = carry
            rng, reach_seq, avoid_seq = sample_once(rng)
            return rng, reach_seq, avoid_seq

        rng, reach_seq, avoid_seq = sample_once(rng)
        init_carry = (rng, reach_seq, avoid_seq)
        _, reach_seq, avoid_seq = jax.lax.while_loop(cond_fun, body_fun, init_carry)

        reach_set_sizes = jp.sum(reach_seq != 0, axis=1)
        avoid_set_sizes = jp.sum(avoid_seq != 0, axis=1)
        return {
            "reach": {"seq": reach_seq, "set_sizes": reach_set_sizes},
            "avoid": {"seq": avoid_seq, "set_sizes": avoid_set_sizes},
            "length": jp.array([length]),
        }
    
    def sample_training(self, obs: jp.ndarray, rng: jp.ndarray) -> ReachAvoidSeq:
        length_rng, rng = jax.random.split(rng)
        length = jax.random.randint(length_rng, shape=(), minval=1, maxval=self.sequence_size + 1)
        return self.sample(rng, length)
    
    def sample_eval(self, obs: jp.ndarray, rng: jp.ndarray):
        length_rng, rng = jax.random.split(rng)
        length = self.sequence_size
        return self.sample(rng, length)

  
if __name__ == "__main__":
    n = 200
    seq_size = 1
    sampler = ZoneSampler(seq_size, sample_avoid=False)
    sample = jax.jit(jax.vmap(lambda rng: sampler.sample_eval(None, rng)))
    rng = jax.random.PRNGKey(100)
    rngs = jax.random.split(rng, n)
    reach_avoid = sample(rngs)

    #for i in range(n):
        #print(jax.tree.map(lambda x: x[i], reach_avoid))

    print('reach set sizes 0 ', jp.sum(reach_avoid['reach']['set_sizes'] == 0) / (n * seq_size) )
    print('reach set sizes 1 ', jp.sum(reach_avoid['reach']['set_sizes'] == 1) / (n * seq_size) )
    print('reach set sizes 2 ', jp.sum(reach_avoid['reach']['set_sizes'] == 2) / (n * seq_size) )
    print('reach set sizes 3 ', jp.sum(reach_avoid['reach']['set_sizes'] == 3) / (n * seq_size) )
    print('reach set sizes 4 ', jp.sum(reach_avoid['reach']['set_sizes'] == 4) / (n * seq_size) )

    print('avoid set sizes 0 ', jp.sum(reach_avoid['avoid']['set_sizes'] == 0) / (n * seq_size) )
    print('avoid set sizes 1 ', jp.sum(reach_avoid['avoid']['set_sizes'] == 1) / (n * seq_size) )
    print('avoid set sizes 2 ', jp.sum(reach_avoid['avoid']['set_sizes'] == 2) / (n * seq_size) )
    print('avoid set sizes 3 ', jp.sum(reach_avoid['avoid']['set_sizes'] == 3) / (n * seq_size) )
    print('avoid set sizes 4 ', jp.sum(reach_avoid['avoid']['set_sizes'] == 4) / (n * seq_size) )

    print('Finished')