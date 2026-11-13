from brax.envs.base import State, Wrapper
import jax
from jax import numpy as jp

class AutoResetWrapper(Wrapper):
    """
    Reimplementation of the brax.envs.wrappers.training.AutoResetWrapper
    The default brax AutoResetWrapper cannot reset fields in the info dictionary
    """

    def __init__(self, env):
        self.reset_keys = getattr(env, "reset_keys", set())
        super().__init__(env)

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        for key in self.reset_keys:
            state.info['first_' + key] = state.info[key]
        state.info["first_pipeline_state"] = state.pipeline_state
        state.info["first_obs"] = state.obs

        return state

    def step(self, state: State, action: jax.Array) -> State:
        if "steps" in state.info:
            steps = state.info["steps"]
            steps = jp.where(state.done, jp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(state, action)

        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
            return jp.where(done, x, y)

        pipeline_state = jax.tree.map(where_done, state.info["first_pipeline_state"], state.pipeline_state)
        obs = jax.tree.map(where_done, state.info["first_obs"], state.obs)
    
        for key in self.reset_keys:
            state.info[key] =  jax.tree.map(where_done, state.info['first_' + key], state.info[key])
        return state.replace(pipeline_state=pipeline_state, obs=obs,)
    
import hydra
import omegaconf
@hydra.main(config_path="../../config", config_name="main", version_base=None)
def main(cfg: omegaconf.DictConfig) -> None:
    from ..wheelbot_setpoint import WheelbotSetpoint
    from ...reach_avoid.samplers import SetpointEvalSampler
    from .sequence_wrapper import SequenceWrapper
        
    env = WheelbotSetpoint(cfg=cfg.env)
    sampler = SetpointEvalSampler()
    env = SequenceWrapper(env, sampler)
    env = AutoResetWrapper(env)

    # define the jit reset/step functions
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # initialize the state
    rng = jax.random.PRNGKey(4)
    state = jit_reset(rng)
    rollout = [state.pipeline_state]

    for i in range(700):
        ctrl = jp.array([-0.6, 0.0]) 
        state = jit_step(state, ctrl)
        rollout.append(state.pipeline_state)
        if state.reward != 0:
            print('reward', i, state.reward)

if __name__ == "__main__":
    main()