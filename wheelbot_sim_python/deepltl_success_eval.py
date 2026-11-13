from jax import numpy as jp
import jax
import mediapy as media
from .envs.wheelbot_base import WHEELBOT_ROOT_PATH
from .envs.wheelbot_setpoint import WheelbotSetpoint
import hydra
import omegaconf


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(cfg: omegaconf.DictConfig) -> None:
    from omegaconf import dictconfig, OmegaConf
    import numpy as np
    from .envs.wrappers.sequence_wrapper import SequenceWrapper
    from .reach_avoid.setpoint_sampler import SetpointSampler
    from .envs.wrappers.repeat_wrapper import RepeatWrapper
    from .envs.wrappers.deep_ltl_training import DoneStopWrapper
    from brax.io import model
    from .nets.deep_ltl_sac_net import make_deep_ltl_sac_networks
    from brax.training.agents.sac import networks as sac_networks
    from brax.training.acme import running_statistics

    # ---------- settings ----------
    seed = 2
    num_envs = 1000

    # register and instantiate environment
    env_cfg = OmegaConf.load(
        (WHEELBOT_ROOT_PATH / "config/env/wheelbot.yaml").as_posix()
    )

    env = WheelbotSetpoint(cfg=env_cfg)
    sampler = SetpointSampler()
    env = SequenceWrapper(env, sampler, eval=True)
    env = RepeatWrapper(env, num_repeat=1, eval=True)
    env = DoneStopWrapper(env, stop_at_fail=True)

    # ---------- simulate ----------
    # load the model
    model_path = cfg.agent_dir
    params = model.load_params(model_path)[:2]
    normalize_fn = running_statistics.normalize
    sac_network = make_deep_ltl_sac_networks(
        observation_size=env.observation_size,
        env_observation_size=env.env_observation_size,
        action_size=env.action_size,
        sequence_size=env.sequence_size,
        num_assignments=env.context.num_assignments,
        cfg=cfg.deep_ltl,
        preprocess_observations_fn=normalize_fn
    )

    # make inference
    make_inference_fn = sac_networks.make_inference_fn(sac_network)
    inference_fn = make_inference_fn(params, deterministic=True)
    jit_inference_fn = jax.jit(lambda obs, act_rng: inference_fn(obs, act_rng)[0])

    # define the jit reset/step functions (batched over num_envs)
    jit_reset = jax.jit(jax.vmap(env.reset))
    jit_step = jax.jit(jax.vmap(env.step, in_axes=(0, 0)))

    # initialize the state
    rng = jax.random.PRNGKey(seed)
    keys = jax.random.split(rng, num_envs + 1)
    rng = keys[0]
    reset_rng = keys[1:]
    state = jit_reset(reset_rng)

    # DoneStopWrapper zeroes the reward once an env is done, so the sum is the terminal reward
    total_reward = jp.zeros(num_envs)
    # step at which each env first got reward 1 (-1 = never)
    steps_to_success = np.full(num_envs, -1, dtype=np.int64)

    for i in range(1500):
        print(i, end='\r')
        keys = jax.random.split(rng, num_envs + 1)
        rng = keys[0]
        act_rng = keys[1:]
        ctrl = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        total_reward = total_reward + state.reward
        newly_reached = (np.asarray(state.reward) == 1) & (steps_to_success < 0)
        steps_to_success[newly_reached] = i + 1

    success_rate = jp.mean(total_reward == 1)
    failure_rate = jp.mean(state.info['num_failures'] > 0)
    not_finished_rate = 1 - success_rate - failure_rate

    print(f"success rate:      {float(success_rate) * 100:.1f} %")
    print(f"failure rate:      {float(failure_rate) * 100:.1f} %")
    print(f"not finished rate: {float(not_finished_rate) * 100:.1f} %")

    reached_steps = steps_to_success[steps_to_success >= 0]
    if reached_steps.size > 0:
        print(f"avg steps to success (successful envs only, n={reached_steps.size}): {reached_steps.mean():.1f}")
    else:
        print("avg steps to success: no environment reached the setpoint")


if __name__ == "__main__":
    main()
