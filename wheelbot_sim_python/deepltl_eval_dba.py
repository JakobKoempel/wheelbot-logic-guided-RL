
from .nets.deep_ltl_sac_net import make_deep_ltl_sac_networks, make_value_inference_fn
from brax.training.agents.sac import networks as sac_networks
from brax.training.acme import running_statistics
from brax import envs
from brax.io import model
import omegaconf
import hydra
import jax
import jax.numpy as jp
from jax import jit, vmap
import pandas as pd
from .automata.formulae import turn_either
from .envs.wheelbot_setpoint import SetpointContext
from .envs.wheelbot_setpoint import WheelbotSetpoint
from .envs.wrappers.done_wrapper import DoneStopWrapper
from .envs.wrappers.dba_wrapper import DBAWrapper
from brax.envs.wrappers.training import EpisodeWrapper
from .envs.wrappers.plot_trajectory_wrapper import PlotTrajectoryWrapper
from .envs.wrappers.repeat_wrapper import RepeatWrapper


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(cfg: omegaconf.DictConfig) -> None:
    num_envs = 1000
    n_steps = 1500
    #create environment
    dba = SetpointContext.formula_to_dba(turn_either, seq_size=4)
    env = WheelbotSetpoint(cfg=cfg.env)
    env = DBAWrapper(env, dba)
    env = RepeatWrapper(env, num_repeat=1, eval=True)
    env = DoneStopWrapper(env)
    env = EpisodeWrapper(env, episode_length=n_steps, action_repeat=1)
    env = PlotTrajectoryWrapper(env, episode_length=n_steps)

    # load the model
    model_path = cfg.agent_dir
    norm_params, pol_params, q_params = model.load_params(model_path)
    
    normalize_fn = running_statistics.normalize
    sac_network = make_deep_ltl_sac_networks(
                    observation_size=env.observation_size,
                    env_observation_size=env.env_observation_size,
                    action_size=env.action_size,
                    sequence_size=env.sequence_size,
                    num_assignments=env.context.num_assignments,
                    cfg = cfg.deep_ltl,
                    preprocess_observations_fn=normalize_fn
    )

    #make value inference function and add it to the DBA Wrapper
    make_value_inf_fn = make_value_inference_fn(sac_network)
    value_inference_fn = make_value_inf_fn(norm_params, pol_params, q_params)
    env.set_value_inference_fn(value_inference_fn)

    #make inference
    make_inference_fn = sac_networks.make_inference_fn(sac_network)
    inference_fn = make_inference_fn((norm_params, pol_params), deterministic=True)
    jit_inference_fn = jax.jit(lambda obs, act_rng: inference_fn(obs, act_rng)[0]) # removed extras

    jit_reset = jit(vmap(env.reset))
    jit_step = jit(vmap(env.step, in_axes=(0, 0)))

    # initialize the state
    rng = jax.random.PRNGKey(100)
    keys = jax.random.split(rng, num_envs + 1)
    rng = keys[0]
    reset_rng = keys[1:]
    state = jit_reset(reset_rng)
    
    for i in range(n_steps):
        keys = jax.random.split(rng, num_envs + 1)
        rng = keys[0]
        act_rng = keys[1:]
        ctrl = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)

    print("percentage of envs that need a sequence resample after a failure : ", jp.count_nonzero(state.info['num_failures']) / num_envs)

    df = pd.DataFrame({
    "Setpoints Reached": state.info['num_successes'],
    "Formula Violations": state.info['num_failures']
    })

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    print(df)

    prob_success = df["Setpoints Reached"].sum() / num_envs
    print(f"Pob successes: {prob_success}")

if __name__ == "__main__":
    main()