from .nets.deep_ltl_sac_net import make_deep_ltl_sac_networks
from brax.training.agents.sac import networks as sac_networks
from brax import envs
from brax.io import model
import omegaconf
import hydra
import jax
import jax.numpy as jp
from jax import jit, vmap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from .reach_avoid.zone_env_sampler import ZoneSampler
from .envs.wrappers.done_wrapper import DoneCountWrapper
from .envs.wrappers.sequence_wrapper import SequenceWrapper
from .envs.wheelbot_zone_env import WheelbotZoneEnv



@hydra.main(config_path="config", config_name="main", version_base=None)
def main(cfg: omegaconf.DictConfig) -> None:
    #create environment
    num_envs = 20
    n_steps = 5
    env = WheelbotZoneEnv(num_repeat=3, episode_length=650, raw_actions=True, fixed_zones=None, cfg=cfg)
    sampler = ZoneSampler(sequence_size=1, sample_avoid=False)
    env = SequenceWrapper(env, sampler, eval=True)
    env = DoneCountWrapper(env)
    
    # load the model
    model_path = cfg.agent_dir
    params = model.load_params(model_path)[:2]
    normalize_fn = lambda x, y: x #no observation normalization
    sac_network = make_deep_ltl_sac_networks(
                    observation_size=env.observation_size,
                    env_observation_size=env.env_observation_size,
                    action_size=env.action_size,
                    sequence_size=env.sequence_size,
                    num_assignments=env.context.num_assignments,
                    cfg = cfg.deep_ltl,
                    preprocess_observations_fn=normalize_fn
    )

    #make inference
    make_inference_fn = sac_networks.make_inference_fn(sac_network)
    inference_fn = make_inference_fn(params, deterministic=True)
    jit_inference_fn = jax.jit(lambda obs, act_rng: inference_fn(obs, act_rng)[0]) # removed extras

    # delete old plots
    output_dir = "zone_eval_plots"
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        if os.path.isfile(file_path):  # delete only files
            os.remove(file_path)

    jit_reset = jit(vmap(env.reset))
    jit_step = jit(vmap(env.step, in_axes=(0, 0)))

    # initialize the state
    rng = jax.random.PRNGKey(2)
    keys = jax.random.split(rng, num_envs + 1)
    rng = keys[0]
    reset_rng = keys[1:]
    state = jit_reset(reset_rng)
    
    setpoints = []
    positions = []
    reach_sets_1 = state.info['reach_avoid_state'].reach_avoid['reach']['seq'][: , 0]
    reach_sizes_1 = state.info['reach_avoid_state'].reach_avoid['reach']['set_sizes'][: , 0]
    reach_sets_2 = state.info['reach_avoid_state'].reach_avoid['reach']['seq'][: , 1]
    reach_sizes_2 = state.info['reach_avoid_state'].reach_avoid['reach']['set_sizes'][: , 1]
    avoid_sets_1 = state.info['reach_avoid_state'].reach_avoid['avoid']['seq'][: , 0]
    avoid_sizes_1 = state.info['reach_avoid_state'].reach_avoid['avoid']['set_sizes'][: , 0]
    avoid_sets_2 = state.info['reach_avoid_state'].reach_avoid['avoid']['seq'][: , 1]
    avoid_sizes_2 = state.info['reach_avoid_state'].reach_avoid['avoid']['set_sizes'][: , 1]
    
    for i in range(num_envs):
        print(i)
        print("reach 1",env.context.assignment_indices_to_frozen_assignment_set(reach_sets_1[i], reach_sizes_1[i]))
        print("avoid 1",env.context.assignment_indices_to_frozen_assignment_set(avoid_sets_1[i], avoid_sizes_1[i]))
        print("reach 2",env.context.assignment_indices_to_frozen_assignment_set(reach_sets_2[i], reach_sizes_2[i]))
        print("avoid 2",env.context.assignment_indices_to_frozen_assignment_set(avoid_sets_2[i], avoid_sizes_2[i]))

    for i in range(n_steps):
        keys = jax.random.split(rng, num_envs + 1)
        rng = keys[0]
        act_rng = keys[1:]
        ctrl = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        setpoints.append(state.info['setpoint'])
        positions.append(state.info['robot_xy'])

    setpoints = jp.array(setpoints)
    positions = jp.array(positions)
    success_mask = np.array(state.info['reach_avoid_state'].status == 1)
    ongoing_mask = np.array(state.info['reach_avoid_state'].status == 0)
    fail_mask = np.array(state.info['reach_avoid_state'].status == -1)

    done_at = state.metrics['done_at']
    print('done_at step : ', done_at)
    steps_performed = jp.where(done_at != -1, done_at + 1, n_steps)
    success_prob = jp.sum(success_mask) / num_envs
    fail_prob = jp.sum(fail_mask) / num_envs
    print('success probability', success_prob)
    print('failure probability', fail_prob)

    # Convert JAX arrays to NumPy for plotting
    zones = np.array(state.info['centers'])

    # Loop over each environment in the batch
    for i in range(num_envs):

        # Determine outcome type
        if success_mask[i]:
            result_type = "success"
        elif ongoing_mask[i]:
            result_type = "ongoing"
        else:
            result_type = "failure"

        # Create plot
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_aspect('equal', adjustable='box')
        ap_labels = ['red', 'green', 'blue', 'yellow']
        for j in range(env.num_colors):
            color = ap_labels[j]
            for k in range(env.zones_per_color):
                color_xy = zones[i, j, k]
                circle = plt.Circle(color_xy, radius=env.zone_radius, color=color, alpha=0.2, zorder=1)
                ax.add_artist(circle)

        
        # Plot setpoints
        for j in range(steps_performed[i]):
            setpoint = setpoints[j, i]
            position = positions[j, i]
            ax.scatter(setpoint[0], setpoint[1], color='grey', label='setpoint' if j == 0 else "", s=80, zorder=3)
            circle = plt.Circle(setpoint, radius=0.25, color='grey', alpha=0.2, zorder=1)
            ax.add_artist(circle)
            # Label the point with its step number (starting from 1)
            ax.text(setpoint[0], setpoint[1], str(j + 1), fontsize=9, color='black', ha='center', va='center', zorder=4)

            # Plot agent position
            ax.scatter(position[0], position[1], color='blue', label='Agent' if j == 0 else "", s=50, zorder=4)
            # Label the point with its step number (starting from 1)
            ax.text(position[0], position[1], str(j + 1), fontsize=9, color='black', ha='center', va='center', zorder=4)

        # Fixed frame centered on origin
        frame_range = 2.0
        ax.set_xlim(-frame_range, frame_range)
        ax.set_ylim(-frame_range, frame_range)

        # Grid + legend
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Env {i}: {result_type}")

        # Save figure
        filename = os.path.join(output_dir, f"env_{i}_{result_type}.png")
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {num_envs} evaluation plots to {output_dir}/")

if __name__ == '__main__':
    main()