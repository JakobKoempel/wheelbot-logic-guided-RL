from datetime import datetime
import functools

from wheelbot_sim_python.algorithms import classical_sac as sac
from wheelbot_sim_python.envs.wheelbot_classical_setpoint import WheelbotClassicalSetpoint
from brax.io import model
import jax
import jax.numpy as jp

import wandb
import hydra
import omegaconf
from omegaconf import OmegaConf


@hydra.main(config_path="config", config_name="classical_main", version_base=None)
def main(cfg: omegaconf.DictConfig):
    print('start')
    # initialize wandb logging
    wnb_cfg = OmegaConf.to_container(
        cfg,
        resolve=True,
    )

    wandb.init(
        # set the wandb project where this run will be logged
        project=cfg.wandb_project,
        # track hyperparameters and run metadata
        config=wnb_cfg,
        job_type="train",
        group=cfg.algorithm.name,
        name=cfg.experiment,
    )
    wandb.define_metric("agent_train/step")
    wandb.define_metric("agent_train/actor_loss", step_metric="agent_train/step")
    wandb.define_metric("agent_train/critic_loss", step_metric="agent_train/step")
    wandb.define_metric("agent_train/alpha_loss", step_metric="agent_train/step")
    wandb.define_metric("agent_eval/return_mean", step_metric="agent_train/step")
    wandb.define_metric("agent_eval/return_histogram", step_metric="agent_train/step")
    wandb.define_metric("agent_eval/best_return", step_metric="agent_train/step")

    # post evaluation logging callback
    def progress(num_steps, metrics):
        t = datetime.now()
        times.append(t)

        if "training/actor_loss" in metrics:
            data = {
                "agent_train/step": num_steps,
                "agent_train/actor_loss": metrics["training/actor_loss"],
                "agent_train/critic_loss": metrics["training/critic_loss"],
                "agent_train/alpha_loss": metrics["training/alpha_loss"],
                "agent_eval/return_mean": metrics["eval/episode_reward"],
                "agent_eval/return_histogram": wandb.Histogram(np_histogram=metrics["eval/episode_reward_histogram"]),
                "agent_eval/best_return": metrics["eval/best_return"],
            }
            wandb.log(data)

    # domain randomization callback
    def domain_randomize(sys, rng):
        """Randomizes the mjx.Model."""

        @jax.vmap
        def rand(rng):
            rng, key = jax.random.split(rng, 2)
            min_val = jp.array([0.8, 0.8, 1e-4, 1e-5, 1e-5])
            max_val = jp.array([1.2, 1.2, 1e-2, 1e-3, 1e-3])
            friction = jax.random.uniform(key, (5,), minval=min_val, maxval=max_val)
            friction = sys.pair_friction.at[:, :].set(friction)
            del key

            rng, key = jax.random.split(rng, 2)
            min_val = jp.array(
                [[1e-4, 1e-4, 2e-4], [8e-5, 3e-5, 3e-5], [8e-5, 3e-5, 3e-5]]
            )
            max_val = jp.array(
                [[3e-4, 3e-4, 5e-4], [1.5e-4, 7e-5, 7e-5], [1.5e-4, 7e-5, 7e-5]]
            )
            body_inertia = jax.random.uniform(
                key, (3, 3), minval=min_val, maxval=max_val
            )
            body_inertia = sys.body_inertia.at[2:, :].set(body_inertia)
            del key

            return friction, body_inertia

        friction, body_inertia = rand(rng)

        in_axes = jax.tree_util.tree_map(lambda x: None, sys)
        in_axes = in_axes.tree_replace(
            {
                "pair_friction": 0,
                "body_inertia": 0,
            }
        )

        sys = sys.tree_replace(
            {
                "pair_friction": friction,
                "body_inertia": body_inertia,
            }
        )

        return sys, in_axes


    # register the environnment
    train_cfg = cfg.algorithm
    env = WheelbotClassicalSetpoint(cfg=cfg.env)
    eval_env = WheelbotClassicalSetpoint(cfg=cfg.env, eval=True)

    # training function
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    train_fn = functools.partial(
        sac.train,
        num_timesteps=train_cfg.num_timesteps,
        episode_length=train_cfg.episode_length,
        num_envs=train_cfg.num_envs,
        num_eval_envs=train_cfg.num_eval_envs,
        learning_rate=train_cfg.learning_rate,
        discounting=train_cfg.discounting,
        batch_size=train_cfg.batch_size,
        num_evals=train_cfg.num_evals,
        normalize_observations=train_cfg.normalize_observations,
        max_replay_size=train_cfg.max_replay_size,
        deterministic_eval=train_cfg.deterministic_eval,
        grad_updates_per_step=train_cfg.grad_updates_per_step,
        hidden_layer_sizes=train_cfg.hidden_layer_sizes,
        checkpoint_logdir=output_dir,
        action_repeat=train_cfg.action_repeat,
        seed=cfg.seed,
        min_replay_size=train_cfg.min_replay_size,
        agent_dir=train_cfg.agent_dir,
        reward_scaling=train_cfg.reward_scaling,
    )

    times = [datetime.now()]

    _, params, _ = train_fn(
        environment=env,
        eval_env=eval_env,
        progress_fn=progress,
        randomization_fn=domain_randomize,
    )

    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")

    # save the model
    model_path = output_dir + "/final_mjx_brax_policy"
    model.save_params(model_path, params)
    wandb.finish()


if __name__ == "__main__":
    main()
