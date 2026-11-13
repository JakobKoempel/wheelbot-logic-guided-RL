from datetime import datetime
import functools

from brax import envs
from wheelbot_sim_python.algorithms import deep_ltl_sac
from wheelbot_sim_python.envs.wheelbot_zone_env import WheelbotZoneEnv
from wheelbot_sim_python.reach_avoid.zone_env_sampler import ZoneSampler
from wheelbot_sim_python.envs.wrappers.sequence_wrapper import SequenceWrapper
from brax.io import model
import jax
import jax.numpy as jp

import wandb
import hydra
import omegaconf
from omegaconf import OmegaConf


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(cfg: omegaconf.DictConfig):
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
        group=cfg.high_level_algorithm.name,
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

    sampler = ZoneSampler(sequence_size=2, sample_avoid=False)
    env = WheelbotZoneEnv(num_repeat=3, episode_length=650, fixed_zones=None, cfg=cfg)
    env = SequenceWrapper(env, sampler, eval=False, partial_rewards=False)
    eval_env = WheelbotZoneEnv(num_repeat=3, episode_length=650, fixed_zones=None, cfg=cfg)
    eval_env = SequenceWrapper(eval_env, sampler, eval=False)

    # training function
    # the high-level agent is trained with high_level_algorithm; cfg.algorithm only belongs to the fixed low level
    train_cfg = cfg.high_level_algorithm
    sac_net_config = cfg.deep_ltl
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    train_fn = functools.partial(
        deep_ltl_sac.train,
        num_timesteps=train_cfg.num_timesteps,
        episode_length=train_cfg.episode_length,
        sac_net_config = sac_net_config,
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
        checkpoint_logdir=output_dir,
        action_repeat=train_cfg.action_repeat,
        seed=cfg.seed,
        min_replay_size=train_cfg.min_replay_size,
        agent_dir=train_cfg.agent_dir,
        reward_scaling=train_cfg.reward_scaling,
        hindsight_experience_replay=train_cfg.her,
        share_params=train_cfg.share_params
    )

    times = [datetime.now()]

    _, params, _ = train_fn(
        environment=env,
        eval_env=eval_env,
        progress_fn=progress,
    )

    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")

    # save the model
    model_path = output_dir + "/final_policy"
    model.save_params(model_path, params)
    wandb.finish()


if __name__ == "__main__":
    main()