from jax import numpy as jp
import jax
import omegaconf
import hydra
from wheelbot_sim_python.envs.wheelbot_base import WHEELBOT_ROOT_PATH
from wheelbot_sim_python.envs.wheelbot_setpoint import WheelbotSetpoint


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(cfg: omegaconf.DictConfig) -> None:
    from omegaconf import dictconfig, OmegaConf
    import numpy as np
    from matplotlib.patches import Circle
    import matplotlib.pyplot as plt
    from wheelbot_sim_python.envs.wrappers.sequence_wrapper import SequenceWrapper
    from wheelbot_sim_python.reach_avoid.setpoint_sampler import SetpointSampler
    from wheelbot_sim_python.envs.wrappers.auto_reset_wrapper import AutoResetWrapper
    from wheelbot_sim_python.envs.wrappers.repeat_wrapper import RepeatWrapper
    from wheelbot_sim_python.envs.wrappers.deep_ltl_training import DoneStopWrapper
    from wheelbot_sim_python.envs.wrappers.plot_trajectory_wrapper import PlotTrajectoryWrapper
    from brax.io import model
    from wheelbot_sim_python.nets.deep_ltl_sac_net import make_deep_ltl_sac_networks
    from brax.training.agents.sac import networks as sac_networks
    from brax.training.acme import running_statistics
    import pickle
    from pathlib import Path
    import mediapy as media

    # ---------- settings ----------
    seed = 2
    force_resimulate = True  # True = ignore the cache and simulate again
    generate_video = True     # False = skip rendering and writing the video
    setpoint_x, setpoint_y = -0.6, 0.1  # cartesian setpoint, the robot starts at the origin

    # the env samples the setpoint in polar coordinates, so convert (min = max pins it)
    distance = float(np.hypot(setpoint_x, setpoint_y))
    theta = float(np.arctan2(setpoint_y, setpoint_x))

    camera = "origin_right"
    W, H = 1920, 960       # trail image resolution (max = offwidth/offheight in the XML)
    diff_threshold = 30    # pixel difference (0-255) that counts as "robot"
    min_alpha = 0.7        # opacity of the oldest ghost
    max_alpha = 1.0        # opacity of the newest ghost
    min_gap = 0.3         # approx. metres between shadows; larger = wider gaps
    bg_step = 10           # use every n-th frame for the median background

    # motion effect: faint copies of the robot just behind every ghost
    n_echo = 4             # number of faint copies behind each robot (0 = no effect)
    echo_gap = 0.045        # metres of driving between two copies (n_echo * echo_gap should stay below min_gap)
    echo_alpha = 0.35      # opacity of the copy closest to the robot (the others fade out)

    # cache file name depends on the settings, so changing them triggers a new simulation
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / f"rollout_r{distance}_th{theta}_seed{seed}.pkl"

    # register and instantiate environment (needed for rendering in any case)
    env_cfg = OmegaConf.load(
        (WHEELBOT_ROOT_PATH / "config/env/wheelbot.yaml").as_posix()
    )
    env_cfg['min_distance'] = distance
    env_cfg['max_distance'] = distance
    env_cfg['min_theta'] = theta
    env_cfg['max_theta'] = theta

    env = WheelbotSetpoint(cfg=env_cfg)
    sampler = SetpointSampler()
    env = SequenceWrapper(env, sampler, eval=True)
    env = RepeatWrapper(env, num_repeat=1, eval=True)
    env = DoneStopWrapper(env, stop_at_fail=True)

    # ---------- load from cache or simulate ----------
    if cache_path.exists() and not force_resimulate:
        print(f"Loading cached rollout from {cache_path}")
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        # back to jax arrays, in case env.render expects them
        rollout = jax.tree_util.tree_map(
            lambda x: jp.asarray(x) if isinstance(x, np.ndarray) else x,
            cached["rollout"],
        )
        setpoint = jp.asarray(cached["setpoint"])
        print('setpoint ', setpoint)
    else:
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
        jit_inference_fn = jax.jit(lambda obs, act_rng: jp.ravel(inference_fn(obs, act_rng)[0]))

        # define the jit reset/step functions
        jit_reset = jax.jit(env.reset)
        jit_step = jax.jit(env.step)

        # initialize the state
        rng = jax.random.PRNGKey(seed)
        state = jit_reset(rng)
        setpoint = state.info['setpoint']
        print('setpoint ', setpoint)
        rollout = [state.pipeline_state]

        for i in range(1500):
            print(i, end='\r')
            ctrl = jit_inference_fn(state.obs, rng)
            state = jit_step(state, ctrl)
            rollout.append(state.pipeline_state)

            if state.reward != 0:
                print("reward ", state.reward)
                print("reward : at ", i)

            if state.done:
                print('done at ', i)
                break

        # save to cache (jax arrays -> numpy, so pickling is safe)
        with open(cache_path, "wb") as f:
            pickle.dump(
                {
                    "rollout": jax.device_get(rollout),
                    "setpoint": np.asarray(setpoint),
                },
                f,
            )
        print(f"Saved rollout cache to {cache_path}")

    # ---------- trajectory plot ----------
    xy_traj = np.array([s.qpos[:2] for s in rollout])
    x = xy_traj[:, 0]
    y = xy_traj[:, 1]

    plt.figure(figsize=(6, 6))
    ax = plt.gca()

    plt.plot(x, y, color="orange", linewidth=2, label="Trajectory", zorder=1)
    plt.scatter(x[0], y[0], color="orange", s=80, label="Start", zorder=2)

    circle = Circle(
        (float(setpoint[0]), float(setpoint[1])),
        radius=0.25,
        edgecolor="grey",
        facecolor="grey",
        linewidth=2,
        zorder=4,
    )
    ax.add_patch(circle)

    ax.set_frame_on(True)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.axis('equal')
    ax.grid(True, color='lightgrey', linewidth=0.5)
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    plt.tight_layout()
    plt.savefig('trajectory_setpoint.png', bbox_inches='tight')
    plt.close()

    print('Trajectory plotted!')

    # ---------- video ----------
    if generate_video:
        frames = env.render(rollout, camera=camera, width=W, height=H)
        media.write_video("video_setpoint.mp4", frames, fps=1.0 / env.dt)
        print("Saved video_setpoint.mp4!")

    # ---------- Motion-trail image: robot "drives" through the picture ----------
    from scipy import ndimage

    # mask cleanup, in pixels
    open_iter, close_iter, dilate_iter = 2, 8, 2

    # --- choose ghost frames: equal distance along the path ---
    T = len(rollout)
    xy = np.array([s.qpos[:2] for s in rollout])
    step_dist = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    dist = np.concatenate([[0.0], np.cumsum(step_dist)])

    n_gaps = max(1, int(round(dist[-1] / min_gap)))
    targets = np.linspace(0.0, dist[-1], n_gaps + 1)   # includes start and final pose
    ghost_idx = np.unique(np.searchsorted(dist, targets))
    ghost_idx = np.clip(ghost_idx, 0, T - 1)

    alphas = np.linspace(min_alpha, max_alpha, len(ghost_idx))

    # --- choose the faint copies behind every ghost (equal driving distance between them) ---
    echoes = []   # one list per ghost: (frame index, weight), farthest behind first
    for idx in ghost_idx:
        found = {}
        for j in range(1, n_echo + 1):
            target = dist[idx] - j * echo_gap
            if target < 0:
                break                                  # the robot has not driven that far yet
            e = int(np.searchsorted(dist, target))
            if e >= idx or e in found:
                continue                               # same frame as the robot or as another copy
            found[e] = 1.0 - (j - 1) / max(n_echo, 1)  # closest copy: 1.0, farthest: 1/n_echo
        echoes.append(list(found.items())[::-1])
    echo_idx = sorted({e for lst in echoes for e, _ in lst})

    # --- render only the frames we need ---
    bg_idx = np.arange(0, T, bg_step)
    needed = np.unique(np.concatenate([ghost_idx, bg_idx, np.array(echo_idx, dtype=int)]))
    hi_frames = env.render(
        [rollout[i] for i in needed],
        camera=camera,
        width=W,
        height=H,
    )
    hi = {int(i): np.asarray(f) for i, f in zip(needed, hi_frames)}
    del hi_frames

    # background without the robot: per-pixel median over time
    background = np.median(
        np.stack([hi[int(i)] for i in bg_idx]), axis=0
    ).astype(np.float32)

    def robot_mask(frame):
        diff = np.abs(frame - background).max(axis=-1)
        mask = diff > diff_threshold
        mask = ndimage.binary_opening(mask, iterations=open_iter)
        mask = ndimage.binary_closing(mask, iterations=close_iter)
        return ndimage.binary_dilation(mask, iterations=dilate_iter)

    # --- composite: first all faint copies, then the robots on top of them ---
    composite = background.copy()

    for lst, a in zip(echoes, alphas):
        for e, w in lst:
            frame = hi[int(e)].astype(np.float32)
            alpha = (robot_mask(frame).astype(np.float32) * (echo_alpha * w * a))[..., None]
            composite = composite * (1 - alpha) + frame * alpha

    for idx, a in zip(ghost_idx, alphas):
        frame = hi[int(idx)].astype(np.float32)
        alpha = (robot_mask(frame).astype(np.float32) * a)[..., None]
        composite = composite * (1 - alpha) + frame * alpha

    composite = np.clip(composite, 0, 255).astype(np.uint8)

    fig = plt.figure(figsize=(W / 200, H / 200))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(composite)
    ax.axis("off")
    fig.savefig("setpoint_trail.pdf", dpi=200)
    plt.close(fig)
    print("Saved setpoint_trail.pdf!")


if __name__ == "__main__":
    main()
