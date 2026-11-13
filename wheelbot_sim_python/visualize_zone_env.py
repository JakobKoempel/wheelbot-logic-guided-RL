from jax import numpy as jp
import jax
import hydra
import omegaconf
from brax.io import model
from brax.training.agents.sac import networks as sac_networks
from .nets.deep_ltl_sac_net import make_deep_ltl_sac_networks
from .envs.wheelbot_zone_env import WheelbotZoneEnv
from .envs.wrappers.dba_wrapper_zone_env import DBAWrapper


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(cfg: omegaconf.DictConfig) -> None:
    import hashlib
    import pickle
    from pathlib import Path
    import numpy as np
    import mediapy as media

    zones = jp.array([[[0.7900920510292053, -0.3798096776008606], [-0.07152072340250015, -0.9711779952049255]], [[0.24166765809059143, 0.9303094148635864], [-0.8283154368400574, 0.09091756492853165]], [[0.5022664070129395, 0.33988624811172485], [-0.13928696513175964, 0.33106136322021484]], [[-0.10068108141422272, -0.37453851103782654], [-0.5928919315338135, -0.5963943004608154]]])
    
    # ---------- settings ----------
    formula = "G F blue"
    n_steps = 7
    force_resimulate = True  # True = ignore the cache and simulate again
    generate_video = True     # False = skip rendering and writing the video

    camera = "origin_right_far"
    W, H = 1920, 1440      # trail image resolution (max = offwidth/offheight in the XML)
    diff_threshold = 30    # pixel difference (0-255) that counts as "robot"
    min_alpha = 0.7        # opacity of the oldest ghost
    max_alpha = 1.0        # opacity of the newest ghost
    min_gap = 0.3          # approx. metres between shadows; larger = wider gaps
    bg_step = 10           # use every n-th frame for the median background

    # motion effect: faint copies of the robot just behind every ghost
    n_echo = 4             # number of faint copies behind each robot (0 = no effect)
    echo_gap = 0.045        # metres of driving between two copies (n_echo * echo_gap should stay below min_gap)
    echo_alpha = 0.35      # opacity of the copy closest to the robot (the others fade out)

    # cache file name depends on the settings, so changing them triggers a new simulation
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    settings = f"{formula}|{n_steps}|{cfg.agent_dir}|{cfg.low_level_agent_dir}|{np.asarray(zones).tolist()}"
    cache_path = cache_dir / f"zone_rollout_{hashlib.md5(settings.encode()).hexdigest()[:12]}.pkl"

    # the plain zone env is needed for plotting and rendering in any case
    zone_env = WheelbotZoneEnv(num_repeat=3, episode_length=650, raw_actions=True, jitable_step=False, fixed_zones=zones, cfg=cfg)

    # ---------- load from cache or simulate ----------
    if cache_path.exists() and not force_resimulate:
        print(f"Loading cached rollout from {cache_path}")
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        # back to jax arrays, in case env.render expects them
        zone_env.rollout = jax.tree_util.tree_map(
            lambda x: jp.asarray(x) if isinstance(x, np.ndarray) else x,
            cached["rollout"],
        )
        zone_env.setpoints = [jp.asarray(sp) for sp in cached["setpoints"]]
        zone_env.centers = jp.asarray(cached["centers"])
        print(cached["centers"])
        status = cached["status"]
    else:
        dba = zone_env.context.formula_to_dba(formula, seq_size=2, strict_avoid=False, simplify_labels=True, draw=True)
        env = DBAWrapper(zone_env, dba)
        #env = EpisodeWrapper(env, episode_length=1)
        #env = AutoResetWrapper(env)

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
        jit_inference_fn = jax.jit(lambda obs, act_rng: jp.ravel(inference_fn(obs, act_rng)[0])) # removed extras

        jit_reset = (env.reset)
        jit_step = (env.step)

        rng = jax.random.PRNGKey(7)
        reset_rng, act_rng = jax.random.split(rng)
        state = jit_reset(reset_rng)

        print(state.info['centers'])

        for i in range(n_steps):
            print(i, end='\r')
            act = jit_inference_fn(state.obs, act_rng)
            state = jit_step(state, act)
            if state.done:
                break

        status = state.info['eval_status']

        # save to cache (jax arrays -> numpy, so pickling is safe)
        with open(cache_path, "wb") as f:
            pickle.dump(
                {
                    "rollout": jax.device_get(zone_env.rollout),
                    "setpoints": [np.asarray(sp) for sp in zone_env.setpoints],
                    "centers": np.asarray(zone_env.centers),
                    "status": np.asarray(status),
                },
                f,
            )
        print(f"Saved rollout cache to {cache_path}")

    print('status ', status)
    zone_env.plot_rollout(draw_setpoints=False)
    if generate_video:
        media.write_video("video_zone_env.mp4", zone_env.env.render(zone_env.rollout, camera="origin_right_far", width=W, height=H), fps=1.0 / zone_env.env.dt)

    # ---------- Motion-trail image: robot "drives" through the picture ----------
    import matplotlib.pyplot as plt
    from scipy import ndimage

    render_env = zone_env.env
    rollout = zone_env.rollout

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
    hi_frames = render_env.render(
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
    fig.savefig("zone_env_trail.pdf", dpi=200)
    plt.close(fig)
    print("Saved zone_env_trail.pdf!")
    
if __name__ == "__main__":
    main()
