from jax import numpy as jp
import jax
from jax import jit
from ..reach_avoid.setpoint_sampler import SetpointSampler
from .wheelbot_setpoint import WheelbotSetpoint
from .wrappers.sequence_wrapper import SequenceWrapper
from brax.envs.wrappers.training import EpisodeWrapper
from .wrappers.repeat_wrapper import RepeatWrapper
from brax.envs.base import PipelineEnv, State
from brax.training.types import Transition
from brax.io import model
from ..nets.deep_ltl_sac_net import make_deep_ltl_sac_networks
from brax.training.agents.sac import networks as sac_networks
from brax.training.acme import running_statistics
from omegaconf import dictconfig
from ..automata.context import Context
import matplotlib.pyplot as plt
from typing import Optional
from matplotlib.patches import FancyBboxPatch

colors = [(0.9059, 0.1176, 0.3137, 0.9), (0.0980, 0.4000, 0.1020, 0.9), (0.1255, 0.9490, 0.9451, 1.0), (0.9569, 0.8824, 0.1098, 1.0)]
ap_labels = ['red', 'green', 'blue', 'yellow']
possible_assignments = jp.concatenate([ jp.zeros((1,4), dtype=bool), jp.eye(4, dtype=bool)])
ZoneEnvContext = Context(ap_labels, possible_assignments) 

class WheelbotZoneEnv(PipelineEnv):
    def __init__(
        self,
        num_repeat: int,
        episode_length: int,
        raw_actions: bool = True,
        jitable_step = True,
        fixed_zones: Optional[jp.ndarray] = None,
        cfg: dictconfig.DictConfig = dictconfig.DictConfig({}),
        **kwargs,
    ):
        self.min_setpoint_distance = cfg.env.min_distance
        self.max_setpoint_distance = cfg.env.max_distance
        self.min_setpoint_theta = cfg.env.min_theta
        self.max_setpoint_theta = cfg.env.max_theta
        self.setpoint_radius = cfg.env.setpoint_radius
        self.zone_radius = 0.25
        self.num_colors = 4
        self.zones_per_color = 2
        self.zone_spawn_radius = 1
        self.min_zone_distance = 0
        self.raw_actions = raw_actions
        self.jitable_step = jitable_step
        self.fixed_zones = fixed_zones
        self.rollout = []
        self.setpoints = []
        self.context = ZoneEnvContext
        self.reset_keys = {'low_level_state', 'active_assignment'}
        # state.info fields that are stored in the transition samples and are necessary for HER
        self.extra_fields = ('robot_xy', 'robot_yaw', 'centers')
        # create environment
        env = WheelbotSetpoint(cfg=cfg.env, zones=self.fixed_zones, zone_colors=colors, zone_radius=self.zone_radius)
        sampler = SetpointSampler()
        env = SequenceWrapper(env, sampler, deploy=True)
        env = RepeatWrapper(env, num_repeat=num_repeat, deploy=True)
        env = EpisodeWrapper(env, episode_length=episode_length, action_repeat=1)
        self.env = env
        # load the model
        model_path = cfg.low_level_agent_dir
        params = model.load_params(model_path)[:2]
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

        #make inference
        make_inference_fn = sac_networks.make_inference_fn(sac_network)
        inference_fn = make_inference_fn(params, deterministic=True)
        self.inference_fn = jit(lambda obs: jp.ravel(inference_fn(obs, None)[0])) # removed extras
        self.ll_step = jit(env.step)
        self.ll_reset = jit(env.reset)

    @property
    def action_size(self) -> int:
        return 2
    
    @property
    def zone_observation_size(self) -> int:
        return self.num_colors * self.zones_per_color * 3   
    
    @property
    def observation_size(self) -> int:
        return self.zone_observation_size
    
  
    def plot_rollout(self, draw_setpoints: bool = False):
        """
        Plots Trajectory and Setpoints. Only works when self.jitable_step == False
        """
        assert not self.jitable_step

        # Extract (x, y) from each pipeline_state
        xy = [ps.qpos[:2] for ps in self.rollout]
        xy = jp.stack(xy)
        x, y = jp.array(xy[:, 0]), jp.array(xy[:, 1])

        plt.figure(figsize=(6, 6))
        plt.plot(x, y, label='Trajectory', linewidth=2, color='orange')
        plt.scatter(x[0], y[0], color='orange', label='Start', s=80)

        if draw_setpoints:
            for sp in self.setpoints:
                plt.scatter(sp[0], sp[1], color='grey', s=30)
                circle = plt.Circle(sp, self.setpoint_radius, color='grey', alpha=0.3)
                plt.gca().add_patch(circle)

        for i in range(self.num_colors):
            color = colors[i]
            for j in range(self.zones_per_color):
                color_xy = self.centers[i, j]
                circle = plt.Circle(color_xy, self.setpoint_radius, color=color)
                plt.gca().add_patch(circle)

        ax = plt.gca()
        ax.set_frame_on(False)    
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        plt.axis('equal')
        plt.grid(True)
        plt.savefig('trajectory_zone_env', bbox_inches='tight')
        print('Trajectory plotted!')


    def _sample_zones(self, rng: jp.ndarray) -> tuple[jp.ndarray, jp.ndarray]:
        """
        Sample non-overlapping zone centers and assign colors.
        Returns (centers:(Z,2), colors:(Z,))
        """
        total_zones = self.num_colors * self.zones_per_color

        centers0 = jp.zeros((total_zones, 2), dtype=jp.float32)

        def cond_fun(carry):
            rng, centers, count = carry
            return count < total_zones

        def body_fun(carry):
            rng, centers, count = carry
            rng, r_rng, theta_rng = jax.random.split(rng, 3)
            proposal_theta = jax.random.uniform(theta_rng, shape=(), minval=-jp.pi, maxval=jp.pi)
            min_distance_to_origin = jp.where(jp.logical_and(proposal_theta < jp.pi/2, proposal_theta > -jp.pi/2), 0.35, self.zone_radius + 0.1)
            proposal_radius = jax.random.uniform(r_rng, shape=(), minval=min_distance_to_origin, maxval=self.zone_spawn_radius)
            
            proposal = proposal_radius * jp.array([jp.cos(proposal_theta), jp.sin(proposal_theta)])
            idxs = jp.arange(total_zones)
            mask = idxs < count
            dists = jp.linalg.norm(centers - proposal, axis=1)
            cond_ok = dists >= self.min_zone_distance + 2 * self.zone_radius
            all_ok = jp.all(jp.logical_or(~mask, cond_ok))
            centers = jax.lax.cond(
                all_ok,
                lambda c: c.at[count].set(proposal),
                lambda c: c,
                centers,
            )
            count = count + jp.where(all_ok, 1, 0)
            return rng, centers, count

        rng, centers, count = jax.lax.while_loop(
            cond_fun, body_fun, (rng, centers0, 0)
        )

        centers = jax.random.permutation(rng, centers)
        return jp.reshape(centers, shape=(self.num_colors, self.zones_per_color, 2))
    
    def get_rel_polar_coordinates(self, robot_xy: jp.ndarray, robot_yaw: float, center: jp.ndarray) -> float:
        difference = center - robot_xy
        distance = jp.linalg.norm(difference)

        target_angle = jp.arctan2(difference[1], difference[0])
        yaw_error = robot_yaw - target_angle
        yaw_error = (yaw_error + jp.pi) % (2 * jp.pi) - jp.pi

        return jp.array([distance, jp.sin(yaw_error), jp.cos(yaw_error)])
    
    def update_zone_obs(self, state: State):
        robot_xy = state.info['robot_xy']
        robot_yaw = state.info['robot_yaw']
        robot_obs = state.info['low_level_state'].obs
        centers_polar = jax.vmap(
                                jax.vmap(
                                    lambda c: self.get_rel_polar_coordinates(robot_xy, robot_yaw, c),
                                    in_axes=0
                                ),
                                in_axes=0
                                )(state.info['centers'])
        red_polar, green_polar, blue_polar, yellow_polar = centers_polar
        red_dist = red_polar[:, 0]
        green_dist = green_polar[:, 0]
        blue_dist = blue_polar[:, 0]
        yellow_dist = yellow_polar[:, 0]
        red = jp.any(red_dist <= self.zone_radius)
        green = jp.any(green_dist <= self.zone_radius)
        blue = jp.any(blue_dist <= self.zone_radius)
        yellow = jp.any(yellow_dist <= self.zone_radius)
        
        assignment = jp.array([red, green, blue, yellow])
        assignment_index = ZoneEnvContext.assignment_to_index(assignment)
        state.info['active_assignment'] = assignment_index
        obs = state.obs.at[:self.zone_observation_size].set(centers_polar.reshape(-1))
        return state.replace(obs=obs[:self.observation_size])
    
    def reset(self, rng: jp.ndarray) -> State:
        zone_rng, rng = jax.random.split(rng)
        ll_state = self.ll_reset(rng)
        reward, done = jp.zeros(2)
        metrics = {}
        centers = self._sample_zones(zone_rng)
        
        if self.fixed_zones is not None:
            centers = self.fixed_zones

        robot_xy = ll_state.pipeline_state.qpos[:2]
        robot_yaw = ll_state.obs[0]
        info = {'robot_xy': robot_xy, 'robot_yaw': robot_yaw, 'centers' : centers, 'low_level_state' : ll_state, 'setpoint': jp.zeros(2)} 
        state = State(None, jp.zeros(self.observation_size), reward, done, metrics, info)
        return self.update_zone_obs(state)
        
    def step(self, state: State, action: jp.ndarray) -> State:
        #scale raw actions from (-1,1) to polar coordinates
        if self.raw_actions:
            r_interval = self.max_setpoint_distance - self.min_setpoint_distance
            r = (action[0] + 1) * r_interval * 0.5 + self.min_setpoint_distance
            theta_interval = self.max_setpoint_theta - self.min_setpoint_theta
            theta = (action[1] + 1) * theta_interval * 0.5 + self.min_setpoint_theta
        else:
            r, theta = action
        r = jp.clip(r, self.min_setpoint_distance, self.max_setpoint_distance)
        theta = jp.clip(theta, self.min_setpoint_theta, self.max_setpoint_theta)
        
        ll_state = state.info['low_level_state']
        ll_state.info['steps'] = 0. # reset EpisodeWrapper step counter. 
        ll_state.info['truncation'] = 0. # reset EpisodeWrapper truncation flag
        ll_state.info['num_successes'] = 0. #reset RepeatWrapper num_successes
        robots_yaw = ll_state.obs[0]
        theta = (theta + robots_yaw + jp.pi) % (2 * jp.pi) - jp.pi
        setpoint = jp.array([r * jp.cos(theta), r * jp.sin(theta)]) + ll_state.pipeline_state.qpos[:2]
        state.info['setpoint'] = setpoint
        state.info['low_level_state'] = self.env.update_setpoint_state(ll_state, action=jp.zeros(2), setpoint=setpoint)

        # Let the low-level agent run until the setpoint is reached, episode is truncated or a zone is entered
        def cond_fun(carry):
            state, prev_assignment = carry
            ll_state = state.info['low_level_state']
            assignment = state.info['active_assignment']
            assignment_change = prev_assignment != assignment
            in_zone = assignment > 2 # 0 padding, 1 eps, 2 not in any zone
            
            not_entered_zone = jp.logical_not(jp.logical_and(assignment_change, in_zone))
            not_truncated = jp.logical_not(ll_state.info['truncation'])
            not_success = ll_state.info['num_successes'] < 1

            return jp.logical_and(not_entered_zone, jp.logical_and(not_truncated, not_success))
   
        def body_fun(carry):
            state, prev_assignment = carry
            prev_assignment = state.info['active_assignment']
            ll_state = state.info['low_level_state']
            action = self.inference_fn(ll_state.obs)
            ll_state = self.ll_step(ll_state, action)
            state.info['low_level_state'] = ll_state
            state.info['robot_xy'] = ll_state.pipeline_state.qpos[:2]
            state.info['robot_yaw'] = ll_state.obs[0]

            return self.update_zone_obs(state), prev_assignment
        
        if self.jitable_step:
            state, _ = jax.lax.while_loop(cond_fun, body_fun, (state, state.info['active_assignment']))
        else:
            prev_assignment = state.info['active_assignment']
            self.setpoints.append(setpoint)
            self.centers = state.info['centers']
            while cond_fun((state, prev_assignment)):
                self.rollout.append(state.info['low_level_state'].pipeline_state)
                state, prev_assignment = body_fun((state, prev_assignment))
             
        return state
    
    def her_applicable(self, last_transition: Transition) -> bool:
        done = 1. - last_transition.discount
        ongoing = last_transition.reward == 0. # cant use reach avoid state to check whether episode is ongoing because the AutoReseWrapper resets it
        robot_xy = last_transition.extras['state_extras']['robot_xy']
        centers = last_transition.extras['state_extras']['centers']
        dist_to_centers = jp.linalg.norm(centers - robot_xy, axis=1)
        not_in_zone = jp.all(dist_to_centers > self.zone_radius)
        ongoing_and_not_in_zone = jp.logical_and(ongoing, not_in_zone)
        return jp.logical_and(done == 1.0, ongoing_and_not_in_zone)

    def create_her_episode(self, episode: Transition, episode_length: int) -> Transition:
        # shift a zone of the reach set such that the agent reaches it
        centers = episode.extras['state_extras']['centers'][episode_length - 1]
        robot_xy = episode.extras['state_extras']['robot_xy'][episode_length - 1]
        robot_yaw = episode.extras['state_extras']['robot_yaw'][episode_length - 1]
        reach_set = episode.extras['state_extras']['reach_avoid_state'].reach_avoid['reach']['seq'][episode_length - 1][0]
        target_zone = reach_set[0] - 3 # assignment index is offset by 3: padding, eps and not in a zone
        new_zone_center = robot_xy + (self.zone_radius - 0.001) * jp.array([jp.cos(robot_yaw), jp.sin(robot_yaw)])
        her_centers = centers.at[target_zone].set(new_zone_center)

        #reconstruct first her state
        first_done = 0.0
        first_reward = 0.0
        first_extras = jax.tree.map(lambda x: x[0], episode.extras['state_extras'])
        first_info = {'robot_xy': jp.array([0., 0.]), 'robot_yaw': 0.,
                    'centers' : her_centers, 'active_assignment': 2, 
                    'reach_avoid_state': first_extras['reach_avoid_state']}
        first_obs = episode.observation[0]
        first_state = State(None, first_obs, first_reward, first_done, {}, first_info)
        first_state = self.update_zone_obs(first_state)
        first_state = SequenceWrapper.update_sequence_state(first_state)

        # recompute her episode states
        def f(state, xs):
            robot_xy, robot_yaw = xs
            state.info['robot_xy'] = robot_xy
            state.info['robot_yaw'] = robot_yaw
            state = self.update_zone_obs(state)
            state = SequenceWrapper.update_sequence_state(state)
            
            return state, state 
    
        _, next_states = jax.lax.scan(f, first_state, xs=(episode.extras['state_extras']['robot_xy'], episode.extras['state_extras']['robot_yaw']))

        # Set truncation & done flag if her episode doesn't end in success or fail
        her_truncation = jp.zeros_like(episode.extras['state_extras']['truncation'])
        ongoing = jp.where(jp.sum(next_states.done) == 0, 1.0, 0.0)
        her_truncation = her_truncation.at[episode_length - 1].set(ongoing)
        her_done = next_states.done.at[episode_length - 1].set(1. - ongoing)
    
        # build her episode struct
        her_episode = Transition(
            observation=jp.concatenate([first_state.obs[None, :], next_states.obs[:-1]], axis=0),
            action=episode.action,
            reward=next_states.reward,
            discount=1. - her_done,
            next_observation=next_states.obs,
            extras=episode.extras
        )
        batch_dim = episode.extras['state_extras']['centers'].shape[0]
        her_episode.extras['state_extras']['centers'] = jp.broadcast_to(her_centers, (batch_dim, *her_centers.shape))
        her_episode.extras['state_extras']['reach_avoid_state'] = next_states.info['reach_avoid_state']
        her_episode.extras['state_extras']['eval_status'] = next_states.info['eval_status']
        her_episode.extras['state_extras']['truncation'] = her_truncation
        return her_episode
