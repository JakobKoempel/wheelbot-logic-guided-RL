from jax import numpy as jp
import numpy as np
from etils import epath
import jax
from brax.envs.base import State
from omegaconf import dictconfig
import xml.etree.ElementTree as ET
from brax import envs
import mediapy as media
from .wheelbot_base import WheelbotBase, WHEELBOT_ROOT_PATH, qpos_qvel_to_robot_state
from .util.visualization import compute_circle_element
from ..automata.context import Context
import itertools
from typing import Optional
import hydra
import omegaconf

def yaw_to_setpoint(robot_xy: jp.ndarray, robot_yaw: float, setpoint: jp.ndarray) -> float:
    dx, dy = setpoint - robot_xy
    target_angle = jp.arctan2(dy, dx)
    yaw_error = robot_yaw - target_angle
    yaw_error = (yaw_error + jp.pi) % (2 * jp.pi) - jp.pi

    return yaw_error

def distance_to_setpoint(robot_xy: jp.ndarray, setpoint: jp.ndarray) -> float:
    difference = setpoint - robot_xy
    return jp.linalg.norm(difference)

ap_labels=['d', 'rv', 'lr', 'rr', 'a', 's']
possible_assignments = jp.array([assignment for assignment in itertools.product([False, True], repeat=len(ap_labels))
                        if assignment[2] != assignment[3] or not assignment[2]]) # cannot roll left and right at the same time

SetpointContext = Context(ap_labels, possible_assignments) 

class WheelbotSetpoint(WheelbotBase):
    def __init__(
        self,
        zones: Optional[jp.ndarray] = None,
        zone_colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1), (1, 1, 0, 1)],
        zone_radius: float = 0.25,
        setpoint_markers: Optional[jp.ndarray] = None,
        cfg: dictconfig.DictConfig = dictconfig.DictConfig({}),
        **kwargs,
    ):
   
        mjcf_path = (WHEELBOT_ROOT_PATH / "mjcf").as_posix()
        xml_path = mjcf_path + "/wheelbot_alpha.xml"
        xml_str = epath.Path(xml_path).read_text()
        root = ET.fromstring(xml_str)

        # set the mesh file paths
        asset = root.find("asset")
        for mesh in asset.findall("mesh"):
            if mesh.attrib.get("name") == "body" or mesh.attrib.get("name") == "wheel":
                relative_file = mesh.get("file")
                mesh.set("file", mjcf_path + "/" + relative_file)

        track_body = root.find(".//body[@name='track']")
        if zones is not None:
            assert zones.shape[0] == 4, "Zones are not divided into 4 colors"
            assert len(zone_colors) == 4, "Number of colors is not 4"

            for color_group, rgba in zip(zones, zone_colors):
                for zone in color_group:
                    elem = compute_circle_element(xy=(zone[0], zone[1]), radius=zone_radius, rgba=rgba)
                    track_body.append(elem)

        # grey ground markers (rendering only) at the given setpoints
        if setpoint_markers is not None:
            for sp in setpoint_markers:
                elem = compute_circle_element(xy=(float(sp[0]), float(sp[1])), radius=0.2, rgba=(0.4, 0.4, 0.4, 1), z=0.0002)
                track_body.append(elem)

        xml_str = ET.tostring(root, encoding="unicode")
        super().__init__(xml_str, cfg, **kwargs)
        self.context = SetpointContext
        self.min_theta = jp.clip(cfg.get("min_theta", -jp.pi), -jp.pi, jp.pi)
        self.max_theta = jp.clip(cfg.get("max_theta", jp.pi), -jp.pi, jp.pi)
        self.min_distance = cfg.get("min_distance", 0.5)
        self.max_distance = cfg.get("max_distance", 1.5)
        self.speed_threshold = cfg.get("speed_threshold", 7)
        self.roll_threshold = cfg.get("roll_threshold", 0.07)
        self.setpoint_radius = cfg.get("setpoint_radius", 0.25)
        self.robot_mask = jp.array([1, 1, 1, 1, 1, 1, 0, 1, 0, 1])
        self.yaw_to_setpoint_grad_fn = jax.grad(yaw_to_setpoint, argnums=(0,1))
        self.distance_to_setpoint_grad_fn = jax.grad(distance_to_setpoint, argnums=(0,))
        self.reset_keys = {'setpoint'}
        self.extra_fields = tuple()

        assert self.min_theta <= self.max_theta
        assert self.min_distance <= self.max_distance
        assert self.min_distance > 0
        assert self.roll_threshold > 0
        assert self.speed_threshold >= 0
        assert self.setpoint_radius > 0


    def is_driving(self, obs: jp.ndarray, action: jp.ndarray) -> bool:
        driving_torque = action[0]
        speed = obs[7]
        pitch = obs[2]
        #neg driving torque => drive forward
        driving = jp.logical_and(jp.logical_or(driving_torque < 0. , speed > self.speed_threshold), pitch > -0.02)
        not_crashed = jp.logical_not((jp.abs(obs[:10]) > self.max_state).any())
        return jp.logical_and(driving, not_crashed)
    
    def positive_roll_vel(self, obs: jp.ndarray) -> bool:
        roll_vel = obs[4]
        return roll_vel > 0.
    
    def right_roll(self, obs: jp.ndarray) -> bool:
        roll = obs[1]
        yaw_vel = obs[3]
        return jp.logical_and(roll > self.roll_threshold, yaw_vel < 3.)
    
    def left_roll(self, obs: jp.ndarray) -> bool:
        roll = obs[1]
        yaw_vel = obs[3]
        return jp.logical_and(roll < -self.roll_threshold, yaw_vel > -3.)

    def reached_yaw(self, obs: jp.ndarray) -> bool:
        setpoint_angle = obs[12]
        setpoint_distance = obs[10]
        aligned_angle = jp.max(jp.array([jp.arctan(self.setpoint_radius / setpoint_distance), 0.15]))
        return jp.isclose(setpoint_angle, 0.0, atol=aligned_angle)

    def reached_setpoint(self, obs: jp.ndarray) -> bool:
        setpoint_dist = obs[10]
        return jp.isclose(setpoint_dist, 0.0, atol=self.setpoint_radius)
    
  
    def get_active_assignment(self, obs: jp.ndarray, action: jp.ndarray) -> int:
        """Returns active assignment by its index"""
        is_driving = self.is_driving(obs, action)
        positive_roll_vel = self.positive_roll_vel(obs)
        left_roll = self.left_roll(obs)
        right_roll = self.right_roll(obs)
        reached_yaw = self.reached_yaw(obs)
        reached_setpoint = self.reached_setpoint(obs)
        assignment = jp.array([is_driving, positive_roll_vel, left_roll, right_roll, reached_yaw, reached_setpoint])
        return SetpointContext.assignment_to_index(assignment)
    
    def get_setpoint_yaw_and_vel(self, robot_xy, robot_xy_vel, robot_yaw, robot_yaw_vel, setpoint):
        #Compute setpoint yaw and velocity using automatic differentiation and the chain rule.
        setpoint_yaw = yaw_to_setpoint(robot_xy, robot_yaw, setpoint)
        vels  = jp.array([robot_xy_vel[0], robot_xy_vel[1], robot_yaw_vel])
        grads_robot_xy, grad_robot_yaw = self.yaw_to_setpoint_grad_fn(robot_xy, robot_yaw, setpoint)
        setpoint_yaw_grads = jp.stack([*grads_robot_xy, grad_robot_yaw])
        setpoint_yaw_rate = jp.dot(setpoint_yaw_grads, vels)

        return setpoint_yaw, setpoint_yaw_rate 
    
    def get_setpoint_distance_and_vel(self, robot_xy, robot_xy_vel, setpoint):
        #Compute setpoint distance and velocity using automatic differentiation and the chain rule.
        setpoint_distance = distance_to_setpoint(robot_xy, setpoint)
        setpoint_distance_grads = jp.array(self.distance_to_setpoint_grad_fn(robot_xy, setpoint))
        setpoint_vel = jp.squeeze(jp.dot(setpoint_distance_grads, robot_xy_vel))

        return setpoint_distance, setpoint_vel 
    
    def reset(self, rng: jp.ndarray) -> State:
        return self._reset(rng, init_robot_state=jp.zeros(10), init_robot_xy=jp.zeros(2))
    
    def repeat(self, state: State, rng: jp.ndarray) -> State:
        robot_state = state.obs[:10]
        current_robot_xy = state.pipeline_state.qpos[:2]
        return self._reset(rng, init_robot_state=robot_state, init_robot_xy=current_robot_xy)
    
    def update_setpoint_state(self, state: State, action: jp.ndarray, setpoint: jp.ndarray) -> State:
        robot_state = state.obs[:10]
        robot_xy = state.pipeline_state.qpos[:2]
        robot_xy_vel = state.pipeline_state.qvel[:2]
        robot_yaw = robot_state[0]
        robot_yaw_vel = robot_state[3]
        robot_state_masked = robot_state * self.robot_mask
        setpoint_distance, setpoint_vel = self.get_setpoint_distance_and_vel(robot_xy=robot_xy, 
                                                                                    robot_xy_vel=robot_xy_vel,
                                                                                    setpoint=setpoint)
        setpoint_yaw, setpoint_yaw_vel = self.get_setpoint_yaw_and_vel(robot_xy=robot_xy,
                                                                        robot_xy_vel=robot_xy_vel,
                                                                        robot_yaw=robot_yaw, 
                                                                        robot_yaw_vel=robot_yaw_vel,
                                                                        setpoint=setpoint)
        obs = jp.array([*robot_state_masked, setpoint_distance, setpoint_vel, setpoint_yaw, setpoint_yaw_vel])
        
        # ensure state.obs has at least length of obs
        obs_len = obs.shape[0]
        state_obs_len = state.obs.shape[0]
        if state_obs_len < obs_len:
            pad_width = obs_len - state_obs_len
            state_obs = jp.concatenate([state.obs, jp.zeros(pad_width)])
        else:
            state_obs = state.obs

        updated_obs = state_obs.at[:obs_len].set(obs)
        state = state.replace(obs=updated_obs)
        state.info['setpoint'] = setpoint
        state.info['active_assignment'] = self.get_active_assignment(obs, action)
        return state

    def _reset(self, rng: jp.ndarray, init_robot_state: jp.ndarray, init_robot_xy: jp.ndarray) -> State:     
        radius_rng, theta_rng = jax.random.split(rng)
        r = jax.random.uniform(radius_rng, shape=(), minval=self.min_distance, maxval=self.max_distance)
        theta = jax.random.uniform(theta_rng, shape=(), minval=self.min_theta, maxval=self.max_theta)
        setpoint = jp.array([r * jp.cos(theta), r * jp.sin(theta)]) + init_robot_xy 
        
        #set the robot to the initial robot state and create the initial observation
        state = super().reset_at(init_xy=init_robot_xy, init_robot_state=init_robot_state)
        state = state.replace(obs=init_robot_state)
        return self.update_setpoint_state(state, action=jp.zeros(2), setpoint=setpoint)
    
    def step(self, state: State, action: jp.ndarray) -> State:
        #perform one step using the base class. The shape of the base observations is (10,) and only include the robot's state
        robot_state = state.obs[:10] * self.robot_mask
        base_state = super().step(state.replace(obs=robot_state), action)
        
        #get new state
        next_robot_state = qpos_qvel_to_robot_state(base_state.pipeline_state.qpos, base_state.pipeline_state.qvel) 
        setpoint = state.info['setpoint']
        state = state.replace(pipeline_state=base_state.pipeline_state, obs=next_robot_state)
        return self.update_setpoint_state(state, action, setpoint)


class WheelbotFixedSetpoints(WheelbotSetpoint):
    """Setpoint env that walks through a fixed array of setpoints in absolute world coordinates.
    The first setpoint is used on reset, every repeat advances to the next one.
    Once the array is exhausted the last setpoint is kept and info['all_setpoints_reached'] is set to 1,
    which the RepeatWrapper turns into done in deploy mode.
    With draw_setpoints=True, grey circles are added to the scene at the setpoints."""
    def __init__(self, setpoints: jp.ndarray, draw_setpoints: bool = False, **kwargs):
        if draw_setpoints:
            kwargs['setpoint_markers'] = np.asarray(setpoints)
        super().__init__(**kwargs)
        setpoints = jp.asarray(setpoints, dtype=jp.float32)
        assert setpoints.ndim == 2 and setpoints.shape[1] == 2, "setpoints must have shape (N, 2)"
        self.setpoints = setpoints
        self.reset_keys = self.reset_keys | {'setpoint_index'}

    def _reset_to_index(self, index: jp.ndarray, init_robot_state: jp.ndarray, init_robot_xy: jp.ndarray) -> State:
        setpoint = self.setpoints[jp.minimum(index, self.setpoints.shape[0] - 1)]
        state = super(WheelbotSetpoint, self).reset_at(init_xy=init_robot_xy, init_robot_state=init_robot_state)
        state = state.replace(obs=init_robot_state)
        state = self.update_setpoint_state(state, action=jp.zeros(2), setpoint=setpoint)
        state.info['setpoint_index'] = index
        # the last setpoint was reached once the index runs past the end of the array
        state.info['all_setpoints_reached'] = jp.where(index >= self.setpoints.shape[0], 1.0, 0.0)
        return state

    def reset(self, rng: jp.ndarray) -> State:
        return self._reset_to_index(jp.array(0, dtype=jp.int32), jp.zeros(10), jp.zeros(2))

    def repeat(self, state: State, rng: jp.ndarray) -> State:
        index = state.info['setpoint_index'] + 1
        return self._reset_to_index(index, state.obs[:10], state.pipeline_state.qpos[:2])

