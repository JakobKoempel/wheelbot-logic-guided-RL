from jax import numpy as jp
from etils import epath
import jax
from brax.envs.base import State
from omegaconf import dictconfig
import xml.etree.ElementTree as ET
from brax import envs
import mediapy as media
from .wheelbot_base import WheelbotBase, WHEELBOT_ROOT_PATH, qpos_qvel_to_robot_state
from .util.visualization import compute_line_element
from ..automata.context import Context
import itertools
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

class WheelbotClassicalSetpoint(WheelbotBase):
    def __init__(
        self,
        cfg: dictconfig.DictConfig = dictconfig.DictConfig({}),
        eval: bool = False,
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

        #track_body = root.find(".//body[@name='track']")
        #line_1 = compute_line_element(jp.array([0.,1.]), jp.array([0,1.1]))
        #track_line_elements = [line_1]
        #for elem in track_line_elements:
            #track_body.append(elem)

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

        assert self.min_theta <= self.max_theta
        assert self.min_distance <= self.max_distance
        assert self.min_distance > 0
        assert self.roll_threshold > 0
        assert self.speed_threshold >= 0
        assert self.setpoint_radius > 0

        self.eps = cfg.get("eps", 0.25)
        self.reach_reward = cfg.get("reach_reward", 200.)
        self.alpha_dist = cfg.get("alpha_dist", 4.)
        self.alpha_align = cfg.get("alpha_align", 4.)
        self.torque_reward_scale = cfg.get("torque_reward_scale", 0.5)
        self.fail_penalty = cfg.get("fail_penalty", 100.)
        self.eval = eval

    @property
    def normalization_size(self) -> int:
        return self.observation_size

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

        #get reward
        crashed = (jp.abs(robot_state) > self.max_state).any()
        reached = (setpoint_distance < self.eps).astype(jp.float32)
        fail = crashed
        done = jp.logical_or(reached, fail).astype(jp.float32)
        reach_reward = self.reach_reward * reached
        alignment_reward = jp.exp(-jp.abs(setpoint_yaw) * self.alpha_align)
        torque_reward = -1 * action[0] * self.torque_reward_scale
        fail_penalty = self.fail_penalty * crashed

        if self.eval:
            reward = reached
        else:
            reward = reach_reward + torque_reward + alignment_reward - fail_penalty 
        
        # ensure state.obs has at least length of obs
        obs_len = obs.shape[0]
        state_obs_len = state.obs.shape[0]
        if state_obs_len < obs_len:
            pad_width = obs_len - state_obs_len
            state_obs = jp.concatenate([state.obs, jp.zeros(pad_width)])
        else:
            state_obs = state.obs

        updated_obs = state_obs.at[:obs_len].set(obs)
        state = state.replace(obs=updated_obs, reward=reward, done=done)
        state.info['setpoint'] = setpoint
        state.info['active_assignment'] = self.get_active_assignment(obs, action)
        return state
    
    def reset(self, rng: jp.ndarray) -> State:
        return self._reset(rng, init_robot_state=jp.zeros(10), init_robot_xy=jp.zeros(2))
    
    def repeat(self, state: State, rng: jp.ndarray) -> State:
        robot_state = state.obs[:10]
        current_robot_xy = state.pipeline_state.qpos[:2]
        return self._reset(rng, init_robot_state=robot_state, init_robot_xy=current_robot_xy)
    
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



@hydra.main(config_path="../config", config_name="classical_main", version_base=None)
def main(cfg: omegaconf.DictConfig) -> None:
    from brax.io import model
    from ..nets.partial_norm_obs_net import make_inference_fn, make_sac_networks
    from brax.training.acme import running_statistics
    import matplotlib.pyplot as plt 
    import numpy as np
    from matplotlib.patches import Circle

    cfg.env['min_theta'] = 0
    cfg.env['max_theta'] = 0

    #create environment
    env = WheelbotClassicalSetpoint(cfg=cfg.env, eval=True)
    # load the model
    """model_path = cfg.agent_dir
    params = model.load_params(model_path)[0:2]
    normalize_fn = running_statistics.normalize
    sac_network = make_sac_networks(
                    observation_size=env.observation_size,
                    normalization_size=env.normalization_size,
                    action_size=env.action_size,
                    hidden_layer_sizes=cfg.algorithm.hidden_layer_sizes,
                    preprocess_observations_fn=normalize_fn
    )

    # make policy inference
    make_inf_fn = make_inference_fn(sac_network)
    inference_fn = make_inf_fn(params, deterministic=True)
    jit_inference_fn = jax.jit(lambda obs, act_rng: jp.ravel(inference_fn(obs, act_rng)[0]))""" # removed extras

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    n_steps = 1000

    # initialize the state
    rng = jax.random.PRNGKey(11)
    state = jit_reset(rng)
    setpoint = state.info['setpoint']
    print('setpoint', setpoint)
    print(state.obs)
    rollout = [state.pipeline_state]
    balancing_wheel_speed = [state.obs[9]]

    for i in range(3000):
        ctrl = jp.array([-0., 0]) #jit_inference_fn(state.obs, rng)
        state = jit_step(state, ctrl)
        rollout.append(state.pipeline_state)
        balancing_wheel_speed.append(state.obs[9])

    print(state.obs)
    for i in range(1000):
        ctrl = jp.array([-0.5, 0]) #jit_inference_fn(state.obs, rng)
        state = jit_step(state, ctrl)
        rollout.append(state.pipeline_state)
        balancing_wheel_speed.append(state.obs[9])
    
    
    #media.write_video("video_floating.mp4", env.render(rollout, camera="floating_far"), fps=1.0 / env.dt)
    print("Saved video_floating.mp4!!!")

    
    xy_traj = np.array([s.qpos[:2] for s in rollout])
    x = xy_traj[:, 0]
    y = xy_traj[:, 1]
    balancing_wheel_speed = np.array(balancing_wheel_speed)
    time_steps = np.arange(len(balancing_wheel_speed))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Trajectory plot
    ax1.plot(x, y, color="orange", linewidth=2, label="Trajectory", zorder=1)
    ax1.scatter(x[0], y[0], color="orange", s=80, label="Start", zorder=2)

    circle = Circle(
        (setpoint[0], setpoint[1]),
        radius=0.25,  
        edgecolor="grey",
        facecolor="grey",
        linewidth=2,
        zorder=4,
    )
    ax1.add_patch(circle)

    ax1.set_frame_on(True)
    ax1.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax1.tick_params(axis='both', which='major', labelsize=12)

    ax1.grid(True, color='lightgrey', linewidth=0.5)
    ax1.set_xlim(-1.9, 1.9)
    ax1.set_ylim(-1.9, 1.9)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Trajectory XY')
    ax1.legend()

    # Balancing wheel speed over time
    ax2.plot(time_steps, balancing_wheel_speed, color="blue", linewidth=2, label="Balancing Wheel Speed")
    ax2.set_frame_on(True)
    ax2.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.grid(True, color='lightgrey', linewidth=0.5)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Angle Speed (rad/s)')
    ax2.set_title('Balancing Wheel Angle Speed')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('trajectory_xy.png', bbox_inches='tight')
    plt.close()

    print('Trajectory plotted!')

if __name__ == "__main__":
    main()