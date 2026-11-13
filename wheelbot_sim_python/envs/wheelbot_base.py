from jax import numpy as jp
from jax.scipy.spatial.transform import Rotation
import jax
import mujoco
from brax.io import mjcf
from brax.envs.base import PipelineEnv, State
from omegaconf import dictconfig
from etils import epath

def Rx(theta):
    """
    Rotation matrix around x-axis.
    """
    return jp.array(
        [
            [1, 0, 0],
            [0, jp.cos(theta), -jp.sin(theta)],
            [0, jp.sin(theta), jp.cos(theta)],
        ]
    )


def Ry(theta):
    """
    Rotation matrix around y-axis.
    """
    return jp.array(
        [
            [jp.cos(theta), 0, jp.sin(theta)],
            [0, 1, 0],
            [-jp.sin(theta), 0, jp.cos(theta)],
        ]
    )


def Rz(theta):
    """
    Rotation matrix around z-axis.
    """
    return jp.array(
        [
            [jp.cos(theta), -jp.sin(theta), 0],
            [jp.sin(theta), jp.cos(theta), 0],
            [0, 0, 1],
        ]
    )


def skew_symmetric(w):
    """
    Skew symmetric matrix.
    """
    v = w.flatten()
    return jp.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def mat2Quat(mat):
    """
    Converts a rotation matrix to a quaternion.
    """
    r = Rotation.from_matrix(mat)
    q = r.as_quat(scalar_first=True)
    return q


def quat2Mat(q):
    """
    Converts a quaternion to a rotation matrix.
    """

    r = Rotation.from_quat(q[tuple([[1, 2, 3, 0]])])
    return r.as_matrix()


def jacobian_w2euler(q1, q2):
    return jp.array(
        [
            [jp.cos(q2), 0, jp.sin(q2)],
            [jp.sin(q2) * jp.tan(q1), 1, -jp.cos(q2) * jp.tan(q1)],
            [-jp.sin(q2) / jp.cos(q1), 0, jp.cos(q2) / jp.cos(q1)],
        ]
    )


del_jacobian_w2euler_del_q1 = jax.jacobian(jacobian_w2euler, 0)
del_jacobian_w2euler_del_q2 = jax.jacobian(jacobian_w2euler, 1)


def jacobian_dot_w2euler(q1, q2, dq1, dq2):
    jac_dot = del_jacobian_w2euler_del_q1(q1, q2) * jp.array(
        [dq1]
    ) + del_jacobian_w2euler_del_q2(q1, q2) * jp.array([dq2])
    return jac_dot


def robot_state_to_qpos_qvel(state, O_r_OC=None):
    C_R_W = Ry(state[2])
    O_R_C = Rz(state[0]) @ Rx(state[1])
    C_r_CW = jp.array([[0, 0, 0.032]]).T
    W_r_WB = jp.array([[0, 0, 0.0325]]).T
    O_r_dot_OC = (
        0.032
        * (state[5] + state[7])
        * jp.array([[jp.cos(state[0]), jp.sin(state[0]), 0]]).T
    )
    O_w_C = (
        jp.array([[0, 0, state[3]]]).T + Rz(state[0]) @ jp.array([[state[4], 0, 0]]).T
    )
    C_w_W = jp.array([[0, state[5], 0]]).T
    S_O_w_C = skew_symmetric(O_w_C)
    S_C_w_W = skew_symmetric(C_w_W)
    if O_r_OC is None:
        O_r_OC = jp.zeros((3, 1))
    O_r_OB = O_r_OC + O_R_C @ C_r_CW + O_R_C @ C_R_W @ W_r_WB
    R = O_R_C @ C_R_W
    quat = mat2Quat(R)
    qpos = jp.array((*(O_r_OB.flatten()), *(quat.flatten()), state[6], state[8]))
    O_w_B = R.T @ (O_w_C + O_R_C @ C_R_W @ C_w_W)
    O_r_dot_OB = (
        O_r_dot_OC
        + S_O_w_C @ O_R_C @ (C_r_CW + C_R_W @ W_r_WB)
        + O_R_C @ S_C_w_W @ C_R_W @ W_r_WB
    )
    qvel = jp.array((*(O_r_dot_OB.flatten()), *(O_w_B.flatten()), state[7], state[9]))
    return qpos, qvel


def qpos_qvel_to_robot_state(qpos, qvel):
    quat = qpos[3:7]
    R = quat2Mat(quat)
    R = R.reshape((3, 3))
    phi = jp.arctan2(-R[0, 1], R[1, 1])
    theta = jp.arcsin(R[2, 1])
    psi = jp.arctan2(-R[2, 0], R[2, 2])

    alpha = qpos[7]
    beta = qpos[8]

    omega = qvel[3:6]
    euler = jacobian_w2euler(theta, psi) @ omega

    euler = euler[tuple([[2, 0, 1]])]
    alpha_dot = qvel[6]
    beta_dot = qvel[7]

    state = jp.array([phi, theta, psi, *euler, alpha, alpha_dot, beta, beta_dot])
    return state

WHEELBOT_ROOT_PATH = epath.Path(epath.resource_path("wheelbot_sim_python"))

class WheelbotBase(PipelineEnv):
    def __init__(
        self,
        xml_str: str,
        cfg: dictconfig.DictConfig = dictconfig.DictConfig({}),
        **kwargs,
    ):
        # create the mujoco model from the xml file
        mj_model = mujoco.MjModel.from_xml_string(xml_str)
        # set the solver
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        # set the number of iterations for the solver
        mj_model.opt.iterations = 6
        # set the number of line search iterations inside each solver iteration
        mj_model.opt.ls_iterations = 6
        # timestep of the simulation
        mj_model.opt.timestep = 0.001

        # load the model into the mjx system
        sys = mjcf.load_model(mj_model)

        kwargs["n_frames"] = kwargs.get("n_frames", cfg.get("control_interval", 6))
        kwargs["backend"] = "mjx"

        super().__init__(sys, **kwargs)

        self.K_roll = jp.array(cfg.get("K_roll", [-1.3e0, -1.6e-1, -0.8e-04, -4e-04]))
        self.K_pitch = jp.array(cfg.get("K_pitch", [-400e-3, -40e-3, -4e-3, -3e-3]))
        self.max_state = jp.array([*(cfg.get("max_state",[jp.inf, jp.pi / 4, jp.pi / 4, 40, 40, 40, jp.inf, 800, jp.inf, 800]))])
        self.action_lim = cfg.get("action_lim", 0.5)
        self.action_scale = cfg.get("action_scale", 0.1)

    def reset_at(self, init_xy: jp.ndarray, init_robot_state: jp.ndarray) -> State:
        qpos, qvel = robot_state_to_qpos_qvel(init_robot_state)
        qpos = qpos.at[:2].set(init_xy)

        data = self.pipeline_init(qpos, qvel)
        obs = jp.zeros(0)
        reward, done = jp.zeros(2)
        metrics = {}
        info = {}
        
        return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jp.ndarray) -> State:
        # raw physics engine state
        data0 = state.pipeline_state
        obs = state.obs
        action_clipped = jp.clip(action, -1, 1)
        driving_wheel_torque = -self.K_pitch @ (obs[tuple([[2, 5, 6, 7]])])
        balancing_wheel_torque = -self.K_roll @ (obs[tuple([[1, 4, 8, 9]])])

        LC_torque = jp.array([driving_wheel_torque, balancing_wheel_torque])
        applied_torque = jp.clip(
            self.action_scale * action_clipped + LC_torque,
            -self.action_lim,
            self.action_lim,
        )
        data = self.pipeline_step(data0, applied_torque)

        return state.replace(pipeline_state=data)