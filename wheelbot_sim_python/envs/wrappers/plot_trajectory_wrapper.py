from brax.envs.base import State, Wrapper
import jax
from jax import numpy as jp
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches
import math


class PlotTrajectoryWrapper(Wrapper):
    """Stops batched Brax envs that are done"""
    def __init__(self, env, episode_length: int, save_dir="trajectory_plots"):
        super().__init__(env)
        self.num_plot_steps = math.ceil(episode_length / 10)
        self.save_dir = save_dir

    def log_state(self, state: State) -> State:    
        log = state.info['log']
        step = log['step']

        pos_xy = state.pipeline_state.qpos[0:2]
        setpoint = state.info['setpoint']

        log['pos_xy'] = jax.lax.cond(
            step % 10 == 0,
            lambda l: l.at[step // 10].set(pos_xy),
            lambda l: l,
            operand=log['pos_xy']
        )
        log['setpoints'] = jax.lax.cond(
            step % 10 == 0,
            lambda l: l.at[step // 10].set(setpoint),
            lambda l: l,
            operand=log['setpoints']
        )
        log['step'] = step + 1

        info = {**state.info, 'log': log}
        return state.replace(info=info)

    def reset(self, rng: jp.ndarray) -> State:
        state = self.env.reset(rng)
        log = {
                'pos_xy' : jp.zeros(shape=(self.num_plot_steps, 2)),
                'setpoints' : jp.zeros(shape=(self.num_plot_steps, 2)),
                'step' : 0,
                }
        info = {**state.info, 'log': log}
        state = state.replace(info=info)
        return self.log_state(state)


    def step(self, state: State, action: jp.ndarray) -> State:
        state = self.env.step(state, action)
        
        return self.log_state(state)
    
    def plot_trajectories(self, final_state):
        """Plots one trajectory per batch element."""
        logs = final_state.info['log']

        pos_xy = jp.array(logs['pos_xy'])      # shape (batch_size, num_plot_steps, 2)
        setpoints = jp.array(logs['setpoints'])# shape (batch_size, num_plot_steps, 2)

        # ensure unbatched (in case state is not batched)
        if pos_xy.ndim == 2:  # (T, 2)
            pos_xy = pos_xy[jp.newaxis, ...]
            setpoints = setpoints[jp.newaxis, ...]

        os.makedirs(self.save_dir, exist_ok=True)

        batch_size = pos_xy.shape[0]

        for i in range(batch_size):
            px, py = pos_xy[i].T
            sp = setpoints[i]

            # --- Efficient consecutive-unique filtering in JAX ---
            # compute diffs between consecutive setpoints
            diffs = jp.abs(sp[1:] - sp[:-1])
            # mark where a setpoint changes (sum of abs diffs > 0)
            changed = jp.any(diffs > 1e-8, axis=1)
            # first index is always a "change"
            indices = jp.concatenate([jp.array([0]), jp.nonzero(changed, size=None, fill_value=-1)[0] + 1])
            indices = indices[indices >= 0]  # filter out padding if any

            unique_setpoints = sp[indices]
            unique_setpoints = jp.array(unique_setpoints)

            plt.figure(figsize=(5, 5))
            line, = plt.plot(px, py, label='trajectory', lw=2, color='orange')
            traj_color = line.get_color()  # extract the color actually used
            plt.scatter(px[0], py[0], color=traj_color, label='start')

            ax = plt.gca()
            for idx, (x, y) in enumerate(unique_setpoints, start=1):
                circle = patches.Circle(
                    (x, y),
                    radius=self.setpoint_radius,
                    fill=True,
                    facecolor='gray',
                    edgecolor='gray',       
                    zorder=1                
                )
                ax.add_patch(circle)
                ax.text(x, y, str(idx), color='black', fontsize=11,
                        ha='center', va='center', weight='bold', zorder=2)
                
            plt.axis('equal')
            plt.tight_layout()

            filename = os.path.join(self.save_dir, f"trajectory_{i}.png")
            plt.savefig(filename, dpi=150)
            plt.close()  