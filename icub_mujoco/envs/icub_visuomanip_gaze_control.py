from icub_mujoco.envs.icub_visuomanip import ICubEnv
import numpy as np


class ICubEnvGazeControl(ICubEnv):

    def __init__(self, **kwargs):
        if len(kwargs.get('objects')) != 1:
            raise ValueError('There must be one and only one objects in the environment. Quitting.')

        super().__init__(**kwargs)

        self.object_com_x_y_z = self.env.physics.data.qpos[self.joint_ids_objects[0:3]]
        self.com_object_uv = self.points_in_pixel_coord([self.object_com_x_y_z])[0]
        self.goal_pixel_tolerance = 5

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # Set target w.r.t. current position for the controlled joints, while maintaining the initial position
        # for the other joints
        action += self.env.physics.data.qpos[self.joints_to_control_ids]
        action -= self.init_qpos[self.joints_to_control_ids]
        null_action = np.zeros(len(self.init_qpos))
        np.put(null_action, self.joints_to_control_ids, action)
        action = null_action
        target = np.clip(np.add(self.init_qpos, action), self.state_space.low, self.state_space.high)
        self.do_simulation(target, self.frame_skip)
        object_com_x_y_z_after_sim = self.env.physics.data.qpos[self.joint_ids_objects[0:3]]
        com_object_uv_after_sim = self.points_in_pixel_coord([object_com_x_y_z_after_sim])[0]
        done_limits = len(self.joints_out_of_range()) > 0
        done_goal = self.goal_reached(com_object_uv_after_sim)
        observation = self._get_obs()
        reward = self._get_reward(com_object_uv_after_sim, done_limits, done_goal)
        self.com_object_uv = com_object_uv_after_sim.copy()
        done_timesteps = self.steps >= self._max_episode_steps
        done_object_falling = self.falling_object() and self.use_table
        done = done_limits or done_goal or done_timesteps or done_object_falling
        info = {'Steps': self.steps,
                'Done': {'timesteps': done_timesteps,
                         'goal_reached': done_goal,
                         'limits exceeded': self.joints_out_of_range(),
                         'object falling from the table': done_object_falling}}
        if done and self.print_done_info:
            print(info)

        return observation, reward, done, info

    def _get_reward(self, com_object_uv_after_sim, done_limits, done_goal):
        if done_limits:
            return self.reward_out_of_joints
        reward = (np.linalg.norm(self.com_object_uv - np.array([320, 240]))
                  - np.linalg.norm(com_object_uv_after_sim - np.array([320, 240]))) * self.reward_single_step_multiplier
        if done_goal:
            reward += self.reward_goal
        return reward

    def goal_reached(self, com_object_uv_after_sim):
        return np.linalg.norm(com_object_uv_after_sim - np.array([320, 240])) < self.goal_pixel_tolerance
