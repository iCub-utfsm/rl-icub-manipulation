import numpy as np
from rl_icub_dexterous_manipulation.envs.icub_visuomanip_refine_grasp_goto import ICubEnvRefineGrasp


DEFAULT_TASKPARAMS = dict(
    model_path="../models/icub_visuomanip_ikin_limits.xml",
    initial_qpos_path='../config/icub_visuomanip_initial_qpos.yaml',
    icub_observation_space=["joints","joints_vel","cartesian","object_pose"],
    icub_action_space=["joints",],
    obs_camera="head_cam_track_hand",
    track_object=True,
    eef_name="r_hand_dh_frame",
    render_cameras=["front_cam",], #["front_cam","head_cam"]
    reward_goal=1.0,
    reward_out_of_joints=-1.0,
    reward_end_timesteps=-1.0,
    reward_single_step_multiplier=10.0,
    reward_dist_superq_center=False,
    reward_line_pregrasp_superq_center=False,
    reward_dist_original_superq_grasp_position=False,
    high_negative_reward_approach_failures=False,
    rotated_dist_superq_center=False,
    goal_reached_only_with_lift_refine_grasp=False,
    exclude_vertical_touches=False,
    min_fingers_touching_object=5,
    scale_pos_lift_reward_wrt_touching_fingers=False,
    print_done_info=True,
    random_ycb_video_graspable_object=False,
    ycb_video_graspable_objects_config_path='../config/ycb_video_objects_graspable_poses.yaml',
    random_mujoco_scanned_object=False,
    done_moved_object_mso_angle=90.0,
    mujoco_scanned_objects_config_path='../config/mujoco_scanned_objects_graspable.yaml',
    objects=["005_tomato_soup_can",],
    use_table=False,
    objects_positions=[np.array([-0.3, 0.05, 1.009],dtype=np.float32),],
    objects_quaternions=[np.array([1.0,0,0,0],dtype=np.float32),],
    randomly_rotate_object_z_axis=False,
    randomly_move_objects=False,
    random_initial_pos=not True,
    training_components=["torso_pitch","torso_yaw","torso_roll","r_arm"],
    ik_components=["torso_pitch","torso_yaw","torso_roll","r_arm"],
    cartesian_components="all_ypr",
    joints_margin=0.0,
    superquadrics_camera='head_cam', 
    feature_extractor_model_name="MAE",
    done_if_joints_out_of_limits=False,
    do_not_consider_done_z_pos=False,
    lift_object_height=1.02,
    moved_object_height=0.98,
    curriculum_learning=False, 
    curriculum_learning_approach_object=False,
    curriculum_learning_approach_object_start_step=0,
    curriculum_learning_approach_object_end_step=1000000,
    learning_from_demonstration=False,
    max_lfd_steps=10000,
    lfd_keep_only_successful_episodes=False,
    lfd_with_approach=False,
    approach_in_reset_model=True,
    pregrasp_distance_from_grasp_pose=0.05,
    max_delta_qpos=0.05,
    max_delta_cartesian_pos=0.002,
    max_delta_cartesian_rot=0.01,
    distanced_superq_grasp_pose=False,
    control_gaze=False,
    ik_solver="ikin",
    limit_torso_pitch_ikin=True,
    use_only_right_hand_model=False,
    grasp_planner='superquadrics',
    pretrained_model_dir=None,
    max_episode_steps=512,
)
DEFAULT_ENVPARAMS = dict(task_kwargs=DEFAULT_TASKPARAMS)
env = ICubEnvRefineGrasp(**DEFAULT_ENVPARAMS)


####
# Parámetros de la señal
amplitud = 1  # Ajusta la amplitud según necesites
frecuencia = 2.0  # Ajusta la frecuencia en Hertz
fase = 0.0  # Ajusta la fase en radianes

# Parámetros del bucle
duracion = 2.0  # Duración de la señal en segundos
paso_tiempo = 0.001  # Paso de tiempo en segundos

# Generación de los tiempos
tiempos = np.arange(0, duracion, paso_tiempo)

# Inicialización de la señal
# senal = np.zeros_like(tiempos)
####

images = []
obs, _ = env.reset()
# Evaluate the agent
episode_reward = 0
while True:
    
    # obs, reward, terminated, truncated, info = env.step_cartsolv()
    # action = env.action_space.sample()

    t = tiempos[env.steps]
    senal = amplitud * np.sin(2 * np.pi * frecuencia * t + fase)
    
    action = np.zeros(len(env.action_space.low), dtype=np.float32)
    # action = env.action_space.sample()
    # env.actuator_names
    # env.actuators_to_control
    # action = env.init_icub_act_after_superquadrics[env.actuators_to_control_ids]
    # action = env.target[env.actuators_to_control_ids] # Este valor es al que tiene que llegar la actuacion
    # action = env.initial_qpos[env.actuators_to_control_ids]
    
    # print(env.env.control_timestep()) # En realidad es este tiempo x 5
    # action[0] += senal
    # action[1] += senal
    # action[2] += senal
    action[3] += senal
    # action[4] += senal
    # action[5] += senal
    # action[6] += senal
    # action[7] += senal
    # action[8] += senal
    obs, reward, terminated, truncated, info = env.step(action)

    imgs = env.render()
    print(f'{info}')

    episode_reward += reward
    
    if terminated or truncated:
        break