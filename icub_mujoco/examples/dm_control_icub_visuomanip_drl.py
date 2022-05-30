import numpy as np
from icub_mujoco.envs.icub_visuomanip_reaching import ICubEnvReaching
from icub_mujoco.envs.icub_visuomanip_gaze_control import ICubEnvGazeControl
from icub_mujoco.envs.icub_visuomanip_refine_grasp import ICubEnvRefineGrasp
from icub_mujoco.envs.icub_visuomanip_keep_grasp import ICubEnvKeepGrasp
from icub_mujoco.envs.icub_visuomanip_lift_grasped_object import ICubEnvLiftGraspedObject
from icub_mujoco.external.stable_baselines3_mod.sac import SAC
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--test_model',
                    action='store_true',
                    help='Test the best_model.zip stored in --eval_dir.')
parser.add_argument('--record_video',
                    action='store_true',
                    help='Test the best_model.zip stored in --eval_dir and record.')
parser.add_argument('--xml_model_path',
                    action='store',
                    type=str,
                    default='../models/icub_position_actuators_actuate_hands.xml',
                    help='Set the path of the xml model.')
parser.add_argument('--tensorboard_dir',
                    action='store',
                    type=str,
                    default='tensorboards',
                    help='Set the directory where tensorboard files are saved. Default directory is tensorboards.')
parser.add_argument('--buffer_size',
                    action='store',
                    type=int,
                    default=1000000,
                    help='Set the size of the replay buffer. Default is 1000000.')
parser.add_argument('--reward_goal',
                    action='store',
                    type=float,
                    default=1.0,
                    help='Set the reward for reaching the goal. Default is 1.0.')
parser.add_argument('--reward_out_of_joints',
                    action='store',
                    type=float,
                    default=-1.0,
                    help='Set the reward for violating joints limits. Default is -1.0.')
parser.add_argument('--reward_end_timesteps',
                    action='store',
                    type=float,
                    default=-1.0,
                    help='Set the reward for exceeding number of timesteps. Default is -1.0.')
parser.add_argument('--reward_single_step_multiplier',
                    action='store',
                    type=float,
                    default=10.0,
                    help='Set the multiplication factor of the default per-step reward in meters or pixels.')
parser.add_argument('--joints_margin',
                    action='store',
                    type=float,
                    default=0.0,
                    help='Set the margin from joints limits for joints control.')
parser.add_argument('--net_arch',
                    type=int,
                    nargs='+',
                    default=[64, 64],
                    help='Set the architecture of the MLP network. Default is [64, 64]')
parser.add_argument('--train_freq',
                    action='store',
                    type=int,
                    default=10,
                    help='Set the update frequency for SAC. Default is 10')
parser.add_argument('--total_training_timesteps',
                    action='store',
                    type=int,
                    default=10000000,
                    help='Set the number of training episodes for SAC. Default is 10M')
parser.add_argument('--eval_dir',
                    action='store',
                    type=str,
                    default='logs_eval',
                    help='Set the directory where evaluation files are saved. Default directory is logs_eval.')
parser.add_argument('--eval_freq',
                    action='store',
                    type=int,
                    default=100000,
                    help='Set the evaluation frequency for SAC. Default is 100k')
parser.add_argument('--icub_observation_space',
                    type=str,
                    nargs='+',
                    default='joints',
                    help='Set the observation space: joints will use as observation space joints positions, '
                         'camera will use information from the camera specified with the argument obs_camera, '
                         'features the features extracted by the camera specified with the argument obs_camera '
                         'and touch the tactile information. If you pass multiple argument, you will use a '
                         'MultiInputPolicy.')
parser.add_argument('--eef_name',
                    type=str,
                    default='r_hand',
                    help='Specify the name of the body to be considered as end-effector')
parser.add_argument('--render_cameras',
                    type=str,
                    nargs='+',
                    default=[],
                    help='Set the cameras used for rendering. Available cameras are front_cam and head_cam.')
parser.add_argument('--obs_camera',
                    type=str,
                    default='head_cam',
                    help='Set the cameras used for observation. Available cameras are front_cam and head_cam.')
parser.add_argument('--track_object',
                    action='store_true',
                    help='Set the target of the tracking camera to the object, instead of the default r_hand')
parser.add_argument('--curriculum_learning',
                    action='store_true',
                    help='Use curriculum learning for joints offsets.')
parser.add_argument('--superquadrics_camera',
                    type=str,
                    default='head_cam',
                    help='Set the cameras used for observation. Available cameras are front_cam and head_cam.')
parser.add_argument('--print_done_info',
                    action='store_true',
                    help='Print information at the end of each episode')
parser.add_argument('--objects',
                    type=str,
                    nargs='+',
                    default=[],
                    help='Specify YCB-Video objects to be added to the scene. Available objects are: '
                         '002_master_chef_can, 003_cracker_box, 004_sugar_box, 005_tomato_soup_can, '
                         '006_mustard_bottle, 007_tuna_fish_can, 008_pudding_box, 009_gelatin_box, '
                         '010_potted_meat_can, 011_banana, 019_pitcher_base, 021_bleach_cleanser, 024_bowl, 025_mug, '
                         '035_power_drill, 036_wood_block, 037_scissors, 040_large_marker, 051_large_clamp, '
                         '052_extra_large_clamp, 061_foam_brick')
parser.add_argument('--use_table',
                    action='store_true',
                    help='Add table in the environment')
parser.add_argument('--fixed_initial_pos',
                    action='store_true',
                    help='Use a fixed initial position for the controlled joints and the objects in the environment.')
parser.add_argument('--objects_positions',
                    type=str,
                    nargs='+',
                    default=[],
                    help='Specify objects initial positions. They must be in the order x_1 y_1 z_1 ... x_n y_n z_n '
                         'for the n objects specified with the argument objects. '
                         'If the value are not specified, the initial position of all the objects is set randomly.')
parser.add_argument('--objects_quaternions',
                    type=str,
                    nargs='+',
                    default=[],
                    help='Specify objects initial positions. They must be in the order w_1 x_1 y_1 z_1 ... w_n x_n y_n '
                         'z_n for the n objects specified with the argument objects. '
                         'If the value are not specified, the initial orientation of all the objects is set randomly.')
parser.add_argument('--randomly_rotate_object_z_axis',
                    action='store_true',
                    help='Randomy rotate objects on the table around the z axis.')
parser.add_argument('--task',
                    type=str,
                    default='reaching',
                    help='Set the task to perform.')
parser.add_argument('--training_components',
                    type=str,
                    nargs='+',
                    default=[],
                    help='Specify the joints that must be trained. Choose values in r_arm, l_arm, r_wrist, l_wrist, '
                         'r_hand, l_hand, neck, torso, torso_yaw or all to train all the joints.')
parser.add_argument('--ik_components',
                    type=str,
                    nargs='+',
                    default=[],
                    help='Specify the joints that must be used for inverse kinematics computation. Choose values in '
                         'r_arm, l_arm, r_hand, l_hand, neck, torso, torso_yaw or all to use all the joints.')
parser.add_argument('--cartesian_components',
                    type=str,
                    nargs='+',
                    default=['all'],
                    help='Specify the joints that must be used for cartesian control. Choose values in x, y, z, qw, '
                         'qx, qy, qz or all (default option) to use all the joints.')
parser.add_argument('--training_device',
                    type=str,
                    default='auto',
                    help='Set the training device. Available options are cuda, cpu or auto, which is also the default '
                         'value.')
parser.add_argument('--null_reward_out_image',
                    action='store_true',
                    help='Set reward equal to 0 for the gaze control task, if the center of mass of the object is '
                         'outside the image.')
parser.add_argument('--feature_extractor_model_name',
                    type=str,
                    default='alexnet',
                    help='Set feature extractor to process image input if features in icub_observation_space.')
parser.add_argument('--lift_object_height',
                    action='store',
                    type=float,
                    default=1.05,
                    help='Set the height of the object to complete the grasp refinement task. Default is 1.02.')
parser.add_argument('--learning_from_demonstration',
                    action='store_true',
                    help='Use demonstrations for replay buffer initialization.')
parser.add_argument('--max_delta_qpos',
                    action='store',
                    type=float,
                    default=0.1,
                    help='Set max delta qpos for joints control. Default is 0.1.')
parser.add_argument('--max_delta_cartesian_pos',
                    action='store',
                    type=float,
                    default=0.02,
                    help='Set max delta pos for cartesian control. Default is 0.02.')

args = parser.parse_args()

objects_positions = []
num_pos = 0
curr_obj_pos = ''
for pos in args.objects_positions:
    curr_obj_pos += pos
    if num_pos < 2:
        curr_obj_pos += ' '
        num_pos += 1
    else:
        objects_positions.append(curr_obj_pos)
        num_pos = 0
        curr_obj_pos = ''

objects_quaternions = []
num_quat = 0
curr_obj_quat = np.empty(shape=0, dtype=np.float32)
for quat in args.objects_quaternions:
    curr_obj_quat = np.append(curr_obj_quat, quat)
    if num_quat < 3:
        num_quat += 1
    else:
        objects_quaternions.append(curr_obj_quat)
        num_quat = 0
        curr_obj_quat = np.empty(shape=0, dtype=np.float32)

if args.task == 'reaching':
    iCub = ICubEnvReaching(model_path=args.xml_model_path,
                           icub_observation_space=args.icub_observation_space,
                           obs_camera=args.obs_camera,
                           track_object=args.track_object,
                           eef_name=args.eef_name,
                           render_cameras=tuple(args.render_cameras),
                           reward_goal=args.reward_goal,
                           reward_out_of_joints=args.reward_out_of_joints,
                           reward_end_timesteps=args.reward_end_timesteps,
                           reward_single_step_multiplier=args.reward_single_step_multiplier,
                           print_done_info=args.print_done_info,
                           objects=args.objects,
                           use_table=args.use_table,
                           objects_positions=objects_positions,
                           objects_quaternions=objects_quaternions,
                           random_initial_pos=not args.fixed_initial_pos,
                           training_components=args.training_components,
                           joints_margin=args.joints_margin,
                           feature_extractor_model_name=args.feature_extractor_model_name,
                           learning_from_demonstration=args.learning_from_demonstration,
                           max_delta_qpos=args.max_delta_qpos,
                           max_delta_cartesian_pos=args.max_delta_cartesian_pos)
elif args.task == 'gaze_control':
    iCub = ICubEnvGazeControl(model_path=args.xml_model_path,
                              icub_observation_space=args.icub_observation_space,
                              obs_camera=args.obs_camera,
                              track_object=args.track_object,
                              eef_name=args.eef_name,
                              render_cameras=tuple(args.render_cameras),
                              reward_goal=args.reward_goal,
                              reward_out_of_joints=args.reward_out_of_joints,
                              reward_end_timesteps=args.reward_end_timesteps,
                              reward_single_step_multiplier=args.reward_single_step_multiplier,
                              print_done_info=args.print_done_info,
                              objects=args.objects,
                              use_table=args.use_table,
                              objects_positions=objects_positions,
                              objects_quaternions=objects_quaternions,
                              random_initial_pos=not args.fixed_initial_pos,
                              training_components=args.training_components,
                              joints_margin=args.joints_margin,
                              null_reward_out_image=args.null_reward_out_image,
                              feature_extractor_model_name=args.feature_extractor_model_name,
                              learning_from_demonstration=args.learning_from_demonstration,
                              max_delta_qpos=args.max_delta_qpos,
                              max_delta_cartesian_pos=args.max_delta_cartesian_pos)
elif args.task == 'refine_grasp':
    iCub = ICubEnvRefineGrasp(model_path=args.xml_model_path,
                              icub_observation_space=args.icub_observation_space,
                              obs_camera=args.obs_camera,
                              track_object=args.track_object,
                              eef_name=args.eef_name,
                              render_cameras=tuple(args.render_cameras),
                              reward_goal=args.reward_goal,
                              reward_out_of_joints=args.reward_out_of_joints,
                              reward_end_timesteps=args.reward_end_timesteps,
                              reward_single_step_multiplier=args.reward_single_step_multiplier,
                              print_done_info=args.print_done_info,
                              objects=args.objects,
                              use_table=args.use_table,
                              objects_positions=objects_positions,
                              objects_quaternions=objects_quaternions,
                              randomly_rotate_object_z_axis=args.randomly_rotate_object_z_axis,
                              random_initial_pos=not args.fixed_initial_pos,
                              training_components=args.training_components,
                              ik_components=args.ik_components,
                              cartesian_components=args.cartesian_components,
                              joints_margin=args.joints_margin,
                              superquadrics_camera=args.superquadrics_camera,
                              feature_extractor_model_name=args.feature_extractor_model_name,
                              done_if_joints_out_of_limits=False,
                              lift_object_height=args.lift_object_height,
                              curriculum_learning=args.curriculum_learning,
                              learning_from_demonstration=args.learning_from_demonstration,
                              max_delta_qpos=args.max_delta_qpos,
                              max_delta_cartesian_pos=args.max_delta_cartesian_pos)
elif args.task == 'keep_grasp':
    iCub = ICubEnvKeepGrasp(model_path=args.xml_model_path,
                            icub_observation_space=args.icub_observation_space,
                            obs_camera=args.obs_camera,
                            track_object=args.track_object,
                            eef_name=args.eef_name,
                            render_cameras=tuple(args.render_cameras),
                            reward_goal=args.reward_goal,
                            reward_out_of_joints=args.reward_out_of_joints,
                            reward_end_timesteps=args.reward_end_timesteps,
                            reward_single_step_multiplier=args.reward_single_step_multiplier,
                            print_done_info=args.print_done_info,
                            objects=args.objects,
                            use_table=args.use_table,
                            objects_positions=objects_positions,
                            objects_quaternions=objects_quaternions,
                            random_initial_pos=not args.fixed_initial_pos,
                            training_components=args.training_components,
                            ik_components=args.ik_components,
                            cartesian_components=args.cartesian_components,
                            joints_margin=args.joints_margin,
                            superquadrics_camera=args.superquadrics_camera,
                            feature_extractor_model_name=args.feature_extractor_model_name,
                            done_if_joints_out_of_limits=False,
                            lift_object_height=args.lift_object_height,
                            curriculum_learning=args.curriculum_learning,
                            learning_from_demonstration=args.learning_from_demonstration,
                            max_delta_qpos=args.max_delta_qpos,
                            max_delta_cartesian_pos=args.max_delta_cartesian_pos)
elif args.task == 'lift_grasped_object':
    iCub = ICubEnvLiftGraspedObject(model_path=args.xml_model_path,
                                    icub_observation_space=args.icub_observation_space,
                                    obs_camera=args.obs_camera,
                                    track_object=args.track_object,
                                    eef_name=args.eef_name,
                                    render_cameras=tuple(args.render_cameras),
                                    reward_goal=args.reward_goal,
                                    reward_out_of_joints=args.reward_out_of_joints,
                                    reward_end_timesteps=args.reward_end_timesteps,
                                    reward_single_step_multiplier=args.reward_single_step_multiplier,
                                    print_done_info=args.print_done_info,
                                    objects=args.objects,
                                    use_table=args.use_table,
                                    objects_positions=objects_positions,
                                    objects_quaternions=objects_quaternions,
                                    random_initial_pos=not args.fixed_initial_pos,
                                    training_components=args.training_components,
                                    ik_components=args.ik_components,
                                    cartesian_components=args.cartesian_components,
                                    joints_margin=args.joints_margin,
                                    superquadrics_camera=args.superquadrics_camera,
                                    feature_extractor_model_name=args.feature_extractor_model_name,
                                    done_if_joints_out_of_limits=False,
                                    lift_object_height=args.lift_object_height,
                                    curriculum_learning=args.curriculum_learning,
                                    learning_from_demonstration=args.learning_from_demonstration,
                                    max_delta_qpos=args.max_delta_qpos,
                                    max_delta_cartesian_pos=args.max_delta_cartesian_pos)
else:
    raise ValueError('The task specified as argument is not valid. Quitting.')

if args.test_model:
    model = SAC.load(args.eval_dir + '/best_model.zip', env=iCub)
    obs = iCub.reset()
    images = []
    # Evaluate the agent
    episode_reward = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = iCub.step(action)
        imgs = iCub.render()
        if args.record_video:
            images.append(imgs)
        episode_reward += reward
        if done:
            break
    if args.record_video:
        print('Recording video.')
        for i in range(len(args.render_cameras)):
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            writer = cv2.VideoWriter(args.eval_dir + '/{}.mp4'.format(args.render_cameras[i]), fourcc, 30, (640, 480))
            for num_img, imgs in enumerate(images):
                writer.write(imgs[i][:, :, ::-1])
            writer.release()
    print("Reward:", episode_reward)
else:
    if ('joints' in args.icub_observation_space or 'cartesian' in args.icub_observation_space
        or 'features' in args.icub_observation_space or 'touch' in args.icub_observation_space
        or 'flare' in args.icub_observation_space) \
            and len(args.icub_observation_space) == 1:
        model = SAC("MlpPolicy",
                    iCub,
                    verbose=1,
                    tensorboard_log=args.tensorboard_dir,
                    policy_kwargs=dict(net_arch=args.net_arch),
                    train_freq=args.train_freq,
                    create_eval_env=True,
                    buffer_size=args.buffer_size,
                    device=args.training_device,
                    curriculum_learning=args.curriculum_learning,
                    curriculum_learning_components=iCub.cartesian_actions_curriculum_learning,
                    learning_from_demonstration=args.learning_from_demonstration)
    elif 'camera' in args.icub_observation_space and len(args.icub_observation_space) == 1:
        model = SAC("CnnPolicy",
                    iCub,
                    verbose=1,
                    tensorboard_log=args.tensorboard_dir,
                    policy_kwargs=dict(net_arch=args.net_arch),
                    train_freq=args.train_freq,
                    create_eval_env=True,
                    buffer_size=args.buffer_size,
                    device=args.training_device,
                    curriculum_learning=args.curriculum_learning,
                    curriculum_learning_components=iCub.cartesian_actions_curriculum_learning,
                    learning_from_demonstration=args.learning_from_demonstration)
    elif ('camera' in args.icub_observation_space
          or 'joints' in args.icub_observation_space
          or 'cartesian' in args.icub_observation_space
          or 'features' in args.icub_observation_space
          or 'touch' in args.icub_observation_space
          or 'flare' in args.icub_observation_space) and len(args.icub_observation_space) > 1:
        model = SAC("MultiInputPolicy",
                    iCub,
                    verbose=1,
                    tensorboard_log=args.tensorboard_dir,
                    policy_kwargs=dict(net_arch=args.net_arch),
                    train_freq=args.train_freq,
                    create_eval_env=True,
                    buffer_size=args.buffer_size,
                    device=args.training_device,
                    curriculum_learning=args.curriculum_learning,
                    curriculum_learning_components=iCub.cartesian_actions_curriculum_learning,
                    learning_from_demonstration=args.learning_from_demonstration)
    else:
        raise ValueError('The observation space specified as argument is not valid. Quitting.')

    model.learn(total_timesteps=args.total_training_timesteps,
                eval_freq=args.eval_freq,
                eval_env=iCub,
                eval_log_path=args.eval_dir)
