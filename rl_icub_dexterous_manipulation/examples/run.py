import time
import os
import yaml
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from absl import app, flags
from rl_icub_dexterous_manipulation.sb3 import utils
from rl_icub_dexterous_manipulation.envs.icub_visuomanip_refine_grasp_goto import ICubEnvRefineGrasp

from datetime import datetime
import seaborn as sns
sns.set_theme()

FLAGS = flags.FLAGS

# Task parameters
flags.DEFINE_string('cfg',
                    default='../models/icub_visuomanip.xml',
                    help='Path of the configuration files containing the parameters for the selected experiment. '
                         'Please note that the parameters in the config file will overwrite the parameters passed '
                         'through the command line.')
flags.DEFINE_string('xml_model_path',
                    default='../models/icub_visuomanip.xml',
                    help='Set the path of the xml model.')
flags.DEFINE_string('initial_qpos_path',
                    default='../config/icub_visuomanip_initial_qpos.yaml',
                    help='Set the path of the initial actuators values.')
flags.DEFINE_string('tensorboard_dir',
                    default='tensorboards',
                    help='Set the directory where tensorboard files are saved. Default directory is tensorboards.')
flags.DEFINE_float('reward_goal', 1.0, 'Set the reward for reaching the goal. Default is 1.0.')
flags.DEFINE_float('reward_out_of_joints', -1.0, 'Set the reward for violating joints limits. Default is -1.0.')
flags.DEFINE_float('reward_end_timesteps', -1.0, 'Set the reward for exceeding number of timesteps. Default is -1.0.')
flags.DEFINE_float('reward_single_step_multiplier', 10.0, 'Set the multiplication factor of the default per-step reward in meters or pixels.')
flags.DEFINE_boolean('reward_dist_superq_center', False, 'Add a reward component in the grasp refinement task for the distance of the superquadric center in the xy axes of the eef.')
flags.DEFINE_boolean('rotated_dist_superq_center', False, 'Compute the superquardric grasp pose reward w.r.t the r_hand_dh_frame rotated of 45° around the y axis.')
flags.DEFINE_boolean('reward_line_pregrasp_superq_center', False, 'Add a reward component in the grasp refinement task for the distance to the line connecting the superquadric center and the distanced superquadric grasp pose.')
flags.DEFINE_boolean('reward_dist_original_superq_grasp_position', False, 'Add a reward component in the grasp refinement task for the distance to the original superquadric grasp position.')
flags.DEFINE_boolean('goal_reached_only_with_lift_refine_grasp', False, 'Successful episode only with object lifted in grasp refinement task.')
flags.DEFINE_boolean('high_negative_reward_approach_failures', False, 'Strongly penalize moved object in the approach phase in the grasp refinement task.')
flags.DEFINE_float('joints_margin', 0.0, 'Set the margin from joints limits for joints control.')
flags.DEFINE_string('eval_dir', 'logs_eval', 'Set the directory where evaluation files are saved. Default directory is logs_eval.')
flags.DEFINE_string('pretrained_model_dir', None, 'Set the directory where the requested pretrained model is saved.')
flags.DEFINE_multi_string('icub_observation_space', ['joints'], 'Set the observation space: joints will use as observation space joints positions, camera will use information from the camera specified with the argument obs_camera, features the features extracted by the camera specified with the argument obs_camera, flare a combination of the features with information at the previous timesteps, pretrained_output the output of the pre-trained policy stored in pretrained_model_dir, grasp_type an integer value that describes the grasp type based on the initial grasp pose and touch the tactile information. If you pass multiple argument, you will use a MultiInputPolicy.')
flags.DEFINE_multi_string('icub_action_space', ['joints'], 'TODO')
flags.DEFINE_boolean('exclude_vertical_touches', False, 'Do not consider vertical contacts to compute the number of fingers touching an object.')
flags.DEFINE_integer('min_fingers_touching_object', 5, 'Set the minimum number of fingers touching the object in the grasp refinement task to get a positive reward when lifting it. Default is 5.')
flags.DEFINE_boolean('scale_pos_lift_reward_wrt_touching_fingers', False, 'Multiply the positive lift rewards by the fraction of fingers in contact with the object.')
flags.DEFINE_string('eef_name', 'r_hand', 'Specify the name of the body to be considered as end-effector')
flags.DEFINE_multi_string('render_cameras', [], 'Set the cameras used for rendering. Available cameras are front_cam and head_cam.')
flags.DEFINE_string('obs_camera', 'head_cam', 'Set the cameras used for observation. Available cameras are front_cam and head_cam.')
flags.DEFINE_boolean('track_object', False, 'Set the target of the tracking camera to the object, instead of the default r_hand')
flags.DEFINE_boolean('curriculum_learning', False, 'Use curriculum learning for joints/cartesian offsets.')
flags.DEFINE_boolean('curriculum_learning_approach_object', False, 'Use curriculum learning in the distance of the pre-grasp pose from the object.')
flags.DEFINE_integer('curriculum_learning_approach_object_start_step', 0, 'Set the initial step for the curriculum learning phase while approaching the object.')
flags.DEFINE_integer('curriculum_learning_approach_object_end_step', 1000000, 'Set the final step for the curriculum learning phase while approaching the object.')
flags.DEFINE_string('superquadrics_camera', 'head_cam', 'Set the cameras used for observation. Available cameras are front_cam and head_cam.')
flags.DEFINE_boolean('print_done_info', False, 'Print information at the end of each episode')
flags.DEFINE_boolean('do_not_consider_done_z_pos', False, 'Do not consider the done_z_pos component in the grasp refinement task.')
flags.DEFINE_boolean('random_ycb_video_graspable_object', False, 'Use a random YCB-Video object.')
flags.DEFINE_string('ycb_video_graspable_objects_config_path', '../config/ycb_video_objects_graspable_poses.yaml', 'Set the path of configuration file with the graspable configurations of the YCB-Video objects.')
flags.DEFINE_boolean('random_mujoco_scanned_object', False, 'Use a random object from the mujoco scanned objects dataset.')
flags.DEFINE_float('done_moved_object_mso_angle', 90.0, 'Set the rotation angle in degrees around the x/y axes to consider an object as moved when using the mujoco scanned objects dataset.')
flags.DEFINE_string('mujoco_scanned_objects_config_path', '../config/mujoco_scanned_objects_graspable.yaml', 'Set the path of configuration file with the graspable mujoco_scanned_objects.')
flags.DEFINE_multi_string('objects', [], 'Specify YCB-Video objects to be added to the scene. Available objects are: 002_master_chef_can, 003_cracker_box, 004_sugar_box, 005_tomato_soup_can, 006_mustard_bottle, 007_tuna_fish_can, 008_pudding_box, 009_gelatin_box, 010_potted_meat_can, 011_banana, 019_pitcher_base, 021_bleach_cleanser, 024_bowl, 025_mug, 035_power_drill, 036_wood_block, 037_scissors, 040_large_marker, 051_large_clamp, 052_extra_large_clamp, 061_foam_brick')
flags.DEFINE_boolean('use_table', False, 'Add table in the environment')
flags.DEFINE_boolean('fixed_initial_pos', False, 'Use a fixed initial position for the controlled joints and the objects in the environment.')
flags.DEFINE_multi_string('objects_positions', [], 'Specify objects initial positions. They must be in the order x_1 y_1 z_1 ... x_n y_n z_n for the n objects specified with the argument objects. If the value are not specified, the initial position of all the objects is set randomly.')
flags.DEFINE_multi_string('objects_quaternions', [], 'Specify objects initial positions. They must be in the order w_1 x_1 y_1 z_1 ... w_n x_n y_n z_n for the n objects specified with the argument objects. If the value are not specified, the initial orientation of all the objects is set randomly.')
flags.DEFINE_boolean('randomly_rotate_object_z_axis', False, 'Randomly rotate objects on the table around the z axis.')
flags.DEFINE_boolean('randomly_move_objects', False, 'Randomly move objects on the table.')
flags.DEFINE_multi_string('training_components', [], 'Specify the joints that must be trained. Choose values in r_arm, l_arm, r_wrist, l_wrist, r_hand, l_hand, r_hand_no_thumb_oppose, l_hand_no_thumb_oppose, neck, torso, torso_yaw or all to train all the joints.')
flags.DEFINE_multi_string('ik_components', [], 'Specify the joints that must be used for inverse kinematics computation. Choose values in r_arm, l_arm, r_hand, l_hand, neck, torso, torso_yaw or all to use all the joints.')
flags.DEFINE_multi_string('cartesian_components', ['all_ypr'], 'Specify the eef components that must be used for cartesian control. Choose values in x, y, z, for eef position. For eef orientation control, choose values in qw, qx, qy, qz for quaternion orientation control or in yaw, pitch, roll for ypr orientation control. To control all the position components and all the rotation components use all_ypr (default option, with ypr orientation control) or all_quaternion (with quaternion orientation control.')
flags.DEFINE_string('training_device', 'auto', 'Set the training device. Available options are cuda, cpu or auto, which is also the default value.')
flags.DEFINE_string('feature_extractor_model_name', 'alexnet', 'Set feature extractor to process image input if features in icub_observation_space.')
flags.DEFINE_float('lift_object_height', 1.02, 'Set the height of the object to complete the grasp refinement task. Default is 1.02. Note that this parameter is not considered if the random_ycb_video_graspable_object is passed. In that case the lift_object_height is set to 10cm above the initial position of the object at hand.')
flags.DEFINE_float('moved_object_height', 0.98, 'Set the height of the object to consider the object as fallen in the grasp refinement task. Default is 0.98. Note that this parameter is not considered if the random_ycb_video_graspable_object is passed.')
flags.DEFINE_boolean('learning_from_demonstration', False, 'Use demonstrations for replay buffer initialization.')
flags.DEFINE_integer('max_lfd_steps', 10000, 'Set max learning from demonstration steps for replay buffer initialization. Default is 10000.')
flags.DEFINE_boolean('lfd_keep_only_successful_episodes', False, 'Store in the replay buffer only successful episodes in the learning from demonstration phase.')
flags.DEFINE_boolean('lfd_with_approach', False, 'Set if the approach to the object is included in the learning from demonstration phase.')
flags.DEFINE_boolean('approach_in_reset_model', False, 'Approach the object when resetting the model.')
flags.DEFINE_float('pregrasp_distance_from_grasp_pose', 0.05, 'Set the pre-grasp distance from the grasp pose.')
flags.DEFINE_float('max_delta_qpos', 0.1, 'Set max delta qpos for joints control. Default is 0.1.')
flags.DEFINE_float('max_delta_cartesian_pos', 0.02, 'Set max delta pos for cartesian control. Default is 0.02.')
flags.DEFINE_float('max_delta_cartesian_rot', 0.1, 'Set max delta rot for cartesian control. Default is 0.1.')
flags.DEFINE_boolean('distanced_superq_grasp_pose', False, 'Start the refine grasping task from a pre-grasp pose distanced --pregrasp_distance_from_grasp_pose from the desired grasp pose.')
flags.DEFINE_boolean('control_gaze', False, 'Set if using gaze control.')
flags.DEFINE_string('ik_solver', 'idyntree', 'Set the IK solver between idyntree, dm_control and ikin.')
flags.DEFINE_boolean('limit_torso_pitch_ikin', False, 'Set if using a limited range for torso_pitch joint in the iKin IK solver.')
flags.DEFINE_boolean('use_only_right_hand_model', False, 'Use only the right hand model instead of the whole iCub.')
flags.DEFINE_string('grasp_planner', 'superquadrics', 'Set the grasp planner between superquadrics and vgn.')
flags.DEFINE_boolean('done_if_joints_out_of_limits', False, 'End the episode if any joint goes out of limits.')
flags.DEFINE_boolean('record_video', False, 'Record video while testing the best model.')
flags.DEFINE_boolean('use_cartsolv', False, 'Use cartesian solver for the task.')

# PPO parameters
flags.DEFINE_string("model_root", None, "Directory where trained policies are stored")
flags.DEFINE_integer('max_episode_steps', 512, 'Maximum number of steps per episode')

def main(_):
    if FLAGS.cfg is not None:
        with open(FLAGS.cfg, 'r') as file:
            params = yaml.safe_load(file)
        for param, val in params.items():
            setattr(FLAGS, param, val)

    objects_positions = []
    num_pos = 0
    curr_obj_pos = np.empty(shape=0, dtype=np.float32)
    for pos in FLAGS.objects_positions:
        curr_obj_pos = np.append(curr_obj_pos, pos)
        if num_pos < 2:
            num_pos += 1
        else:
            objects_positions.append(curr_obj_pos)
            num_pos = 0
            curr_obj_pos = np.empty(shape=0, dtype=np.float32)

    objects_quaternions = []
    num_quat = 0
    curr_obj_quat = np.empty(shape=0, dtype=np.float32)
    for quat in FLAGS.objects_quaternions:
        curr_obj_quat = np.append(curr_obj_quat, quat)
        if num_quat < 3:
            num_quat += 1
        else:
            objects_quaternions.append(curr_obj_quat)
            num_quat = 0
            curr_obj_quat = np.empty(shape=0, dtype=np.float32)

    task_kwargs = dict(
        model_path=FLAGS.xml_model_path,
        initial_qpos_path=FLAGS.initial_qpos_path,
        icub_observation_space=FLAGS.icub_observation_space,
        icub_action_space=FLAGS.icub_action_space,
        obs_camera=FLAGS.obs_camera,
        track_object=FLAGS.track_object,
        eef_name=FLAGS.eef_name,
        render_cameras=tuple(FLAGS.render_cameras),
        reward_goal=FLAGS.reward_goal,
        reward_out_of_joints=FLAGS.reward_out_of_joints,
        reward_end_timesteps=FLAGS.reward_end_timesteps,
        reward_single_step_multiplier=FLAGS.reward_single_step_multiplier,
        reward_dist_superq_center=FLAGS.reward_dist_superq_center,
        reward_line_pregrasp_superq_center=FLAGS.reward_line_pregrasp_superq_center,
        reward_dist_original_superq_grasp_position=FLAGS.reward_dist_original_superq_grasp_position,
        high_negative_reward_approach_failures=FLAGS.high_negative_reward_approach_failures,
        rotated_dist_superq_center=FLAGS.rotated_dist_superq_center,
        goal_reached_only_with_lift_refine_grasp=FLAGS.goal_reached_only_with_lift_refine_grasp,
        exclude_vertical_touches=FLAGS.exclude_vertical_touches,
        min_fingers_touching_object=FLAGS.min_fingers_touching_object,
        scale_pos_lift_reward_wrt_touching_fingers=FLAGS.scale_pos_lift_reward_wrt_touching_fingers,
        print_done_info=FLAGS.print_done_info,
        random_ycb_video_graspable_object=FLAGS.random_ycb_video_graspable_object,
        ycb_video_graspable_objects_config_path=FLAGS.ycb_video_graspable_objects_config_path,
        random_mujoco_scanned_object=FLAGS.random_mujoco_scanned_object,
        done_moved_object_mso_angle=FLAGS.done_moved_object_mso_angle,
        mujoco_scanned_objects_config_path=FLAGS.mujoco_scanned_objects_config_path,
        objects=FLAGS.objects,
        use_table=FLAGS.use_table,
        objects_positions=objects_positions,
        objects_quaternions=objects_quaternions,
        randomly_rotate_object_z_axis=FLAGS.randomly_rotate_object_z_axis,
        randomly_move_objects=FLAGS.randomly_move_objects,
        random_initial_pos=not FLAGS.fixed_initial_pos,
        training_components=FLAGS.training_components,
        ik_components=FLAGS.ik_components,
        cartesian_components=FLAGS.cartesian_components,
        joints_margin=FLAGS.joints_margin,
        superquadrics_camera=FLAGS.superquadrics_camera,
        feature_extractor_model_name=FLAGS.feature_extractor_model_name,
        done_if_joints_out_of_limits=FLAGS.done_if_joints_out_of_limits,
        do_not_consider_done_z_pos=FLAGS.do_not_consider_done_z_pos,
        lift_object_height=FLAGS.lift_object_height,
        moved_object_height=FLAGS.moved_object_height,
        curriculum_learning=FLAGS.curriculum_learning,
        curriculum_learning_approach_object=FLAGS.curriculum_learning_approach_object,
        curriculum_learning_approach_object_start_step=FLAGS.curriculum_learning_approach_object_start_step,
        curriculum_learning_approach_object_end_step=FLAGS.curriculum_learning_approach_object_end_step,
        learning_from_demonstration=FLAGS.learning_from_demonstration,
        max_lfd_steps=FLAGS.max_lfd_steps,
        lfd_keep_only_successful_episodes=FLAGS.lfd_keep_only_successful_episodes,
        lfd_with_approach=FLAGS.lfd_with_approach,
        approach_in_reset_model=FLAGS.approach_in_reset_model,
        pregrasp_distance_from_grasp_pose=FLAGS.pregrasp_distance_from_grasp_pose,
        max_delta_qpos=FLAGS.max_delta_qpos,
        max_delta_cartesian_pos=FLAGS.max_delta_cartesian_pos,
        max_delta_cartesian_rot=FLAGS.max_delta_cartesian_rot,
        distanced_superq_grasp_pose=FLAGS.distanced_superq_grasp_pose,
        control_gaze=FLAGS.control_gaze,
        ik_solver=FLAGS.ik_solver,
        limit_torso_pitch_ikin=FLAGS.limit_torso_pitch_ikin,
        use_only_right_hand_model=FLAGS.use_only_right_hand_model,
        grasp_planner=FLAGS.grasp_planner,
        pretrained_model_dir=FLAGS.pretrained_model_dir,
        max_episode_steps=FLAGS.max_episode_steps
    )

    env = ICubEnvRefineGrasp(task_kwargs=task_kwargs)

    if FLAGS.model_root:
        model = utils.load_policy(
            FLAGS.model_root,
            list(env.observation_space.keys()),
            device=FLAGS.training_device
        )
        @torch.no_grad()
        def policy_fn(time_step):
            obs = time_step
            action, _ = model.predict(obs, deterministic=True)
            return action
        
    # Inicializa un DataFrame vacío
    df = pd.DataFrame()

    obs, _ = env.reset()
    images = []
    # Evaluate the agent
    episode_reward = 0
    post_step = 0

    # Medicion de tiempos de control
    start_time = time.time()
    time_list = []

    # Agrega unos pasos de tiempo sin hacer nada
    for _ in range(100):
        action = np.zeros(len(env.action_space.low), dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        time_list.append(time.time()-start_time)

        imgs = env.render()
        # print(f'{info}')
        if FLAGS.record_video:
            images.append(imgs)
        episode_reward += reward
        
        temp_df = pd.concat([pd.DataFrame([info]), pd.DataFrame([obs])], axis=1)
        df = pd.concat([df, temp_df], ignore_index=True)


    current_step = 0
    while True:
        
        if FLAGS.model_root:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
        else:
            if FLAGS.use_cartsolv:
                obs, reward, terminated, truncated, info = env.step_cartsolv(current_step, total_num_steps=100)
                current_step +=1
            else:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

        time_list.append(time.time()-start_time)

        imgs = env.render()
        # print(f'{info}')
        if FLAGS.record_video:
            images.append(imgs)
        episode_reward += reward
        
        temp_df = pd.concat([pd.DataFrame([info]), pd.DataFrame([obs])], axis=1)
        df = pd.concat([df, temp_df], ignore_index=True)
        if terminated or truncated:
            post_step += 1
            if post_step > 50:
                break

    df = df.assign(timestep=time_list)

    if FLAGS.record_video:
        now = datetime.now()

        print('Recording video.')
        for i in range(len(FLAGS.render_cameras)):
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            if not os.path.exists(FLAGS.eval_dir):
                os.makedirs(FLAGS.eval_dir)
            writer = cv2.VideoWriter(FLAGS.eval_dir + '/{}_{}.mp4'.format(FLAGS.render_cameras[i],
                                                                          now.strftime("%Y-%m-%d_%H-%M-%S")), 
                                                                          fourcc, 30, (640, 480))
            for num_img, imgs in enumerate(images):
                writer.write(imgs[i][:, :, ::-1])
            writer.release()
            
            df.to_json(FLAGS.eval_dir + '/df_{}.json'.format(now.strftime("%Y-%m-%d_%H-%M-%S")), orient='records')

    print("Reward:", episode_reward)

if __name__ == '__main__':
    app.run(main)