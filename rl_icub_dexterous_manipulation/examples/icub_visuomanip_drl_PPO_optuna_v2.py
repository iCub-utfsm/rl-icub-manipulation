from typing import Any, Dict

import gymnasium
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch
import torch.nn as nn

import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor
from rl_icub_dexterous_manipulation.envs.icub_visuomanip_refine_grasp_goto import ICubEnvRefineGrasp
from rl_icub_dexterous_manipulation.envs.env_util import make_vec_env

import stable_baselines3.common.env_util as sb3
from stable_baselines3.common.evaluation import evaluate_policy

N_TRIALS = 100
N_STARTUP_TRIALS = 10
N_EVALUATIONS = 20
N_TIMESTEPS = int(1e6)
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 20

ENV_ID = "CartPole-v1"

DEFAULT_HYPERPARAMS = {
    "policy": "MultiInputPolicy",
    # "env": ENV_ID,
}

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
    randomly_move_objects=True,
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
    max_delta_qpos=0.2,
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
# env = ICubEnvRefineGrasp(**DEFAULT_ENVPARAMS)

def make_env(seed=0, training=True, gamma=0.99, n_workers=1, max_episode_steps=512):
   
    env_id = ICubEnvRefineGrasp.make_env_constructor()
    env = make_vec_env(
            env_id=env_id, 
            n_envs=n_workers,
            seed=seed,
            wrapper_class=TimeLimit,
            env_kwargs=DEFAULT_ENVPARAMS,
            # vec_env_cls=SubprocVecEnv,
            vec_monitor_cls=VecMonitor,
            # start_index=0,
            # monitor_dir=FLAGS.log_root,
            # monitor_kwargs=,
            wrapper_kwargs=dict(max_episode_steps=max_episode_steps),
            # vec_env_kwargs=dict(start_method='fork')
            )
    env = VecNormalize(env, training=training, gamma=gamma, #FLAGS.gamma,
                       norm_obs=True,
                       norm_reward=True)
    return env

def make_venv(kwargs, seed=None, training=True, n_envs=1):
    env_id = ICubEnvRefineGrasp.make_env_constructor()
    venv = sb3.make_vec_env(env_id=env_id,
                            n_envs=n_envs,
                            seed=seed, 
                            env_kwargs=DEFAULT_ENVPARAMS,
                            vec_env_cls=SubprocVecEnv, 
                            wrapper_class=TimeLimit, 
                            wrapper_kwargs=dict(max_episode_steps=kwargs["n_steps"]))
    vnenv = VecNormalize(venv=venv,training=training, gamma=kwargs["gamma"])
    return vnenv

# Code from https://github.com/DLR-RM/rl-baselines3-zoo/blob/633954f786b51871f295b93681a4fe8d5f0de39a/rl_zoo3/hyperparams_opt.py#L11
def sample_ppo_params(trial: optuna.Trial, 
                      n_actions: int = None, 
                      n_envs: int = None, 
                      additional_args: dict = None) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams.

    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512])
    n_steps = trial.suggest_categorical("n_steps", [16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
    vf_coef = trial.suggest_float("vf_coef", 0, 1)
    net_arch_type = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])
    # Uncomment for gSDE (continuous actions)
    # log_std_init = trial.suggest_float("log_std_init", -4, 1)
    # Uncomment for gSDE (continuous action)
    # sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])
    # Orthogonal initialization
    ortho_init = False
    # ortho_init = trial.suggest_categorical('ortho_init', [False, True])
    # activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
    activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    # lr_schedule = "constant"
    # Uncomment to enable learning rate schedule
    # lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    # if lr_schedule == "linear":
    #     learning_rate = linear_schedule(learning_rate)

    # TODO: account when using multiple envs
    if batch_size > n_steps:
        batch_size = n_steps

    # Independent networks usually work best
    # when not working with images
    net_arch = {
        "tiny": dict(pi=[256], vf=[256]),
        "small": dict(pi=[256, 256], vf=[256, 256]),
        "medium": dict(pi=[1024, 1024], vf=[1024, 1024]),
    }[net_arch_type]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn_name]

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        # "sde_sample_freq": sde_sample_freq,
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
    }
    

class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: gymnasium.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need.
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


def objective(trial: optuna.Trial) -> float:
    kwargs = DEFAULT_HYPERPARAMS.copy()
    # Sample hyperparameters.
    kwargs.update(sample_ppo_params(trial))
    
    # Create the RL model.
    # env = make_env(seed=None, training=True, gamma=kwargs["gamma"], n_workers=8, max_episode_steps=kwargs["n_steps"])
    venv = make_venv(kwargs=kwargs, training=True, n_envs=8)
    model = PPO(env=venv, **kwargs)
    
    # Create env used for evaluation.
    # eval_env = Monitor(gymnasium.make(ENV_ID))
    # eval_env = make_env(seed=None, training=False, gamma=kwargs["gamma"], n_workers=8, max_episode_steps=kwargs["n_steps"]) 
    eval_env = make_venv(kwargs=kwargs, training=False, n_envs=8)

    # Create the callback that will periodically evaluate and report the performance.
    # eval_callback = TrialEvalCallback(
    #     eval_env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ, deterministic=True, verbose=1
    # )

    nan_encountered = False
    try:
        # model.learn(N_TIMESTEPS, callback=eval_callback)
        model.learn(N_TIMESTEPS)
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=N_EVAL_EPISODES)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN.
        print(e)
        nan_encountered = True
    finally:
        # Free memory.
        model.env.close()
        eval_env.close()

    # Tell the optimizer that the trial failed.
    if nan_encountered:
        return float("nan")

    # if eval_callback.is_pruned:
    #     raise optuna.exceptions.TrialPruned()

    # return eval_callback.last_mean_reward
    return -1 * mean_reward


if __name__ == "__main__":
    # Set pytorch num threads to 1 for faster training.
    torch.set_num_threads(1)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used.
    # pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3)

    study = optuna.create_study(sampler=sampler, 
                                # pruner=pruner, 
                                direction="maximize",
                                storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
                                study_name="optuna_v3")
    try:
        study.optimize(objective, 
                       n_trials=N_TRIALS, 
                       timeout=600,
                       show_progress_bar=True)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))



