# SPDX-FileCopyrightText: 2023 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

from setuptools import setup, find_packages

setup(
    name='rl_icub_dexterous_manipulation',
    version='0.1.1',
    packages=find_packages(),
    install_requires=[
        'mujoco',
        'imitation',
        'dm_control',
        'stable-baselines3[extra]',
        'pyyaml',
        'torchvision',
        'pyquaternion',
        'clip @ git+https://github.com/openai/CLIP.git',
        'd3rlpy',
        'mvp @ git+https://github.com/ir413/mvp',
        'open3d',
        'vgn @ git+https://github.com/ethz-asl/vgn.git',
    ]
)
