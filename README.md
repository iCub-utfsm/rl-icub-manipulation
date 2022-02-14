# icub_mujoco

**Installation guide**

Install [**MuJoCo**](https://github.com/deepmind/mujoco/):

```
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
mkdir ~/.mujoco
tar -xf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco
```

Install **GLEW**:
```
sudo apt install libglew-dev
```

**Clone** the repository, **download iCub meshes** and **install** it:
```
git clone https://github.com/fedeceola/icub_mujoco.git
cd icub_mujoco/icub_mujoco/meshes/iCub
wget --content-disposition https://istitutoitalianotecnologia-my.sharepoint.com/:u:/g/personal/federico_ceola_iit_it/EYmkAiOMA5RHkuA-flnYeeUB9J52xdH1e-bBeN7cZcPrAg?download=1
unzip -j meshes.zip && rm meshes.zip && cd ../../..
pip install -e .
```

If you need to use the [**YCB-Video**](https://rse-lab.cs.washington.edu/projects/posecnn/) objects, download the models and convert them to the required format:

```
cd icub_mujoco/icub_mujoco/meshes/YCB_Video
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1gmcDD-5bkJfcMKLZb3zGgH_HUFbulQWu' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1gmcDD-5bkJfcMKLZb3zGgH_HUFbulQWu" -O YCB_Video_Models && rm -rf /tmp/cookies.txt
unzip YCB_Video_Models.zip && rm YCB_Video_Models.zip
cd ../../icub_mujoco/external
git clone https://github.com/deepmind/dm_robotics.git
cd dm_robotics/py/manipulation
pip install .
cd ../../../../utils
python ycb_video_obj_to_msh_meshes.py
```
