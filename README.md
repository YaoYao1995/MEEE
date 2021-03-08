# Instruction to reproduce MEEE
Code to reproduce the experiments in MEEE. 
It is noteworthy that our code is mainly based on MBPO, and we refer interested readers to the original code base [MBPO](https://github.com/JannerM/mbpo) for more details.

## Installation
1. Install `MuJoCo 2.0` at `~/.mujoco/mujoco200` and copy your license key to `~/.mujoco/mjkey.txt`, for example, you need to install the following 
dependencies first for Linux platform:
```
sudo yum install patchelf
sudo yum install mesa-libGL-devel mesa-libGLU-devel
sudo yum install mesa-libOSMesa-devel
sudo yum install mesa-libOSMesa
sudo yum install glfw
sudo yum install mesa-libGL
sudo yum install openmpi-devel
```
2. Create a conda environment and install dependencies in `requirements.txt`
```python
cd code_meee
conda create -n "your_env_name" python=3.6
conda activate "your_env_name"
# install cuda to suport tf-gpu==1.13.1
conda install cudatoolkit==10.0.130 
conda install cudnn==7.6.5
pip install -r requirements.txt
```

## Usage
Configuration files can be found in `examples/config`. Use the following command to conduct experiment on Humanoid-v2:
```python
python main.py run_local examples.development --config=examples.config.humanoid.1 --trial-gpus=1
```
Currently only running locally is supported, so just keep the `run_local` and `examples.development` arguments. `examples.config.humanoid.1` determines the configuration file you want to use, and `--trial-gpus=1` indicate that you would like to experiment with one Nvidia GPU, you could change the experiment environment and GPU used by modifying relative arguments. 

#### Logging
The results can be found in the default directory `log_dir=~/ray_meee/`, you could also specify the directory in `examples/config/configuration_files`.
