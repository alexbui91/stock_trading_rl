# Practical Deep Reinforcement Learning Approach for Stock Trading


## Prerequisites 
Python 3.6 envrionment 

### CMake, OpenMPI
Installation of system packages CMake, OpenMPI on Mac 
```bash
brew install cmake openmpi
```
    
### Activate your envrionment using using conda or Anaconda
```bash
source activate myenv
```

### Install gym under this environment
```bash
# pip install gym==0.13 
pip install matplotlib
pip install mpi4py
pip install opencv-python
pip3 install lockfile
pip3 install -U numpy
pip3 install mujoco-py==0.5.7
pip install stable-baselines
pip install tensorflow==1.15
pip install tensorflow-gpu==1.15
```

## Training model and Testing
```bash
python test_all.py 
```

#### Please cite the following paper
Xiong, Z., Liu, X.Y., Zhong, S., Yang, H. and Walid, A., 2018. Practical deep reinforcement learning approach for stock trading, NeurIPS 2018 AI in Finance Workshop.

