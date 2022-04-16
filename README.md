# Dexterous Imitation Made Easy
**Authors**: Sridhar Pandian Arunachalam*, [Sneha Silwal](http://ssilwal.com/)*, [Ben Evans](https://bennevans.github.io/), [Lerrel Pinto](https://lerrelpinto.com)

This is the official implementation of the paper [Dexterous Manipulation Made Easy](https://arxiv.org/abs/2203.13251). 

## Policies on Real Robot
<p align="center">
  <img width="30%" src="https://github.com/NYU-robot-learning/dime/blob/gh-pages/figs/block-8x-optimized.gif">
  <img width="30%" src="https://github.com/NYU-robot-learning/dime/blob/gh-pages/figs/fidget-8x-optimzed.gif">
  <img width="30%" src="https://github.com/NYU-robot-learning/dime/blob/gh-pages/figs/flip-2x-optimized.gif">
 </p>

## Method
![DIME](https://github.com/NYU-robot-learning/dime/blob/gh-pages/figs/intro.png)
DIME consists of two phases: demonstration colleciton, which is performed in real-time with visual feedback, and demonstration-based policy learning, which can learn to solve dexterous tasks from a limited number of demonstrations.

## Setup
The code base is split into 5 separate packages for convenience and this is one out of the five repositories. You can clone and setup each package by following the instructions on their respective repositories. The packages are:
- [Robot controller packages](https://github.com/NYU-robot-learning/DIME-Controllers):
  - [Allegro Hand Controller](https://github.com/NYU-robot-learning/Allegro-Hand-Controller-DIME).
  - [Kinova Arm Controller](https://github.com/NYU-robot-learning/Kinova-Arm-Controller-DIME).
- [Camera packages](https://github.com/NYU-robot-learning/DIME-Camera-Packages)
  - [Realsense-ROS](https://github.com/NYU-robot-learning/Realsense-ROS-DIME).
  - [AR_Tracker_Alvar](https://github.com/ros-perception/ar_track_alvar).
- (Phase 1) Demonstration collection packages:
  - [Teleop with Inverse Kinematics Package](https://github.com/NYU-robot-learning/DIME-IK-TeleOp).
  - [State based and Image based Demonstration collection package](https://github.com/NYU-robot-learning/DIME-Demonstrations).
- (Phase 2) Nearest-Neighbor Imitation Learning (present in this repository).
- Simulation [environments](https://github.com/NYU-robot-learning/dime_env) and DAPG related codebase.

You need to setup the Controller packages and IK-TeleOp package before using this package. 
To install the dependencies for this package with `pip`:
```
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
Then install this package with:
```
pip3 install -e .
```

## Data
All our data can be found in this URL: [https://drive.google.com/drive/folders/1nunGHB2EK9xvlmepNNziDDbt-pH8OAhi](https://drive.google.com/drive/folders/1nunGHB2EK9xvlmepNNziDDbt-pH8OAhi)

## Citation

If you use this repo in your research, please consider citing the paper as follows:
```
@article{arunachalam2022dime,
  title={Dexterous Imitation Made Easy: A Learning-Based Framework for Efficient Dexterous Manipulation},
  author={Sridhar Pandian Arunachalam and Sneha Silwal and Ben Evans and Lerrel Pinto},
  journal={arXiv preprint arXiv:2203.13251},
  year={2022}
}
