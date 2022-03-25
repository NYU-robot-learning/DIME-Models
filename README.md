# Dexterous Imitation Made Easy
**Authors**: Sridhar Pandian Arunachalam*, Sneha Silwal*, Ben Evans, [Lerrel Pinto](https://lerrelpinto.com)

This is the official implementation of the paper [Dexterous Manipulation Made Easy](https://arxiv.org/abs/2203.13251). 

## Execution on Real Robot
<p align="center">
  <img width="30%" src="https://github.com/NYU-robot-learning/dime/blob/gh-pages/figs/block-8x-optimized.gif">
  <img width="30%" src="https://github.com/NYU-robot-learning/dime/blob/gh-pages/figs/fidget-8x-optimzed.gif">
  <img width="30%" src="https://github.com/NYU-robot-learning/dime/blob/gh-pages/figs/flip-2x-optimized.gif">
 </p>

## Method
![DIME](https://github.com/NYU-robot-learning/dime/blob/gh-pages/figs/intro.png)
DIME consists of two phases: demonstration colleciton, which is performed in real-time with visual feedback, and demonstration-based policy learning, which can learn to solve dexterous tasks from a limited number of demonstrations.

## Setup
The code base is split into 4 separate repositories for convenience. You can clone and setup each package by following the instructions on their respective repositories. The packages are:
- Controller code:
  - [Allegro Hand Controller](https://github.com/NYU-robot-learning/Allegro-Hand-Controller-DIME).
  - [Kinova Arm Controller](https://github.com/NYU-robot-learning/Kinova-Arm-Controller-DIME).
- [Teleop with Inverse Kinematics Package](https://github.com/NYU-robot-learning/DIME-IK-TeleOp).
- Imitation Learning Models (present in this repository).
- [State based and Image based Demonstration collection package](https://github.com/NYU-robot-learning/DIME-Demonstrations).

You need to setup the Controller packages and IK-TeleOp package before using this package. 
To install the dependencies for this package with `pip`:
```
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
Then install this package with:
```
pip3 install -e .
```

## Demonstrations
![Demonstrations](https://github.com/NYU-robot-learning/dime/blob/gh-pages/figs/demo_framework.png)
Unlike previous work, which uses one or multiple depth cameras, we collect demonstrations in real-time from a single RGB camera. We use MediaPipe's hand detector and re-target the human fingertip positions into desired fingertip positions for the AllegroHand. The low-level controller uses inverse kinematics and a PD control to reach the desired locations in 3D space.

All our data can be found in this URL: [https://drive.google.com/drive/folders/1nunGHB2EK9xvlmepNNziDDbt-pH8OAhi](https://drive.google.com/drive/folders/1nunGHB2EK9xvlmepNNziDDbt-pH8OAhi)
