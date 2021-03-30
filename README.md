[//]: # tennis.png (Image References)

[image1]: tennis.png "Trained Agent"

# Project 3: Collaboration and Competition

### Introduction

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

  * After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
    This yields a single score for each episode.
  * This yields a single score for each episode. 


![Unity ML-Agents Tennis Environment][image1]

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

### Getting Started
#### Step 1: Clone the DRLND Repository

Follow the [instructions in the DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your Python environment. These instructions can be found in README.md at the root of the repository. By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.

(For Windows users) The ML-Agents toolkit supports Windows 10. While it might be possible to run the ML-Agents toolkit using other versions of Windows, it has not been tested on other versions. Furthermore, the ML-Agents toolkit has not been tested on a Windows VM such as Bootcamp or Parallels.

#### Step 2: Download the Unity Environment

For this project, you will **not** need to install Unity - this is because the environment is already built for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:

 - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
 - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
 - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
 - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip) - **Note: The Agent was implemented and trained on this version !**

Then, place the file in the `p3_collab-compet/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.

(For Windows users) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

### Instructions

Follow the instructions in `Tennis_MADDPG.ipynb` to train your own agent or test the already trained agent:
1. Start the Environment
2. Examine the State and Action Spaces
3. Optionally take Random Actions in the Environment
4. Train the Agent (Check Further Modifications section or optionally you can skip to 5. Testing)
5. Test trained Agent

### Further Modifications

In searching for better performance you can modify:
1. Training process by modifying `hyperparameters` (check Report.md for my hyperparameters search history)

        hyperparameters = {
            'BUFFER_SIZE' : int(1e6),  # replay buffer size
            'BATCH_SIZE' : 256,        # minibatch size
            'GAMMA' : 0.98,            # discount factor
            'TAU' : 0.001,             # for soft update of target parameters
            'LR_ACTOR' : 0.0001,       # learning rate of the actor
            'LR_CRITIC' : 0.0001,      # learning rate of the critic
            'WEIGHT_DECAY' : 0,        # L2 weight decay
            'UPDATE_EVERY' : 4,        # learn every UPDATE_EVERY time steps
            'step_n_updates' : 4       # number of updates during each step (from memory)
        }

 2. Modifying any part of Agent's Neural Network's Policy architecture in section 4.1.

        Actor : state_size --> Linear[256] -> ReLU --> Linear[128] -> ReLU --> Linear[64] -> ReLU --> Linear[action_size] -> tanh
        Critic: (state_size+action_size) --> Linear[256] -> ReLU --> Linear[256] -> ReLU --> Linear[64] -> ReLU --> Linear[1]

3. Implementing different Policy Search Algorithms (like PPO, A2C and others)
4. It would be interesting to try Convolutions(between states), LSTM and [Attention](https://arxiv.org/abs/1706.03762) mechanism for this kind of problem.

For more information check the [Report.md](Report.md)