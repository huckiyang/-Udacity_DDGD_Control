# Udacity_DDGD_Control
Project 2 - Continuous Control  - Udacity Deep Reinforcement Learning Nanodegree

### Introduction of Environment 
<img src="https://github.com/huckiyang/Udacity_DDGD_Control/blob/master/image/train_1.gif" width="500" height="300">
Set-up: Double-jointed arm which can move to target locations.
Goal: The agents must move it's hand to the goal location, and keep it there.
Agents: The environment contains 10 agent linked to a single Brain.
Agent Reward Function (independent):
+0.1 Each step agent's hand is in goal location.
Brains: One Brain with the following observation/action space.
Vector Observation space: 26 variables corresponding to position, rotation, velocity, and angular velocities of the two arm Rigidbodies.
Vector Action space: (Continuous) Size of 4, corresponding to torque applicable to two joints.
Visual Observations: None.
Reset Parameters: Two, corresponding to goal size, and goal movement speed.
Benchmark Mean Reward: 30

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.


### Workspace by Udacity 
To set up your computer to run the python code in this repository, follow the instructions below.

1. Install/Setup Python 3.6+.   See the instructions for how to do this for your operating system on the official [www.python.org](www.python.org) website.

2. [Install pip for python](https://pip.pypa.io/en/stable/installing/)

3. Install dependent python packages
    - numpy (e.g. `pip install numpy`)
    - matplotlib (see [installation instructions](https://matplotlib.org/faq/installing_faq.html))
    - pytorch: Select the correct options in the "Getting Started" section of the [pytorch main page](https://pytorch.org/), then run the command created in the "Run this command:" section of that webpage.
    - jupyter notebook: (e.g. `pip install jupyter`).  If simple pip install doesn't work see jupyter's [official documentation](http://jupyter.org/install)
    
4. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.

    - Next, install the classic control environment group by following the instructions [here](https://github.com/openai/gym#classic-control).

5. Clone this GitHub repository that contains my solution to the problem.  
    - Navigate to the folder where you want to install the repository (e.g. cd C:/bananas/)

    - `git clone https://github.com/huckiyang/Udacity_DQN_Navigation.git`

        `cd python`

        `pip install .`

6. Create an IPython kernel for the drlnd environment.

    e.g. `python -m ipykernel install --user --name drlnd --display-name "drlnd"`

## Instructions
Open a command prompt/terminal and type `jupyter notebook`.  If that doesn't work, return to step 3 of "Getting Started" above to successfully install jupyter notebook.

Run `Continuous_Control.ipynb` for further details.
