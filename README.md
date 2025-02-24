# Super Mario Bros - Dueling DQN Agent

This project implements a reinforcement learning agent that uses the Dueling DQN architecture to play the game *Super Mario Bros*. The agent learns to navigate through the game, overcoming obstacles and enemies, by maximizing its score. The project leverages the *gym-super-mario-bros* environment and PyTorch to implement and train the model.

## Table of Contents

- [Project Overview](#project-overview)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Training Process](#training-process)
- [Evaluation](#evaluation)
- [Technical Details](#technical-details)
- [Results](#results)
- [Challenges and Solutions](#challenges-and-solutions)
- [Future Work](#future-work)
- [License](#license)

## Project Overview

This project utilizes the Dueling DQN architecture for training an agent to play *Super Mario Bros*. The agent learns by interacting with the environment and adjusting its policy based on the rewards it receives. The main objective is to maximize the cumulative score by efficiently navigating through the game levels.

Key highlights:
- **Dueling DQN architecture** improves learning efficiency by splitting the Q-value into state value and action advantage.
- **Gym environment** used for the *Super Mario Bros* game simulation.
- **PyTorch** is employed for training the neural network.

## Dependencies

The project requires the following Python packages:
- `gym`
- `gym-super-mario-bros`
- `torch`
- `numpy`
- `matplotlib`
- `pandas`

Install dependencies using pip:

```bash
pip install gym gym-super-mario-bros torch numpy matplotlib pandas
```
## Project Structure
```
.
├── duel_dqn.py           # Main Dueling DQN agent implementation
├── eval.py               # Evaluation script for the trained agent
├── wrappers.py           # Custom gym wrappers to enhance training
├── curve_picture.py      # Plotting script for training curves (score, loss, time)
├── train.log             # Log file with training progress
└── README.md             # Project documentation
```
### Main Components

1. **Dueling DQN Agent** (`duel_dqn.py`):
   - Implements the Dueling DQN model for reinforcement learning.
   - Uses a custom neural network with convolutional layers, a fully connected layer, and separate value and advantage networks.

2. **Evaluation** (`eval.py`):
   - Evaluates the trained agent by running it through the game environment.
   - Displays real-time scores and game progress.

3. **Wrappers** (`wrappers.py`):
   - Custom wrappers to modify the behavior of the gym environment, such as frame skipping, reward clipping, and state stacking.

4. **Training Curve Visualization** (`curve_picture.py`):
   - Extracts and plots the training logs for score, loss, and time spent per epoch.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/dueling-dqn-mario
   cd dueling-dqn-mario

