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
- `gym: 0.21.0`
- `gym-super-mario-bros: 7.4.0`
- `torch: 1.13.1+cu116`
- `numpy: 1.24.4`
- `opencv-python: 4.11.0.86`
- `matplotlib: 3.7.5`
- `pandas: 2.0.3`
- `scipy: 1.10.1`
- `pygame: 2.6.1`
- `pytorch libraries: torchaudio, torchvision`
- `other libraries: tqdm, pandas, pillow, and others.`

Install dependencies using pip:

```bash
pip install -r requirements.txt
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

## Usage

1. **Train the Model:** To start training the agent, run the following command:
   ```bash
   python duel_dqn.py
   ```
2. **Evaluate the Model:** To test the performance of the trained agent, run:
   ```bash
   python eval.py <path_to_trained_model>
   ```
3. 

## Model Architecture
The Dueling DQN model consists of the following components:

**Convolutional Layers:** These extract features from the game frames (input size: 84x84x4).

**Fully Connected Layer:** Combines extracted features and prepares them for evaluation in the advantage and state value layers.

**State Value Layer:** Represents the value of being in a particular state.

**Advantage Layer:** Represents the relative advantage of each action.

**Q-value Calculation:** Combines state value and advantage to compute the Q-value for each action.

```bash
class DuelingDQN(nn.Module):
    def __init__(self, n_frame, n_action, device):
        super(DuelingDQN, self).__init__()
        self.layer1 = nn.Conv2d(n_frame, 32, 8, 4)
        self.layer2 = nn.Conv2d(32, 64, 3, 1)
        self.fc = nn.Linear(20736, 512)
        self.q = nn.Linear(512, n_action)
        self.v = nn.Linear(512, 1)
        self.device = device

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = x.view(-1, 20736)
        x = torch.relu(self.fc(x))
        adv = self.q(x)
        v = self.v(x)
        q = v + (adv - 1 / adv.shape[-1] * adv.max(-1, True)[0])
        return q
```
   
