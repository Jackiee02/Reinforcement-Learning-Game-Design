# 🍄Dueling DQN for Reinforcement Learning in Super Mario Bros

This project implements a reinforcement learning agent that uses the Dueling DQN architecture to play the game *Super Mario Bros*. The agent learns to navigate through the game, overcoming obstacles and enemies, by maximizing its score. The project leverages the *gym-super-mario-bros* environment and PyTorch to implement and train the model.

## Table of Contents

- [Project Overview](#project-overview)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Evaluation](#evaluation)
- [Results](#results)
- [Future Work](#future-work)

## Project Overview

This project utilizes the Dueling DQN architecture for training an agent to play *Super Mario Bros*. The agent learns by interacting with the environment and adjusting its policy based on the rewards it receives. The main objective is to maximize the cumulative score by efficiently navigating through the game levels.

Key highlights:
- **Dueling DQN architecture** improves learning efficiency by splitting the Q-value into state value and action advantage.
- **Gym environment** used for the *Super Mario Bros* game simulation.
- **PyTorch** is employed for training the neural network.

## Dependencies

The project requires the following Python packages(Please refer to the **_requirements.txt_** file for details.):
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
├── duel_dqn.py                 # Main Dueling DQN agent implementation
├── eval.py                     # Evaluation script for the trained agent
├── wrappers.py                 # Custom gym wrappers to enhance training
├── curve_picture.py            # Plotting script for training curves (score, loss, time)
├── requirements.txt            # List of required Python dependencies
├── MarioModel.z01              # Part 1 of the compressed model
├── MarioModel.z02              # Part 2 of the compressed model
├── MarioModel.z03              # Part 3 of the compressed model
├── MarioModel.zip              # Compressed model file (contains all parts)
├── train.log                   # Log file with training progress
├── loss_vs_epoch.png           # Plot of loss vs epoch
├── score_vs_epoch.png          # Plot of score vs epoch
├── time_spent_vs_epoch.png     # Plot of time spent vs epoch
├── Mario.png                   # Image displaying Mario's game screen
├── Score.png                   # Image showing the agent's score
└── README.md                   # Project documentation
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
   And then, you can get three file:mario_q.pth, mario_q_target.pth and train.log
2. **Evaluate the Model:** To test the performance of the trained agent, run:
   ```bash
   python eval.py <path_to_trained_model>
   ```
   Here, the trained model is tested, and the Mario game interface along with its score is displayed through a visual interface.   
3. **Evaluate My Training Model:**
   
   3.1 If you want to evaluate my model, please：

      (1) download **_MarioModel.z01, MarioModel.z02, MarioModel.z03_**, and **_MarioModel.zip_**,
   
      (2) place them in the same folder,
   
      (3) unzip **_MarioModel.zip_**.

   3.2 If you want to view the training curves of my model, please：
   
      download my **_train.log_** and run it on **_curve_picture.py_** (！！！make sure not to download it into your project beforehand to avoid overwriting it when you train your model).

## Model Architecture
The Dueling DQN model consists of the following components:

- **Convolutional Layers:** These extract features from the game frames (input size: 84x84x4).
- **Fully Connected Layer:** Combines extracted features and prepares them for evaluation in the advantage and state value layers.
- **State Value Layer:** Represents the value of being in a particular state.
- **Advantage Layer:** Represents the relative advantage of each action.
- **Q-value Calculation:** Combines state value and advantage to compute the Q-value for each action.

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

## Evaluation
During evaluation, the agent’s performance is visualized in real-time with the score, stage, and actions taken. Training logs and graphs of loss, score, and time spent are saved for further analysis.
   ```bash
   python curve_picture.py 
   ```
The agent successfully learns to navigate through multiple levels, achieving higher scores over time. Graphs showing the loss, score, and time spent during training can be found in the repository.
- **Score vs Epoch:** Demonstrates how the agent’s score improves with training.
 
  ![score_vs_epoch](https://github.com/user-attachments/assets/011bba92-3220-43fb-8769-9bab3b592235)



- **Loss vs Epoch:** Shows the reduction in loss over time.
 
  ![loss_vs_epoch](https://github.com/user-attachments/assets/08d0f8ff-539c-4d6f-bef2-5593cf2b6442)



- **Time Spent vs Epoch:** Illustrates the efficiency of training.
 
  ![time_spent_vs_epoch](https://github.com/user-attachments/assets/e00ac0c8-c84e-488b-9ecb-ee78719916f9)

## Results
The visualization interface of the Mario game is as follows:

![Mario](https://github.com/user-attachments/assets/11217000-b7d4-4e98-af30-c1afcf61d3b9)

The Mario score result output visualization interface is as follows:

![Score](https://github.com/user-attachments/assets/3a33ad60-3ae4-4559-b140-31c4eb0b36e0)


## Future Work
There are several opportunities for future improvements and extensions to this project:

1. **Double DQN:**
   Implementing Double DQN would further reduce the overestimation bias and improve training stability.
2. **Prioritized Experience Replay:**
   Using prioritized experience replay would allow the agent to learn more efficiently by sampling more important experiences more frequently.
3. **New Game Environments:**
   The agent could be trained on different environments or more challenging levels to test its generalization and improve its learning capabilities.
4. **Curriculum Learning:**
   A curriculum learning approach could be used to gradually increase the difficulty of the levels, making it easier for the agent to learn the optimal strategies.

