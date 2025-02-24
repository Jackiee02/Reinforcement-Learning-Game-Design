# Dueling DQN for Super Mario Bros

This project implements a reinforcement learning agent to play the classic video game *Super Mario Bros* using the Dueling Deep Q-Network (DQN) algorithm. The agent learns to navigate through the game, overcoming obstacles and enemies, by maximizing its score. The project leverages the gym library for the environment, PyTorch for the neural network, and several custom wrappers to enhance training.

## Table of Contents
- [Overview](#overview)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)

## Overview
This project focuses on applying Q-learning, specifically the Dueling DQN architecture, to train an AI agent to play *Super Mario Bros*. The Dueling DQN improves the stability and efficiency of the learning process by separating the state value and action advantages. The agent learns to make optimal decisions by interacting with the environment, using both positive and negative rewards to refine its policy.

## Dependencies
The project requires several Python packages. You can install all dependencies by using the `requirements.txt` file provided.

```bash
pip install -r requirements.txt
