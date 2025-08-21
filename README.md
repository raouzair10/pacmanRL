# Pacman Reinforcement Learning Project

This project implements a reinforcement learning system for the classic Atari Pacman game using the ALE (Atari Learning Environment) and Stable-Baselines3. The codebase includes both human play experiments and AI agent training/evaluation capabilities.

## Features

- **Human Play Mode**: Interactive Pacman gameplay with keyboard controls
- **Agent Play Mode**: AI agent gameplay with human advice integration
- **PPO Training**: Proximal Policy Optimization training for Pacman
- **Model Evaluation**: Comprehensive evaluation of trained agents

## Prerequisites

- Python 3.7 or higher
- Windows/Linux/macOS
- Display (for game rendering)
- Optional: NVIDIA GPU with CUDA support for faster training

## Setup Instructions

### 1. Create Virtual Environment

First, create a virtual environment to isolate the project dependencies:

```bash
# Create virtual environment
python3 -m venv pacman_env

# Activate virtual environment
# On Windows:
pacman_env\Scripts\activate

# On macOS/Linux:
source pacman_env/bin/activate
```

### 2. Install Dependencies

Install the required packages using the provided requirements.txt:

```bash
# Install all dependencies
pip install -r requirements.txt
```

## Running the Project

The project has two main categories of functionality: **Experiments** and **Training/Evaluation**.

### Experiments

#### Human Play Mode

Run the human play experiment where you can play Pacman yourself:

```bash
python3 human_play.py
```

**Features:**
- Interactive gameplay with arrow key controls
- Configurable time limit (default: 10 minutes)
- Score tracking
- Pause/resume functionality (Press 'P')
- Unlimited lives mode

**Controls:**
- **Arrow Keys**: Move Pacman (UP, RIGHT, DOWN, LEFT)
- **P**: Pause/Resume game
- **ESC**: Exit (when paused)

#### Agent Play Mode

Watch an AI agent play Pacman with human advice:

```bash
python3 agent_play.py
```

**Features:**
- AI agent gameplay using a trained PPO model
- Two advice modes: Freeze (wait indefinitely) and Countdown (time-limited)
- Human advice integration every few steps
- Mode switching at halfway point
- Performance statistics and analysis

**Requirements:**
- Must have a trained model file (`ppo_pacman.zip`) in the project directory
- If no model exists, you'll need to train one first (see Training section)

### Training and Evaluation

#### Quick Start (Recommended)

Use the main script for a complete training and evaluation pipeline:

```bash
python3 main.py
```

This script will:
1. Check if a trained model exists
2. Train a new PPO agent if no model is found (10M timesteps)
3. Evaluate the trained agent over 100 episodes
4. Display comprehensive performance statistics

#### Manual Training

If you want more control over the training process:

```bash
# Train only
python3 train_agent.py

# Evaluate only (requires existing model)
python3 train_agent.py
```

## Project Structure

```
pacman/
├── main.py                 # Main training and evaluation script
├── train_agent.py          # PPO training implementation
├── human_play.py           # Human play experiment
├── agent_play.py           # AI agent play with human advice
├── requirements.txt        # Python dependencies
├── ppo_pacman.zip         # Trained model (generated after training)
└── README.md              # This file
```

## Environment Details

### Action Space
- **5 discrete actions**: NOOP, UP, RIGHT, DOWN, LEFT
- Compatible with ALE Pacman's reduced action space

### Observation Space
- **RGB images**: 210x160x3 pixel observations
- Preprocessed with Atari wrappers for optimal training
