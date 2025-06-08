# Pong

This project investigates deep learning techniques to develop an intelligent agent capable of mastering the classic Atari game [Pong](https://en.wikipedia.org/wiki/Pong).

A comprehensive project report is available [here](./project.ipynb).

Detailed development notebooks can be found in the `./notebooks` directory, with more extensive code implementations and scripts located in the `./src` directory.

## Development Setup

```sh
python -m venv venv
source ./venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Scripts

### Random Agent

To view an agent playing Pong that chooses a random action every frame run:

```sh
python src/random_agent.py
```

### Trained DQN Model

To view the best performing model from this project play Pong, run:

```sh
python src/dqn_agent.py
```
