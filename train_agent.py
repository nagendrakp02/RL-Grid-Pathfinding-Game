# train_agent.py
import numpy as np
from agent import QLearningAgent
from grid_environment import GridEnvironment

# Config
GRID_SIZE = 5
START = (0, 0)
GOAL = (4, 4)
OBSTACLES = [(1, 1), (2, 2)]

def train_agent(episodes=500):
    env = GridEnvironment(GRID_SIZE, START, GOAL, OBSTACLES)
    agent = QLearningAgent(GRID_SIZE)

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            agent.learn(state, action, reward, next_state)

            state = next_state
            total_reward += reward

        # Debug print (optional)
        # print(f"Episode {ep+1}: Total Reward = {total_reward}")

    return agent
