import numpy as np
from agent import QLearningAgent
from environment import GridEnvironment

GRID_SIZE = 5
START = (0, 0)
GOAL = (4, 4)
OBSTACLES = [(1, 1), (2, 2)]

def train_agent(episodes=1000, max_steps=100):
    env = GridEnvironment(GRID_SIZE, START, GOAL, OBSTACLES)
    agent = QLearningAgent(GRID_SIZE)

    for _ in range(episodes):
        state = env.reset()
        done = False

        for _ in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state

            if done:
                break

    return agent

if __name__ == "__main__":
    trained_agent = train_agent()
    print("✅ Training complete!")
