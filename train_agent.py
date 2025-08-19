from rl_agent import QLearningAgent
from grid_environment import GridEnvironment

def train_agent():
    size = 5
    env = GridEnvironment(size, (0, 0), (4, 4), [(1, 1), (2, 2)])
    agent = QLearningAgent(size)

    for episode in range(1000):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state

    return agent
