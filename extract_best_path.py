# from agent import QLearningAgent
# from grid_environment import GridEnvironment
# import numpy as np

# GRID_SIZE = 5
# START = (0, 0)
# GOAL = (4, 4)
# OBSTACLES = [(1, 1), (2, 2)]

# def extract_best_path(agent):
#     env = GridEnvironment(GRID_SIZE, START, GOAL, OBSTACLES)
#     state = env.reset()
#     path = [state]
#     total_reward = 0
#     visited = set()

#     while state != GOAL:
#         action = np.argmax(agent.q_table[state[0], state[1]])
#         next_state, reward, done = env.step(action)

#         if next_state in visited:
#             print("⚠️ Loop detected. Breaking out.")
#             break
#         visited.add(next_state)

#         path.append(next_state)
#         total_reward += reward
#         state = next_state

#     print("✅ Optimal Path:", path)
#     print("💰 Total Reward:", total_reward)

# if __name__ == "__main__":
#     from train_agent import train_agent
#     trained_agent = train_agent(episodes=500)
#     extract_best_path(trained_agent)


from agent import QLearningAgent
from grid_environment import GridEnvironment
import numpy as np

# Config
GRID_SIZE = 5
START = (0, 0)
GOAL = (4, 4)
OBSTACLES = [(1, 1), (2, 2)]

def run_best_path(agent):
    env = GridEnvironment(GRID_SIZE, START, GOAL, OBSTACLES)
    state = env.reset()
    path = [state]
    total_reward = 0
    visited = set()

    while state != GOAL:
        action = np.argmax(agent.q_table[state[0], state[1]])
        next_state, reward, done = env.step(action)

        if next_state in visited:
            print("⚠️ Loop detected. Breaking out.")
            break
        visited.add(next_state)

        path.append(next_state)
        total_reward += reward
        state = next_state

    print("\n✅ Best Path to Goal:")
    print(" → ".join([str(p) for p in path]))
    print(f"💰 Total Reward: {total_reward}")

if __name__ == "__main__":
    from train_agent import train_agent
    trained_agent = train_agent(episodes=500)
    run_best_path(trained_agent)
