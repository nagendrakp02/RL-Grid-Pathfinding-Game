# import numpy as np
# import random

# class QLearningAgent:
#     def __init__(self, grid_size, learning_rate=0.1, discount_factor=0.9, epsilon=0.2):
#         self.grid_size = grid_size
#         self.q_table = np.zeros((grid_size, grid_size, 4))  # up, down, left, right
#         self.lr = learning_rate
#         self.gamma = discount_factor
#         self.epsilon = epsilon

#     def choose_action(self, state):
#         """Epsilon-greedy policy"""
#         if random.uniform(0, 1) < self.epsilon:
#             return random.randint(0, 3)  # Explore
#         return np.argmax(self.q_table[state[0], state[1]])  # Exploit

#     def learn(self, state, action, reward, next_state):
#         """Q-learning update"""
#         current_q = self.q_table[state[0], state[1], action]
#         max_future_q = np.max(self.q_table[next_state[0], next_state[1]])
#         new_q = (1 - self.lr) * current_q + self.lr * (reward + self.gamma * max_future_q)
#         self.q_table[state[0], state[1], action] = new_q


import numpy as np
import random

class QLearningAgent:
    def __init__(self, grid_size, learning_rate=0.1, discount_factor=0.9, epsilon=0.2):
        self.grid_size = grid_size
        self.q_table = np.zeros((grid_size, grid_size, 4))  # actions: up, down, left, right
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:   # exploration
            return random.randint(0, 3)
        return np.argmax(self.q_table[state[0], state[1]])  # exploitation

    def learn(self, state, action, reward, next_state):
        current_q = self.q_table[state[0], state[1], action]
        max_future_q = np.max(self.q_table[next_state[0], next_state[1]])
        new_q = (1 - self.lr) * current_q + self.lr * (reward + self.gamma * max_future_q)
        self.q_table[state[0], state[1], action] = new_q
