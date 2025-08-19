# class GridEnvironment:
#     def __init__(self, size, start, goal, obstacles):
#         self.size = size
#         self.start = start
#         self.goal = goal
#         self.obstacles = obstacles
#         self.reset()

#     def reset(self):
#         self.agent_pos = self.start
#         return self.agent_pos

#     def step(self, action):
#         x, y = self.agent_pos
#         if action == 0: x = max(x - 1, 0)               # up
#         elif action == 1: x = min(x + 1, self.size - 1) # down
#         elif action == 2: y = max(y - 1, 0)             # left
#         elif action == 3: y = min(y + 1, self.size - 1) # right

#         next_pos = (x, y)

#         # Penalties and rewards
#         if next_pos in self.obstacles:
#             return self.agent_pos, -10, False  # Hit obstacle
#         if next_pos == self.goal:
#             return next_pos, 20, True          # Goal reached

#         self.agent_pos = next_pos
#         return next_pos, -1, False             # Normal step (small penalty)


class GridEnvironment:
    def __init__(self, size, start, goal, obstacles):
        self.size = size
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.reset()

    def reset(self):
        self.agent_pos = self.start
        return self.agent_pos

    def step(self, action):
        x, y = self.agent_pos
        if action == 0: x = max(x - 1, 0)        # up
        elif action == 1: x = min(x + 1, self.size - 1)  # down
        elif action == 2: y = max(y - 1, 0)      # left
        elif action == 3: y = min(y + 1, self.size - 1)  # right

        next_pos = (x, y)

        if next_pos in self.obstacles:
            return self.agent_pos, -10, False  # heavy penalty
        if next_pos == self.goal:
            return next_pos, 20, True          # big reward
        self.agent_pos = next_pos
        return next_pos, -1, False             # step penalty
