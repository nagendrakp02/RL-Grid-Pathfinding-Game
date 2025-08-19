# import streamlit as st
# import numpy as np
# from environment import GridEnvironment
# from train import train_agent, GRID_SIZE, START, GOAL, OBSTACLES

# # Train agent once
# if 'agent' not in st.session_state:
#     st.session_state.agent = train_agent(episodes=1000)

# def reset():
#     st.session_state.env = GridEnvironment(GRID_SIZE, START, GOAL, OBSTACLES)
#     st.session_state.agent_pos = START
#     st.session_state.total_reward = 0
#     st.session_state.path = [START]
#     st.session_state.done = False

# if 'agent_pos' not in st.session_state:
#     reset()

# st.title("🤖 Reinforcement Learning - Grid Pathfinding")

# # Render grid
# def render_grid(path):
#     grid = [["⬜"] * GRID_SIZE for _ in range(GRID_SIZE)]
#     for ox, oy in OBSTACLES:
#         grid[ox][oy] = "⬛"
#     gx, gy = GOAL
#     grid[gx][gy] = "🏁"
#     for (x, y) in path[:-1]:
#         grid[x][y] = "•"
#     ax, ay = path[-1]
#     grid[ax][ay] = "🤖"
#     return grid

# # Step button
# if st.button("➡️ Next Step") and not st.session_state.done:
#     env = st.session_state.env
#     agent = st.session_state.agent
#     state = st.session_state.agent_pos

#     action = np.argmax(agent.q_table[state[0], state[1]])  # Best move
#     next_state, reward, done = env.step(action)

#     st.session_state.agent_pos = next_state
#     st.session_state.path.append(next_state)
#     st.session_state.total_reward += reward
#     st.session_state.done = done

# # Display grid
# st.subheader("🗺️ Grid:")
# grid = render_grid(st.session_state.path)
# for row in grid:
#     st.write(" ".join(row))

# # Info
# st.subheader("📊 Stats")
# st.write(f"📍 Current Position: `{st.session_state.agent_pos}`")
# st.write(f"🛤️ Path So Far: `{st.session_state.path}`")
# st.write(f"💰 Total Reward: `{st.session_state.total_reward}`")

# if st.session_state.done:
#     st.success("🎯 Goal reached!")


import streamlit as st
import numpy as np
from agent import QLearningAgent
from environment import GridEnvironment

# Configuration
GRID_SIZE = 5
START = (0, 0)
GOAL = (4, 4)
OBSTACLES = [(1, 1), (2, 2)]

# Initialize env + agent once
if 'env' not in st.session_state:
    st.session_state.env = GridEnvironment(GRID_SIZE, START, GOAL, OBSTACLES)
    st.session_state.agent = QLearningAgent(GRID_SIZE)
    st.session_state.state = START
    st.session_state.total_reward = 0
    st.session_state.path = [START]
    st.session_state.game_over = False
    st.session_state.episodes = 0

st.title("🧠 Reinforcement Learning - Grid Pathfinding")

# Reset button
if st.button("🔄 Reset Environment"):
    st.session_state.env.reset()
    st.session_state.state = START
    st.session_state.total_reward = 0
    st.session_state.path = [START]
    st.session_state.game_over = False
    st.session_state.episodes += 1

# Step button
if st.button("➡️ Take One Step") and not st.session_state.game_over:
    agent = st.session_state.agent
    env = st.session_state.env
    state = st.session_state.state

    # Agent chooses action (explore/exploit)
    action = agent.choose_action(state)
    next_state, reward, done = env.step(action)

    # Learn from experience
    agent.learn(state, action, reward, next_state)

    # Update session state
    st.session_state.state = next_state
    st.session_state.total_reward += reward
    st.session_state.path.append(next_state)
    st.session_state.game_over = done

# Grid rendering
def render_grid(path):
    grid = [["⬜"] * GRID_SIZE for _ in range(GRID_SIZE)]
    for ox, oy in OBSTACLES:
        grid[ox][oy] = "⬛"
    gx, gy = GOAL
    grid[gx][gy] = "🏁"
    for (x, y) in path[:-1]:
        grid[x][y] = "•"
    ax, ay = path[-1]
    grid[ax][ay] = "🤖"
    return grid

st.subheader("🗺️ Grid World")
grid = render_grid(st.session_state.path)
for row in grid:
    st.write(" ".join(row))

# Info
st.subheader("📊 Stats")
st.write(f"Episode: `{st.session_state.episodes}`")
st.write(f"Current Position: `{st.session_state.state}`")
st.write(f"Total Reward: `{st.session_state.total_reward}`")
st.write(f"Path: `{st.session_state.path}`")

if st.session_state.game_over:
    st.success("🎯 Goal reached or episode finished!")
