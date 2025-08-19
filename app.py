import streamlit as st
import numpy as np
from agent import QLearningAgent
from grid_environment import GridEnvironment
from train_agent import train_agent

# Configuration
GRID_SIZE = 5
START = (0, 0)
GOAL = (4, 4)
OBSTACLES = [(1, 1), (2, 2)]

# Train agent once
if 'agent' not in st.session_state:
    st.session_state.agent = train_agent(episodes=500)

# Reset function
def reset():
    st.session_state.env = GridEnvironment(GRID_SIZE, START, GOAL, OBSTACLES)
    st.session_state.agent_pos = START
    st.session_state.total_reward = 0
    st.session_state.path = [START]
    st.session_state.game_over = False
    st.session_state.visited = set()

# Initialize session state
if 'agent_pos' not in st.session_state:
    reset()

st.title("🧠 RL Grid Pathfinding — Step-by-Step Traversal")

# Render grid
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

# Step button only
step_pressed = st.button("➡️ Move One Step")

# Step-by-step traversal
if step_pressed and not st.session_state.game_over:
    env = st.session_state.env
    agent = st.session_state.agent
    state = st.session_state.agent_pos

    action = np.argmax(agent.q_table[state[0], state[1]])
    next_state, reward, done = env.step(action)

    if next_state in st.session_state.visited:
        st.error("⚠️ Loop detected. Training may be insufficient.")
        st.session_state.game_over = True
    else:
        st.session_state.visited.add(next_state)
        st.session_state.agent_pos = next_state
        st.session_state.path.append(next_state)
        st.session_state.total_reward += reward
        st.session_state.game_over = done

# Display grid
st.subheader("🗺️ Grid:")
grid = render_grid(st.session_state.path)
for row in grid:
    st.write(" ".join(row))

# Info
st.subheader("📊 Path Info:")
st.write(f"📍 Current Position: `{st.session_state.agent_pos}`")
st.write(f"🛤️ Path So Far: `{st.session_state.path}`")
st.write(f"💰 Total Reward: `{st.session_state.total_reward}`")

if st.session_state.game_over:
    st.success("🎯 Goal reached or traversal complete!")
