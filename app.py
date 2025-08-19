import streamlit as st
from agent import QLearningAgent
from grid_environment import GridEnvironment

# Config
GRID_SIZE = 5
START = (0, 0)
GOAL = (4, 4)
OBSTACLES = [(1, 1), (2, 2)]

# Initialize Q-learning agent and environment once
if 'agent' not in st.session_state:
    st.session_state.agent = QLearningAgent(GRID_SIZE)
if 'env' not in st.session_state:
    st.session_state.env = GridEnvironment(GRID_SIZE, START, GOAL, OBSTACLES)

# Reset function for session state
def reset():
    st.session_state.agent_pos = START
    st.session_state.total_reward = 0
    st.session_state.game_over = False
    st.session_state.path = [START]

if 'agent_pos' not in st.session_state:
    reset()

st.title("🧠 RL Grid Pathfinding Game (Step-by-Step)")

# Function to render the grid with path marked
def render_grid():
    grid = [["⬜"] * GRID_SIZE for _ in range(GRID_SIZE)]
    for ox, oy in OBSTACLES:
        grid[ox][oy] = "⬛"
    gx, gy = GOAL
    grid[gx][gy] = "🏁"
    # Mark the path, except current position
    for (x, y) in st.session_state.path[:-1]:
        grid[x][y] = "•"
    # Mark current agent position
    ax, ay = st.session_state.agent_pos
    grid[ax][ay] = "🤖"
    return grid

# Control buttons
col1, col2, _ = st.columns([1,1,6])
with col1:
    st.button("🔁 Reset", on_click=reset)
with col2:
    step_pressed = st.button("➡️ Move Step")

st.subheader("🗺️ Grid:")
grid = render_grid()
for row in grid:
    st.write(" ".join(row))

# Step action logic
if step_pressed and not st.session_state.game_over:
    env = st.session_state.env
    agent = st.session_state.agent
    state = st.session_state.agent_pos

    action = agent.choose_action(state)
    next_state, reward, done = env.step(action)
    agent.learn(state, action, reward, next_state)

    st.session_state.agent_pos = next_state
    st.session_state.path.append(next_state)
    st.session_state.total_reward += reward
    st.session_state.game_over = done

st.subheader("📊 Game Info:")
st.write(f"📍 Current Position: `{st.session_state.agent_pos}`")
st.write(f"🛤️ Path Taken: `{st.session_state.path}`")
st.write(f"💰 Total Reward: `{st.session_state.total_reward}`")

if st.session_state.game_over:
    st.success("🎉 Goal Reached! The agent has learned an optimal path!")
