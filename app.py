import streamlit as st
from train_agent import train_agent
from grid_environment import GridEnvironment

# Config
GRID_SIZE = 5
START = (0, 0)
GOAL = (4, 4)
OBSTACLES = [(1, 1), (2, 2)]

# Train RL agent once
@st.cache_resource
def get_agent_and_env():
    agent = train_agent()
    env = GridEnvironment(GRID_SIZE, START, GOAL, OBSTACLES)
    return agent, env

agent, env = get_agent_and_env()

# Reset function
def reset():
    st.session_state.agent_pos = START
    st.session_state.total_reward = 0
    st.session_state.game_over = False
    st.session_state.last_reward = 0

# Initialize session state
if "agent_pos" not in st.session_state:
    reset()

# UI Title
st.title("🧠 RL Grid Pathfinding Game")

# Grid Display
def render_grid():
    grid = [["⬜"] * GRID_SIZE for _ in range(GRID_SIZE)]
    for ox, oy in OBSTACLES:
        grid[ox][oy] = "⬛"
    ax, ay = st.session_state.agent_pos
    gx, gy = GOAL
    grid[gx][gy] = "🏁"
    grid[ax][ay] = "🤖"
    return grid

st.subheader("🗺️ Grid:")
grid = render_grid()
for row in grid:
    st.write(" ".join(row))

# Buttons
st.subheader("🎮 Move your agent:")

action = None  # Track selected action
col_up = st.columns([1, 1, 1])
with col_up[1]:
    if st.button("⬆️ Up"):
        action = 0

col_middle = st.columns([1, 1, 1])
with col_middle[0]:
    if st.button("⬅️ Left"):
        action = 2
with col_middle[2]:
    if st.button("➡️ Right"):
        action = 3

col_down = st.columns([1, 1, 1])
with col_down[1]:
    if st.button("⬇️ Down"):
        action = 1

st.markdown("<br>", unsafe_allow_html=True)
col_reset = st.columns([2, 1, 2])
with col_reset[1]:
    st.button("🔁 Reset Game", on_click=reset)

# Move agent if an action was selected
if action is not None and not st.session_state.game_over:
    next_pos, reward, done = env.step(action)
    st.session_state.agent_pos = next_pos
    st.session_state.total_reward += reward
    st.session_state.game_over = done
    st.session_state.last_reward = reward

# Game Feedback
st.subheader("📊 Game Info:")
st.write(f"📍 Current Position: `{st.session_state.agent_pos}`")
st.write(f"💰 Last Reward: `{st.session_state.last_reward}`")
st.write(f"🔁 Total Reward: `{st.session_state.total_reward}`")

if st.session_state.game_over:
    st.success("🎉 Goal Reached! You win!")
