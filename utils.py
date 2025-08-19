import matplotlib.pyplot as plt

def render_grid(grid, path=None, start=None, goal=None):
    cmap = plt.cm.get_cmap("Greys")
    fig, ax = plt.subplots()
    ax.imshow(grid == -1, cmap=cmap, interpolation="nearest")

    if path:
        for (x, y) in path:
            ax.text(y, x, '•', ha='center', va='center', color='blue')

    if start:
        ax.text(start[1], start[0], 'S', ha='center', va='center', color='green')

    if goal:
        ax.text(goal[1], goal[0], 'G', ha='center', va='center', color='red')

    plt.xticks([])
    plt.yticks([])
    return fig
