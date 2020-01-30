import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation


"""

Plot the changes of path construction using matplotlib animation

history - the process of changing the path, format [(nodes_indices, destination_value), ...]
points - the coordinates of nodes 
names - the names of the cities, correspond to coordinates in points

"""
def animation(history, points, names):

    fig, ax = plt.subplots()


    line, = plt.plot([], [], lw=2)

    def init():

        x = [points[i][0] for i in history[0][0]]
        y = [points[i][1] for i in history[0][0]]


        x_init = [points[i][0] for i in range(len(points))]
        y_init = [points[i][1] for i in range(len(points))]

        # plot the dots with cities names
        ax.scatter(x_init, y_init)
        for i, txt in enumerate(names):
            ax.annotate(txt, (x_init[i], y_init[i]))


        ax.set_xlim(min(x) - 5, max(x) + 5)
        ax.set_ylim(min(y) - 5, max(y) + 5)


        line.set_data(x, y)
        return line,

    def update(frame):

        # draw the path update every iteration
        x = [points[i][0] for i in history[frame][0] + [history[frame][0][0]]]
        y = [points[i][1] for i in history[frame][0] + [history[frame][0][0]]]

        line.set_data(x, y)
        return line,


    ani = FuncAnimation(fig, update, frames=range(0, len(history)-1, 1),
                        init_func=init, interval=3, repeat=False)

    plt.show()

def plot_change(history):

    y = [history[i][1] for i in range(len(history))]
    x = [i for i in range(1, len(history)+1)]

    plt.plot(x, y)

    plt.xlabel('Iterations', fontsize=18)
    plt.ylabel('Distance', fontsize=16)
    plt.show()

