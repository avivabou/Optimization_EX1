import numpy as np
import matplotlib.pyplot as plt
import os

SAVE_DIR = './saves/'

def save_plot(title):
    SAVE_PATH = os.path.join(SAVE_DIR, title)

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    plt.savefig(SAVE_PATH)

def plot_contour_lines(f, title, paths, x_lim, y_lim):
    title = title + " Contour Plot"
    X = np.linspace(x_lim[0], x_lim[1])
    Y = np.linspace(y_lim[0], y_lim[1])
    X, Y = np.meshgrid(X,Y)
    Z = np.vectorize(lambda x, y: f(np.array([x, y])))(X, Y)

    fig, ax = plt.subplots(1, 1)
    contour = ax.contourf(X, Y, Z, 50)
    fig.colorbar(contour)
    ax.set_title(title)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    for path, color in zip(paths, ['red', 'blue', 'green', 'orange']):
        xs, ys = zip(*path[0])
        dx = np.diff(xs)
        dy = np.diff(ys)
        ax.quiver(xs[:-1], ys[:-1], dx, dy, scale_units='xy', angles='xy', scale=1, label=path[1], color=color)
            
    
    plt.legend()
    save_plot(title)
    plt.show()


def plot_function_values(paths, title,):
    title = title + " Minimization Iterations"
    fig, ax = plt.subplots(1, 1)
    for path, color in zip(paths, ['red', 'blue', 'green', 'orange']):
        x = np.linspace(0, len(path[0])-1, len(path[0]))
        ax.plot(x, path[0], marker='.', label=path[1], color=color)
    ax.set_title(title)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Function values')
    plt.legend()
    save_plot(title)
    plt.show()
