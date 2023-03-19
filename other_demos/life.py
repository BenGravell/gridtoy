import numpy as np
import numpy.random as npr
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib.animation as animation

n = 200
state = npr.rand(n, n) > 0.90
neighbor_kernel = np.array([[1, 1, 1],
                            [1, 0, 1],
                            [1, 1, 1]], dtype=int)
im = plt.imshow(state, interpolation=None, aspect='equal', cmap='Greys', vmin=0, vmax=1)


def update(frame):
    global state
    # Apply Conway's Game of Life rules to update the state
    neighbors = convolve2d(state, neighbor_kernel, mode='same', boundary='wrap')
    survivors = np.logical_and(np.logical_and(neighbors <= 3, neighbors >= 2), state)
    births = np.logical_and(neighbors == 3, np.logical_not(state))
    state = np.logical_or(survivors, births)
    im.set_array(state)
    return [im]


anim = animation.FuncAnimation(plt.gcf(), update, interval=20)
plt.show()
