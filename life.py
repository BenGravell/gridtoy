import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import matplotlib.animation as animation

n = 400
state = np.zeros([n, n])
m = int(0.25*n)
state[m:3*m, m:3*m] = npr.randint(2, size=(2*m, 2*m))

# G = np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]])  # This is a Gosper glider
# A[3:6, 3:6] = G

start_idx_pair_list = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]

fig = plt.figure(figsize=(8, 8))
im = plt.imshow(state, interpolation='none', aspect='auto', vmin=0, vmax=1, cmap='Greys')


def update(frame):
    global state
    # Compute the neighbor counts
    Ap = np.zeros([n+2, n+2])
    Ap[1:-1, 1:-1] = state
    neighbors = np.zeros([n, n])
    for start_idx_pair in start_idx_pair_list:
        i = start_idx_pair[0]
        j = start_idx_pair[1]
        neighbors += Ap[i:i+n, j:j+n]
    # Apply Conway's Game of Life rules to update the state
    survivors = np.logical_and(np.logical_and(neighbors <= 3, neighbors >= 2), state)
    births = np.logical_and(neighbors == 3, np.logical_not(state))
    state = np.logical_or(survivors, births)
    im.set_array(state)
    return [im]


fps = 30
anim = animation.FuncAnimation(fig, update, interval=1000/fps)