import numpy as np
import numpy.random as npr
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


########################################################################################################################
# Settings
########################################################################################################################

# Dimension of simulation domain
n = 100

# Sampling time
dt = 0.1

# Initialize the state and noise
state = npr.rand(n, n)
noise = np.zeros([n, n])

# Define the diffusion rate - ensure rate*dt < 0.25 for numerical stability using Euler integration
rate = 2.0

# Define the force and noise amount
force_amount = 0.005
noise_amount = 0.040

# Define the force frequency
force_freq = 0.001

# Define the noise inertia (between 0 and 1, 0 is fully white noise, 1 is a constant signal)
noise_inertia = 0.9


########################################################################################################################
# Simulation
########################################################################################################################

# Compute the convolution kernel for diffusion dynamics
diffusion_kernel = np.array([[   0,    rate,    0],
                             [rate, -4*rate, rate],
                             [   0,    rate,    0]])
# Compute the force kernel
s = np.linspace(-1, 1, n)
x, y = np.meshgrid(s, s)
force_kernel = x**2 + y**2 < 0.2


def physics_update(state, noise, t):
    # Linear diffusion dynamics using Euler integration
    state = state + dt*convolve2d(state, diffusion_kernel, mode='same', boundary='wrap')

    # Periodic forcing
    amplitude = np.sin(force_freq*2*np.pi*t)**21
    force = amplitude*force_kernel
    state += force_amount*force

    # Random time-varying Gaussian colored noise
    noise = (1-noise_inertia)*npr.randn(*noise.shape) + noise_inertia*noise
    state += noise_amount*noise

    return state, noise


########################################################################################################################
# Plotting
########################################################################################################################

# Initialize the plot
fig, ax = plt.subplots()
im = plt.imshow(state, vmin=0, vmax=1)
ax.axis('off')
fig.tight_layout()


def update(t):
    global state, noise
    state, noise = physics_update(state, noise, t)
    im.set_data(state)
    return [im]


# Create the animation
animation = FuncAnimation(fig, update, interval=1, blit=True)
plt.show()