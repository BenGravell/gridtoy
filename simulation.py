import numpy as np
import numpy.random as npr
from scipy.signal import convolve2d
from functools import partial


########################################################################################################################
# Option classes
########################################################################################################################
class SimulationOptions:
    """
    Options class for simulation
    :param width: Width of the simulation domain
    :type width: int
    :param height: Height of the simulation domain
    :type height: int
    :param dt: Sampling time
    :type dt: float
    :param step_method: Specification of the differential equation integration scheme, 'euler' or 'rk4'
    :type step_method: str
    """
    def __init__(self, width, height, dt, step_method='rk4'):
        self.width = width
        self.height = height
        self.n = min(self.width, self.height)
        self.dt = dt
        self.step_method = step_method


class DynamicsOptions:
    """Options class for spring-mass-damper dynamics"""
    def __init__(self, mass, damping_ratio_ground, damping_ratio_neighbor, natural_freq_ground, natural_freq_neighbor,
                 diffusion_rate):
        """
        :param mass: Scalar, valid range [0, np.inf), mass of each spring
        :type mass: float
        :param damping_ratio_ground: Scalar, valid range [0, np.inf), damping ratio of components connected to ground,
                                     1 is critical damping, <1 is underdamped, >1 is overdamped
        :type damping_ratio_ground: float
        :param damping_ratio_neighbor: Scalar, valid range [0, np.inf), damping ratio of components connected to neighbors,
                                       1 is critical damping, <1 is underdamped, >1 is overdamped
        :type damping_ratio_neighbor: float
        :param natural_freq_ground: Scalar, valid range [0, np.inf), natural frequency of components connected to ground
        :type natural_freq_ground: float
        :param natural_freq_neighbor: Scalar, valid range [0, np.inf), natural frequency of components connected to neighbor
        :type natural_freq_neighbor: float
        :param diffusion_rate: Diffusion rate of the position state
        :type diffusion_rate: float
        """
        self.mass = mass
        self.spring_ground, self.friction_ground = spring_and_friction(self.mass, damping_ratio_ground, natural_freq_ground)
        self.spring_neighbor, self.friction_neighbor = spring_and_friction(self.mass, damping_ratio_neighbor, natural_freq_neighbor)
        self.diffusion_rate = diffusion_rate
        self.kernel_pos = ground_kernel(self.spring_ground) + neighbor_kernel(self.spring_neighbor)
        self.kernel_vel = ground_kernel(self.friction_ground) + neighbor_kernel(self.friction_neighbor)
        self.kernel_dif = neighbor_kernel(-self.diffusion_rate)


class NoiseOptions:
    """Options class for noise"""
    def __init__(self, pos_amount, vel_amount, pos_inertia, vel_inertia):
        """
        :param pos_amount: Scalar, valid range [0, np.inf), position noise intensity
        :type pos_amount: float
        :param vel_amount: Scalar, valid range [0, np.inf), velocity noise intensity
        :type vel_amount: float
        :param pos_inertia: Scalar, valid range [0, 1], position noise inertia, 0 is fully white noise, 1 is constant signal
        :type pos_inertia: float
        :param vel_inertia: Scalar, valid range [0, 1], velocity noise inertia, 0 is fully white noise, 1 is constant signal
        :type vel_inertia: float
        """
        self.pos_amount = pos_amount
        self.vel_amount = vel_amount
        self.pos_inertia = pos_inertia
        self.vel_inertia = vel_inertia


class ForceOptions:
    """Options class for force"""
    def __init__(self, n, pos_amount, vel_amount, randomness, kernel_radius, orbit_radius, orbit_freq, pulse_freq, dwell_squash, dwell_narrow):
        """
        :param pos_amount: Intensity of the force applied to the position time derivative, valid range [0, np.inf)
        :param vel_amount: Intensity of the force applied to the velocity time derivative, valid range [0, np.inf)
        :param randomness: Amount of randomness in the position and amplitude of the force, valid range [0, 1]
        :param kernel_radius: Radius of the force bump, expressed in domain half-widths (n/2), valid range [0, 1]
        :param orbit_radius: Radius of the force orbit trajectory, expressed in domain half-widths (n/2), valid range [0, 1]
        :param orbit_freq: Frequency of the orbit, expressed in cycles/tick
        :param pulse_freq: Frequency of the amplitude pulse profile, expressed as a multiple of orbit_freq/2.
                           Choose as 1/int for best results.
        :param dwell_squash: Amount of nonlinear squash applied to pulse profile, valid range [0, np.inf),
                             larger values make the profile closer to a piecewise constant / step profile
        :param dwell_narrow: Narrowness of the plateau peaks of the amplitude profile, valid range [0, 1]
        """
        self.pos_amount = pos_amount
        self.vel_amount = vel_amount
        self.randomness = randomness
        self.kernel_radius = kernel_radius
        self.orbit_radius = orbit_radius
        self.kernel_radius_px = int((n/2)*kernel_radius)  # Express kernel radius in pixels
        self.orbit_radius_px = int((n/2)*orbit_radius)  # Express orbit radius in pixels
        self.orbit_freq = orbit_freq
        self.pulse_freq = (orbit_freq/2)*pulse_freq
        self.dwell_squash = dwell_squash
        self.dwell_narrow = dwell_narrow
        self.kernel = bump_kernel(self.kernel_radius_px)


class PhysicsOptions:
    def __init__(self, simulation_options, dynamics_options, force_options, noise_options):
        self.simulation_options = simulation_options
        self.dynamics_options = dynamics_options
        self.force_options = force_options
        self.noise_options = noise_options

    def unpack(self):
        return self.simulation_options, self.dynamics_options, self.force_options, self.noise_options


########################################################################################################################
# Functions
########################################################################################################################
def mix(x, y, ratio):
    return (1 - ratio)*x + ratio*y


def smoothstep(x):
    return -2*x**3 + 3*x**2


def dwell_profile(angle, squash=20, narrow=0.95, phase=0):
    def g(angle):
        return np.tanh(squash*np.sin(angle + np.pi*(narrow - phase)))
    return (g(angle) + g(angle - np.pi*narrow))/2


def ground_kernel(a):
    return np.array([[0, 0, 0],
                     [0, a, 0],
                     [0, 0, 0]])


def neighbor_kernel(b):
    return np.array([[ 0,  -b,  0],
                     [-b, 4*b, -b],
                     [ 0,  -b,  0]])


def bump_kernel(radius):
    s = np.linspace(-1, 1, 2*radius)
    x, y = np.meshgrid(s, s)
    d2 = np.minimum(x**2 + y**2, 1)
    return smoothstep(1 - d2)


def spring_and_friction(mass, damping_ratio, natural_freq):
    """
    Compute spring rate and friction coefficient from a given damping ratio and natural frequency
    :param mass: Mass, valid range [0, np.inf)
    :type mass: float
    :param damping_ratio: Damping ratio of components, valid range [0, np.inf),
                          1 is critical damping, <1 is underdamped, >1 is overdamped
    :type damping_ratio: float
    :param natural_freq: Natural frequency of components, valid range [0, np.inf)
    :type natural_freq: float
    :return: Spring rate and natural frequency
    :rtype: float, float
    """
    spring = mass*(natural_freq**2)
    friction = 2*mass*damping_ratio*natural_freq
    return spring, friction


def generate_force_params(t, simulation_options, force_options):
    # Set the position of the center of the force and associated slices
    orbit_angle = 2*np.pi*force_options.orbit_freq*t + np.pi
    xr = 0.2*force_options.randomness*(npr.rand() - 0.5)
    yr = 0.2*force_options.randomness*(npr.rand() - 0.5)
    shift_x = int(force_options.orbit_radius_px*(np.sin(orbit_angle) + xr))
    shift_y = int(force_options.orbit_radius_px*(np.cos(orbit_angle) + yr))
    cx = int(simulation_options.width/2)
    cy = int(simulation_options.height/2)
    sx = slice(cx - force_options.kernel_radius_px + shift_x, cx + force_options.kernel_radius_px + shift_x)
    sy = slice(cy - force_options.kernel_radius_px + shift_y, cy + force_options.kernel_radius_px + shift_y)

    # Determine the amplitude of the force
    pulse_angle = 2*np.pi*force_options.pulse_freq*t
    a = dwell_profile(pulse_angle, force_options.dwell_squash, force_options.dwell_narrow)
    # # Halve the amplitude for the first half pulse
    # a *= 0.5 if pulse_angle < np.pi else 1 + force_randomness*(2*npr.rand() - 1)
    a *= 1 - force_options.randomness*npr.rand()
    return a, sx, sy


def dxdt(state, t, simulation_options, dynamics_options, force_options, return_force=False):

    # Compute the external force, which is applied to both position and velocity
    a, sx, sy = generate_force_params(t, simulation_options, force_options)
    force_external = a*force_options.kernel


    # Position
    # Compute components
    pos_force_from_vel = state[1]
    pos_force_diffusion = convolve2d(state[0], dynamics_options.kernel_dif, mode='same', boundary='wrap')
    pos_force_external = force_options.pos_amount*force_external
    pos_force = pos_force_from_vel + pos_force_diffusion
    pos_force[sx, sy] += pos_force_external
    ddt_pos = pos_force

    # Velocity
    # Compute components
    vel_force_spring = convolve2d(state[0], dynamics_options.kernel_pos, mode='same', boundary='wrap')
    vel_force_friction = convolve2d(state[1], dynamics_options.kernel_vel, mode='same', boundary='wrap')
    vel_force_external = force_options.vel_amount*force_external
    # Sum the forces
    vel_force = -(vel_force_spring + vel_force_friction)
    vel_force[sx, sy] += vel_force_external
    # Newton's 2nd law
    ddt_vel = vel_force/dynamics_options.mass

    # Stack position and velocity time derivatives together
    ddt_posvel = np.stack([ddt_pos, ddt_vel])

    # Return the external force with peak magnitude 1 for display
    if return_force:
        force_display = np.zeros_like(vel_force)
        force_display[sx, sy] += force_external
    else:
        force_display = None
    return ddt_posvel, force_display


def step(state, t, simulation_options, dynamics_options, force_options, return_force=False):
    dt = simulation_options.dt
    step_method = simulation_options.step_method
    partial_dxdt = partial(dxdt,
                           simulation_options=simulation_options,
                           dynamics_options=dynamics_options,
                           force_options=force_options,
                           return_force=return_force)
    k1, force = partial_dxdt(state, t)
    if step_method == 'euler':
        state_new = state + k1*dt
    elif step_method == 'rk4':
        k2, _ = partial_dxdt(state + (dt/2)*k1, t)
        k3, _ = partial_dxdt(state + (dt/2)*k2, t)
        k4, _ = partial_dxdt(state + dt*k3, t)
        state_new = state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    return state_new, force


def generate_noise(noise, simulation_options, noise_options):
    width, height = simulation_options.width, simulation_options.height
    noise[0] = mix(npr.randn(width, height), noise[0], noise_options.pos_inertia)
    noise[1] = mix(npr.randn(width, height), noise[1], noise_options.vel_inertia)
    return noise


def physics_update(state, noise, count, physics_options, return_force=False):
    simulation_options, dynamics_options, force_options, noise_options = physics_options.unpack()
    t = count*simulation_options.dt
    state, force = step(state, t, simulation_options, dynamics_options, force_options, return_force)
    noise = generate_noise(noise, simulation_options, noise_options)
    state[0] += noise_options.pos_amount*noise[0]
    state[1] += noise_options.vel_amount*noise[1]
    return state, noise, force
