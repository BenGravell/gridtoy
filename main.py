import os
import sys
import numpy as np
from pyqtgraph.Qt import QtGui

from simulation import SimulationOptions, DynamicsOptions, NoiseOptions, ForceOptions, PhysicsOptions
from display import App, DisplayOptions, PlotOption


def demo_physics_settings(demo_str='diffusion_and_spring'):
    # Simulation settings
    simulation_options = SimulationOptions(width=100,
                                           height=100,
                                           dt=0.01,
                                           step_method='euler')
    if demo_str == 'diffusion':
        # Settings for diffusion dynamics only
        dynamics_options = DynamicsOptions(mass=1,
                                           damping_ratio_ground=0,
                                           damping_ratio_neighbor=0,
                                           natural_freq_ground=0,
                                           natural_freq_neighbor=0,
                                           diffusion_rate=10)
        force_options = ForceOptions(n=simulation_options.n,
                                     pos_amount=15,
                                     vel_amount=0,
                                     randomness=0.5,
                                     kernel_radius=0.2,
                                     orbit_radius=0.5,
                                     orbit_freq=1.0,
                                     pulse_freq=1/12,
                                     dwell_squash=10,
                                     dwell_narrow=0.90)
        noise_options = NoiseOptions(pos_amount=0.02,
                                     vel_amount=0,
                                     pos_inertia=0.9,
                                     vel_inertia=0.9)
    elif demo_str == 'spring':
        # Settings for mass-spring-damper dynamics only
        dynamics_options = DynamicsOptions(mass=1,
                                           damping_ratio_ground=0.02,
                                           damping_ratio_neighbor=0.10,
                                           natural_freq_ground=2,
                                           natural_freq_neighbor=6,
                                           diffusion_rate=0)
        force_options = ForceOptions(n=simulation_options.n,
                                     pos_amount=0,
                                     vel_amount=30,
                                     randomness=0.5,
                                     kernel_radius=0.2,
                                     orbit_radius=0.5,
                                     orbit_freq=1.0,
                                     pulse_freq=1/12,
                                     dwell_squash=10,
                                     dwell_narrow=0.90)
        noise_options = NoiseOptions(pos_amount=0.002,
                                     vel_amount=0.02,
                                     pos_inertia=0.9,
                                     vel_inertia=0.9)
    elif demo_str == 'diffusion_and_spring':
        # Settings for combined diffusion and spring-mass-damper dynamics
        dynamics_options = DynamicsOptions(mass=1,
                                           damping_ratio_ground=0.01,
                                           damping_ratio_neighbor=0.10,
                                           natural_freq_ground=2,
                                           natural_freq_neighbor=6,
                                           diffusion_rate=2)
        force_options = ForceOptions(n=simulation_options.n,
                                     pos_amount=10,
                                     vel_amount=20,
                                     randomness=0.5,
                                     kernel_radius=0.2,
                                     orbit_radius=0.5,
                                     orbit_freq=1.0,
                                     pulse_freq=1/12,
                                     dwell_squash=10,
                                     dwell_narrow=0.90)
        noise_options = NoiseOptions(pos_amount=0.01,
                                     vel_amount=0.03,
                                     pos_inertia=0.9,
                                     vel_inertia=0.9)
    else:
        raise Exception('Invalid demo physics setting string')
    # Initialize state and noise
    state = np.stack([np.zeros([simulation_options.width, simulation_options.height]),
                      np.zeros([simulation_options.width, simulation_options.height])])
    noise = np.stack([np.zeros([simulation_options.width, simulation_options.height]),
                      np.zeros([simulation_options.width, simulation_options.height])])
    return PhysicsOptions(simulation_options, dynamics_options, force_options, noise_options), state, noise


def demo_visual_settings(demo_str='basic'):
    if demo_str == 'basic':
        display_options = DisplayOptions()
        plot_options = [PlotOption()]
    elif demo_str == 'advanced':
        output_directory = os.path.join('frames')
        app = QtGui.QApplication(sys.argv)
        screen_rect = app.desktop().screenGeometry()
        screen_width, screen_height = screen_rect.width(), screen_rect.height()
        display_options = DisplayOptions(window_size=(int(0.5*screen_width), int(0.5*screen_height)),
                                         show_maximized=True,
                                         fps_inertia=0.9,
                                         max_fps_report=1000)
        plot_options = [PlotOption(quantity='position_state', show_histogram=True, colormap_str='viridis',
                                   export=False, format_str='posstate_frame_%07d.png', directory=output_directory),
                        PlotOption(quantity='position_noise', show_histogram=True, colormap_str='grey',
                                   export=False, format_str='posnoise_frame_%07d.png', directory=output_directory),
                        PlotOption(quantity='velocity_state', show_histogram=True, colormap_str='viridis',
                                   export=False, format_str='velstate_frame_%07d.png', directory=output_directory),
                        PlotOption(quantity='velocity_noise', show_histogram=True, colormap_str='grey',
                                   export=False, format_str='velnoise_frame_%07d.png', directory=output_directory),
                        PlotOption(quantity='force', show_histogram=True, colormap_str='inferno',
                                   export=False, format_str='force_frame_%07d.png', directory=output_directory)]
    else:
        raise Exception('Invalid demo visual setting string')
    return display_options, plot_options


if __name__ == '__main__':
    # Physics settings
    physics_options, state, noise = demo_physics_settings('diffusion_and_spring')

    # Display settings
    display_options, plot_options = demo_visual_settings('basic')
    plot_options[0].export = True
    plot_options[0].directory = os.path.join('frames')
    plot_options[0].format_str = 'frame_%07d.png'
    plot_options[0].scale = 4

    # Start the display app
    app = QtGui.QApplication(sys.argv)
    thisapp = App(display_options, plot_options, physics_options, state, noise)
    thisapp.show()
    sys.exit(app.exec_())
