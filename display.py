import os
import time
import numpy as np
from scipy.interpolate import RectBivariateSpline
from matplotlib import cm
from PIL import Image
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg

from simulation import physics_update, mix


########################################################################################################################
# Option classes
########################################################################################################################
class PlotOption:
    """Options class for individual plots"""
    def __init__(self, quantity='position_state', center=False, scale=1, show_histogram=False, colormap_str='viridis',
                 export=False, format_str='', directory='.'):
        self.quantity = quantity
        self.center = center
        self.scale = scale
        self.show_histogram = show_histogram
        self.colormap_str = colormap_str
        self.export = export
        self.format_str = format_str
        self.directory = directory


class DisplayOptions:
    """Options class for the overall display"""
    def __init__(self, window_size=(640, 480), show_maximized=False, fps_inertia=0.9, max_fps_report=1000):
        """
        :param window_size: Window size in pixels
        :type window_size: (int, int)
        :param show_maximized: Specify whether to start display in maximized window mode or not
        :type show_maximized: bool
        :param fps_inertia: Weight given to past fps values, valid range [0, 1],
                            smaller values are more responsive, larger values are more stable
        :type fps_inertia: float
        :param max_fps_report: Maximum displayed fps. Set to the largest value you expect to see.
                               Smaller values help avoid erroneous readings due to drops in elapsed time.
        :type max_fps_report: int
        """
        self.window_size = window_size
        self.show_maximized = show_maximized
        self.fps_inertia = fps_inertia
        self.max_fps_report = max_fps_report


########################################################################################################################
# Application classes
########################################################################################################################
class App(QtGui.QMainWindow):
    """Application class which extends the QT class QMainWindow for grid dynamics simulation visualization"""
    def __init__(self, display_options, plot_options, physics_options, state, noise):
        # Make QtGui.QMainWindow a superclass of this App class
        super().__init__()

        # Resize the display window to the desired size
        self.resize(*display_options.window_size)

        # Always choose the plot sizes to be the same as the simulation domain size
        self.plot_size = (physics_options.simulation_options.width, physics_options.simulation_options.height)

        # Save the options
        self.display_options = display_options
        self.plot_options = plot_options
        self.physics_options = physics_options
        self.state, self.noise = state, noise

        # Only bother to return force data if we need to plot it
        self.return_force = any([plot_option.quantity == 'force' for plot_option in self.plot_options])

        # Create GUI elements
        self.mainbox = QtGui.QWidget()
        self.setCentralWidget(self.mainbox)
        self.mainbox.setLayout(QtGui.QVBoxLayout())

        self.canvas = pg.GraphicsLayoutWidget()
        self.mainbox.layout().addWidget(self.canvas)

        if self.display_options.show_maximized:
            self.showMaximized()

        # Add the images and histograms
        self.layouts = []
        self.views = []
        self.images = []
        self.histograms = []
        self.colormaps = []
        for i, plot_option in enumerate(self.plot_options):
            # Advance to next column and add layout
            if i > 0:
                self.canvas.nextColumn()
            layout = self.canvas.addLayout()

            # Add the title label
            title_str = plot_option.quantity.replace('_', ' ').capitalize()
            title_label = layout.addLabel(title_str)
            title_label.setText(title_str, size='20pt')
            layout.nextRow()

            # Add the image
            view = layout.addViewBox()
            view.setAspectLocked(True)
            view.setRange(QtCore.QRectF(0, 0, plot_option.scale*self.plot_size[0], plot_option.scale*self.plot_size[1]))
            image = pg.ImageItem()
            view.addItem(image)

            # Add the histogram
            if plot_option.show_histogram:
                histogram = pg.HistogramLUTItem()
                histogram.setImageItem(image)
                histogram.gradient.loadPreset(plot_option.colormap_str)
                self.canvas.addItem(histogram)
            else:
                histogram = None
                image.setLookupTable(get_lut(plot_option.colormap_str))

            # Get the matplotlib colormaps for use in image export
            if plot_option.export is not None:
                colormap_str = plot_option.colormap_str
                # Remap strings from pyqtgraph names to matplotlib names
                if colormap_str == 'grey':
                    colormap_str = 'gray'
                colormap = cm.get_cmap(colormap_str)
            else:
                colormap = None

            # Save the graphics object handles
            self.layouts.append(layout)
            self.views.append(view)
            self.images.append(image)
            self.histograms.append(histogram)
            self.colormaps.append(colormap)

        # Add the frame label object
        self.label = QtGui.QLabel()
        self.mainbox.layout().addWidget(self.label)

        # Timing initialization
        self.fps = 0.0
        self.lastupdate = time.time()
        self.counter = 0

        # Start
        self.update()

    def update(self):
        # Physics update
        self.state, self.noise, self.force = physics_update(self.state, self.noise, self.counter,
                                                            self.physics_options, self.return_force)

        for plot_option, image, colormap in zip(self.plot_options, self.images, self.colormaps):
            # Set the plot quantity
            if plot_option.quantity == 'position_state':
                image_data = np.copy(self.state[0])
            elif plot_option.quantity == 'velocity_state':
                image_data = np.copy(self.state[1])
            elif plot_option.quantity == 'position_noise':
                image_data = np.copy(self.noise[0])
            elif plot_option.quantity == 'velocity_noise':
                image_data = np.copy(self.noise[1])
            elif plot_option.quantity == 'force':
                image_data = np.copy(self.force)
            else:
                raise Exception("Invalid plot quantity requested")

            # Apply centering
            if plot_option.center:
                image_data_mean = np.mean(image_data)
                image_data -= image_data_mean

            # Apply scaling
            if plot_option.scale != 1:
                image_data = interpolate(image_data, plot_option.scale)

            # Set the image data
            image.setImage(image_data, levels=[-1, 1])

            # Save image
            if plot_option.export:
                output_filename = plot_option.format_str % self.counter
                output_path = os.path.join(plot_option.directory, output_filename)
                output = Image.fromarray((colormap(np.fliplr(image_data.T+1)/2)*255).astype(np.uint8))
                output.save(output_path)

        # Timing
        now = time.time()
        clock_dt = now - self.lastupdate
        clock_dt = max(clock_dt, 1/self.display_options.max_fps_report)
        fps_new = 1.0/clock_dt
        self.lastupdate = now
        self.fps = mix(self.fps, fps_new, 1 - self.display_options.fps_inertia)

        # Display the frame label
        frame_str = ''
        spacer_str = '    '
        frame_str += spacer_str + ('Frame Rate: %6.0f FPS' % self.fps)
        frame_str += spacer_str + ('Frame Count: %07d' % self.counter)
        self.label.setText(frame_str)

        # Update
        QtCore.QTimer.singleShot(1, self.update)
        self.counter += 1


########################################################################################################################
# Functions
########################################################################################################################
def interpolate(z, scale, order=3):
    """Wrapper around scipy.interpolate.RectBivariateSpline for data interpolation.
    :param z: 2-dimensional input data array
    :param scale: Factor by which to scale the data up or down
    :param order: Spline order, higher gives smoother interpolation at greater computational cost
    :type order: int
    :return: 2-dimensional output data array which has been scaled with interpolation
    """
    width, height = z.shape
    x, y = np.linspace(-1, 1, height), np.linspace(-1, 1, width)
    u, v = np.linspace(-1, 1, scale*height), np.linspace(-1, 1, scale*width)
    f = RectBivariateSpline(y, x, z, kx=order, ky=order)
    return f(v, u)


def get_lut(colormap_str):
    """Get a color lookup table array from a matplotlib colormap string for use in pyqtgraph.
    :param colormap_str: String to specify a Matplotlib colormap
    :type colormap_str: str
    :return: Color lookup table with RGB values in the range [0, 255]
    :rtype: Nx3 numpy array of color RGB values
    """
    colormap = cm.get_cmap(colormap_str)
    colormap._init()
    lut = colormap._lut[:256]*255
    return lut
