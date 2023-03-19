import sys
import time
import numpy as np
import numpy.random as npr
from scipy.signal import convolve2d
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg

from simulation import smoothstep, mix
from display import get_lut

n = 200
m = int(n/4)
state = np.zeros([n, n])


state[m:3*m, m:3*m] = npr.rand(2*m, 2*m)
# glider = np.array([[0, 1, 0],
#                    [0, 0, 1],
#                    [1, 1, 1]])



# for k in range(int(0.005*n**2)):
#
#     def a(): return 1+0.1*npr.rand()
#
#     i, j = m+npr.randint(2*m), m+npr.randint(2*m)
#     state[i:i+3, j:j+3] = np.array([[0, a(), 0],
#                    [0, 0, a()],
#                    [a(), a(), a()]])

# state[m:m+3, m:m+3] = 0.99*glider

neighbor_kernel = np.array([[0.5, 1, 0.5],
                            [1, 0, 1],
                            [0.5, 1, 0.5]])
neighbor_kernel = np.array([[1, 1, 1],
                            [1, 0, 1],
                            [1, 1, 1]])

def f(x):
    return smoothstep(np.clip(x, 0, 1))


def g(x, lo, hi):
    y = np.zeros_like(x)
    y = np.where(x > lo-1, smoothstep(x - (lo - 1)), y)
    y = np.where(x > lo, 1, y)
    y = np.where(x > hi, smoothstep(-x + (hi + 1)), y)
    y = np.where(x > hi+1, 0, y)
    return y


def physics_update(state):
    neighbors = convolve2d(state, neighbor_kernel, mode='same', boundary='wrap')
    liveness = f(state)
    companionship = g(neighbors, lo=2, hi=3) - 1
    nascence = g(neighbors, lo=3, hi=3)
    return state + (np.multiply(liveness, companionship) + np.multiply(1 - liveness, nascence))  #+ 0.1*(npr.rand(n, n)-0.5)


class App(QtGui.QMainWindow):
    """Application class which extends the QT class QMainWindow for grid dynamics simulation visualization"""
    def __init__(self, state):
        # Make QtGui.QMainWindow a superclass of this App class
        super().__init__()

        # Always choose the plot sizes to be the same as the simulation domain size
        self.plot_size = (n, n)

        # Save the options
        self.state = state

        show_histogram = False
        colormap_str = 'inferno'

        # Create GUI elements
        self.mainbox = QtGui.QWidget()
        self.setCentralWidget(self.mainbox)
        self.mainbox.setLayout(QtGui.QVBoxLayout())

        self.canvas = pg.GraphicsLayoutWidget()
        self.mainbox.layout().addWidget(self.canvas)

        # Add the images and histograms
        self.layouts = []
        self.views = []
        self.images = []
        self.histograms = []
        self.colormaps = []
        layout = self.canvas.addLayout()

        # Add the title label
        title_str = 'State'
        title_label = layout.addLabel(title_str)
        title_label.setText(title_str, size='20pt')
        layout.nextRow()

        # Add the image
        view = layout.addViewBox()
        view.setAspectLocked(True)
        view.setRange(QtCore.QRectF(0, 0, self.plot_size[0], self.plot_size[1]))
        image = pg.ImageItem()
        view.addItem(image)

        # Add the histogram
        if show_histogram:
            histogram = pg.HistogramLUTItem()
            histogram.setImageItem(image)
            histogram.gradient.loadPreset(colormap_str)
            self.canvas.addItem(histogram)
        else:
            image.setLookupTable(get_lut(colormap_str))

        # Save the graphics object handles
        self.image = image

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
        self.state = physics_update(self.state)
        self.image.setImage(self.state, levels=[0, 1])

        # Timing
        now = time.time()
        clock_dt = now - self.lastupdate
        clock_dt = max(clock_dt, 1/1000)
        fps_new = 1.0/clock_dt
        self.lastupdate = now
        self.fps = mix(self.fps, fps_new, 1 - 0.9)

        # Display the frame label
        frame_str = ''
        spacer_str = '    '
        frame_str += spacer_str + ('Frame Rate: %6.0f FPS' % self.fps)
        frame_str += spacer_str + ('Frame Count: %07d' % self.counter)
        self.label.setText(frame_str)

        # Update
        QtCore.QTimer.singleShot(1, self.update)
        self.counter += 1


# Start the display app
app = QtGui.QApplication(sys.argv)
thisapp = App(state)
thisapp.show()
sys.exit(app.exec_())
