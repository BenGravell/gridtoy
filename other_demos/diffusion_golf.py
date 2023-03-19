from numpy.random import rand as d
from scipy.signal import convolve2d as c
import matplotlib.pyplot as p
from matplotlib.animation import FuncAnimation as F
r=lambda:d(99,99);x,e=r(),r()-0.5;i=p.imshow(x)
def u(_):global x,e;e-=(e-r()+0.5)/8;x=(c(x,[[0,1,0],[1,4,1],[0,1,0]],'same','wrap')+e)/8;i.set_data(x);return i,
a=F(p.gcf(),u,interval=0,blit=True);p.show()
