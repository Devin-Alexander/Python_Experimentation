# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import mpl_toolkits.mplot3d.axes3d as p3
# import matplotlib.animation as animation
# from numpy.random import random

# # Number of particles
# numP = 2
# # Dimensions
# DIM = 3
# timesteps = 2000

# x, y, z = random(timesteps), random(timesteps), random(timesteps)

# # Attaching 3D axis to the figure
# fig = plt.figure()
# ax = p3.Axes3D(fig)

# # Setting the axes properties
# border = 1
# ax.set_xlim3d([-border, border])
# ax.set_ylim3d([-border, border])
# ax.set_zlim3d([-border, border])
# line = ax.plot(x[:1], y[:1], z[:1], 'o')[0]


# def animate(i):
#     global x, y, z, numP
#     idx1 = numP*(i+1)
#     # join x and y into single 2 x N array
#     xy_data = np.c_[x[:idx1], y[:idx1]].T
#     line.set_data(xy_data)
#     line.set_3d_properties(z[:idx1])

# ani = animation.FuncAnimation(fig, animate, frames=timesteps, interval=1, blit=False, repeat=False)
# plt.show()






















# """
# ============
# 3D animation
# ============

# A simple example of an animated plot... In 3D!
# """
# import numpy as np
# import matplotlib.pyplot as plt
# import mpl_toolkits.mplot3d.axes3d as p3
# import matplotlib.animation as animation


# def Gen_RandLine(length, dims=2):
#     """
#     Create a line using a random walk algorithm

#     length is the number of points for the line.
#     dims is the number of dimensions the line has.
#     """
#     lineData = np.empty((dims, length))
#     lineData[:, 0] = np.random.rand(dims)
#     for index in range(1, length):
#         # scaling the random numbers by 0.1 so
#         # movement is small compared to position.
#         # subtraction by 0.5 is to change the range to [-0.5, 0.5]
#         # to allow a line to move backwards.
#         step = ((np.random.rand(dims) - 0.5) * 0.1)
#         lineData[:, index] = lineData[:, index - 1] + step

#     return lineData


# def update_lines(num, dataLines, lines):
#     #plt.cla()
#     for line, data in zip(lines, dataLines):
#         # NOTE: there is no .set_data() for 3 dim data...
#         line.set_data(data[0:2, :num])
#         line.set_3d_properties(data[2, :num])
#     #plt.show()
#     #plt.tight_layout()

#     return lines

# # Attaching 3D axis to the figure
# fig = plt.figure()
# ax = p3.Axes3D(fig)

# # Fifty lines of random 3-D lines
# data = [Gen_RandLine(25, 3) for index in range(50)]

# # Creating fifty line objects.
# # NOTE: Can't pass empty arrays into 3d version of plot()
# lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]

# # Setting the axes properties
# ax.set_xlim3d([0.0, 1.0])
# ax.set_xlabel('X')

# ax.set_ylim3d([0.0, 1.0])
# ax.set_ylabel('Y')

# ax.set_zlim3d([0.0, 1.0])
# ax.set_zlabel('Z')

# ax.set_title('3D Test')

# # Creating the Animation object
# line_ani = animation.FuncAnimation(fig, update_lines, 25, fargs=(data, lines),
#                                    interval=50, blit=False)

# plt.show()




















# from pyqtgraph.Qt import QtCore, QtGui
# import pyqtgraph.opengl as gl
# import numpy as np

# app = QtGui.QApplication([])
# w = gl.GLViewWidget()
# w.show()
# g = gl.GLGridItem()
# w.addItem(g)

# #generate random points from -10 to 10, z-axis positive
# pos = np.random.randint(-10,10,size=(1000,3))
# pos[:,2] = np.abs(pos[:,2])

# sp2 = gl.GLScatterPlotItem(pos=pos)
# w.addItem(sp2)

# #generate a color opacity gradient
# color = np.zeros((pos.shape[0],4), dtype=np.float32)
# color[:,0] = 1
# color[:,1] = 0
# color[:,2] = 0.5
# color[0:100,3] = np.arange(0,100)/100.

# def update():
#     ## update volume colors
#     global color
#     color = np.roll(color,1, axis=0)
#     sp2.setData(color=color)

# t = QtCore.QTimer()
# t.timeout.connect(update)
# t.start(50)









#!/usr/bin/env python

import numpy as np
import time
import matplotlib
# matplotlib.use('GTK5Agg')
from matplotlib import pyplot as plt


def randomwalk(dims=(256, 256), n=20, sigma=5, alpha=0.95, seed=1):
    """ A simple random walk with memory """

    r, c = dims
    gen = np.random.RandomState(seed)
    pos = gen.rand(2, n) * ((r,), (c,))
    old_delta = gen.randn(2, n) * sigma

    while True:
        delta = (1. - alpha) * gen.randn(2, n) * sigma + alpha * old_delta
        pos += delta
        for ii in xrange(n):
            if not (0. <= pos[0, ii] < r):
                pos[0, ii] = abs(pos[0, ii] % r)
            if not (0. <= pos[1, ii] < c):
                pos[1, ii] = abs(pos[1, ii] % c)
        old_delta = delta
        yield pos


def run(niter=1000, doblit=True):
    """
    Display the simulation using matplotlib, optionally using blit for speed
    """

    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal')
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    # ax.hold(True)
    rw = randomwalk()
    x, y = rw.next()

    plt.show(False)
    plt.draw()

    if doblit:
        # cache the background
        background = fig.canvas.copy_from_bbox(ax.bbox)

    points = ax.plot(x, y, 'o')[0]
    tic = time.time()

    for ii in xrange(niter):

        # update the xy data
        x, y = rw.next()
        points.set_data(x, y)

        if doblit:
            # restore background
            fig.canvas.restore_region(background)

            # redraw just the points
            ax.draw_artist(points)

            # fill in the axes rectangle
            fig.canvas.blit(ax.bbox)

        else:
            # redraw everything
            fig.canvas.draw()

    #print "Blit = %s, average FPS: %.2f" % (str(doblit), niter / (time.time() - tic))
    plt.show()

if __name__ == '__main__':
    run(doblit=False)
    run(doblit=True)
