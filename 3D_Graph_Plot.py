# Required installations: 
# Matplotlib
# pandas
# 


print ("3D Graph")



import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


from itertools import count
import pandas as pd
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('fivethirtyeight')

################################################################################################
# Tutorial scatter plot in 3D
################################################################################################

#setup figure size and DPI for screen demo
# plt.rcParams['figure.figsize'] = (6,4)
# plt.rcParams['figure.dpi'] = 150
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

################################################################################################
# Tutorial #9 Plot Sphere at the origin point
################################################################################################
# Make data for origin point sphere
u_sphere = np.linspace(0, 2 * np.pi, 100)
v_sphere = np.linspace(0, np.pi, 100)
origin_point_size = 0.1
x_sphere = origin_point_size * np.outer(np.cos(u_sphere), np.sin(v_sphere))
y_sphere = origin_point_size * np.outer(np.sin(u_sphere), np.sin(v_sphere))
z_sphere = origin_point_size * np.outer(np.ones(np.size(u_sphere)), np.cos(v_sphere))

# Plot the surface
# ax.plot_surface(x_sphere, y_sphere, z_sphere, color='b')

# plt.show()

################################################################################################
# Tutorial scatter plot in 3D
################################################################################################

# Example 3D data
x_vals = np.random.normal(size=100)
y_vals = np.random.normal(size=100)
z_vals = np.random.normal(size=100)


# Add legend information later
# plt.legend()

# plt.tight_layout()
# ax.scatter(x_vals,y_vals,z_vals,c=np.linalg.norm([x_vals,y_vals,z_vals], axis=0))

################################################################################################
# Tutorial #9 Plotting Live Data in Real-Time
################################################################################################


def animate(i):
    # FOR WHEN READING THROUGH A .CSV
    # data = pd.read_csv('data.csv')
    # x = data['x_value']
    # y = data['total_2']

    x = x_vals[i]
    y = y_vals[i]
    z = z_vals[i]

    plt.cla()

    # ax.plot_surface(x_vals, y_vals, z_vals, color='b')

    # Setting the axes properties
    ax.set_xlim3d([-1.0, 1.0])
    ax.set_xlabel('X')

    ax.set_ylim3d([-1.0, 1.0])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-1.0, 1.0])
    ax.set_zlabel('Z')

    ax.set_title('3D Test')


    ax.scatter(x,y,z,c=np.linalg.norm([x,y,z])) #, axis=0))

    # plt.legend(loc='upper left')
    # plt.tight_layout()

# Setting the axes properties

ax.set_title('3D Test')



ani = FuncAnimation(plt.gcf(), animate, interval=1)

# x_vals = []
# y_vals = []

# plt.plot([], [], label='Channel 1')
# plt.plot([], [], label='Channel 2')


# def animate(i):
#     data = pd.read_csv('data.csv')
#     x = data['x_value']
#     y1 = data['total_1']
#     y2 = data['total_2']

#     ax = plt.gca()
#     line1, line2 = ax.lines

#     line1.set_data(x, y1)
#     line2.set_data(x, y2)

#     xlim_low, xlim_high = ax.get_xlim()
#     ylim_low, ylim_high = ax.get_ylim()

#     ax.set_xlim(xlim_low, (x.max() + 5))

#     y1max = y1.max()
#     y2max = y2.max()
#     current_ymax = y1max if (y1max > y2max) else y2max

#     y1min = y1.min()
#     y2min = y2.min()
#     current_ymin = y1min if (y1min < y2min) else y2min

#     ax.set_ylim((current_ymin - 5), (current_ymax + 5))


# ani = FuncAnimation(plt.gcf(), animate, interval=1000)

# plt.legend()
# plt.tight_layout()

ax.view_init(elev=8, azim=80)
ax.grid(True)
plt.rcParams['figure.figsize'] = (6,4)
plt.rcParams['figure.dpi'] = 150
plt.show()