#!/usr/bin/env python
#
# Author: Ying Xiong.
# Created: Mar 18, 2014.


# requires Serial
# requires matplotlib
# requires pandas - Only if reading from csv
#

"""Utility functions for quaternion and spatial rotation.

A quaternion is represented by a 4-vector `q` as::

  q = q[0] + q[1]*i + q[2]*j + q[3]*k.

The validity of input to the utility functions are not explicitly checked for
efficiency reasons.

========  ================================================================
Abbr.      Meaning
========  ================================================================
quat      Quaternion, 4-vector.
vec       Vector, 3-vector.
ax, axis  Axis, 3- unit vector.
ang       Angle, in unit of radian.
rot       Rotation.
rotMatx   Rotation matrix, 3x3 orthogonal matrix.
HProd     Hamilton product.
conj      Conjugate.
recip     Reciprocal.
========  ================================================================
"""


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, ion, show

from itertools import count
import pandas as pd
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

import time

## From the serial collection sketch
import serial
from serial import Serial
ser = serial.Serial('COM13', 115200)

size_of_collection_array = 100
w_vals = np.zeros(size_of_collection_array)
i_vals = np.zeros(size_of_collection_array)
j_vals = np.zeros(size_of_collection_array)
k_vals = np.zeros(size_of_collection_array)

# locks in the code so each part the data aqc and he graphing can work side by side
lock_acq_data = 0
lock_graph = 0

def set_lock_aqc_data_to_zero():
    global lock_acq_data
    lock_acq_data = 0
    print("set_lock_aqc_data = 0")
    # print(lock_acq_data)

def set_lock_aqc_data_to_one():
    global lock_acq_data
    lock_acq_data = 1
    print("set_lock_aqc_data = 1")
    # print(lock_acq_data)

def set_lock_graph_to_zero():
    global lock_graph
    lock_graph = 0
    print("set_lock_graph = 0")
    # print(lock_graph)

def set_lock_graph_to_one():
    global lock_graph
    lock_graph = 1
    print("set_lock_graph = 1")
    # print(lock_graph)






init_vect = np.zeros(3)

def quatConj(q):
    """Return the conjugate of quaternion `q`."""
    return np.append(q[0], -q[1:])

def quatHProd(p, q):
    """Compute the Hamilton product of quaternions `p` and `q`."""
    r = np.array([p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3],
                  p[0]*q[1] + p[1]*q[0] + p[2]*q[3] - p[3]*q[2],
                  p[0]*q[2] - p[1]*q[3] + p[2]*q[0] + p[3]*q[1],
                  p[0]*q[3] + p[1]*q[2] - p[2]*q[1] + p[3]*q[0]])
    return r

def quatRecip(q):
    """Compute the reciprocal of quaternion `q`."""
    return quatConj(q) / np.dot(q,q)

def quatFromAxisAng(ax, theta):
    """Get a quaternion that performs the rotation around axis `ax` for angle
    `theta`, given as::

        q = (r, v) = (cos(theta/2), sin(theta/2)*ax).

    Note that the input `ax` needs to be a 3x1 unit vector."""
    return np.append(np.cos(theta/2), np.sin(theta/2)*ax)

def quatFromRotMatx(R):
    """Get a quaternion from a given rotation matrix `R`."""
    q = np.zeros(4)

    q[0] = ( R[0,0] + R[1,1] + R[2,2] + 1) / 4.0
    q[1] = ( R[0,0] - R[1,1] - R[2,2] + 1) / 4.0
    q[2] = (-R[0,0] + R[1,1] - R[2,2] + 1) / 4.0
    q[3] = (-R[0,0] - R[1,1] + R[2,2] + 1) / 4.0

    q[q<0] = 0   # Avoid complex number by numerical error.
    q = np.sqrt(q)

    q[1] *= np.sign(R[2,1] - R[1,2])
    q[2] *= np.sign(R[0,2] - R[2,0])
    q[3] *= np.sign(R[1,0] - R[0,1])

    return q

def quatToRotMatx(q):
    """Get a rotation matrix from the given unit quaternion `q`."""
    R = np.zeros((3,3))

    R[0,0] = 1 - 2*(q[2]**2 + q[3]**2)
    R[1,1] = 1 - 2*(q[1]**2 + q[3]**2)
    R[2,2] = 1 - 2*(q[1]**2 + q[2]**2)

    R[0,1] = 2 * (q[1]*q[2] - q[0]*q[3])
    R[1,0] = 2 * (q[1]*q[2] + q[0]*q[3])

    R[0,2] = 2 * (q[1]*q[3] + q[0]*q[2])
    R[2,0] = 2 * (q[1]*q[3] - q[0]*q[2])

    R[1,2] = 2 * (q[2]*q[3] - q[0]*q[1])
    R[2,1] = 2 * (q[2]*q[3] + q[0]*q[1])

    return R

def rotVecByQuat(u, q):
    """Rotate a 3-vector `u` according to the quaternion `q`. The output `v` is
    also a 3-vector such that::

        [0; v] = q * [0; u] * q^{-1}

    with Hamilton product."""
    v = quatHProd(quatHProd(q, np.append(0, u)), quatRecip(q))
    return v[1:]

def rotVecByAxisAng(u, ax, theta):
    """Rotate the 3-vector `u` around axis `ax` for angle `theta` (radians),
    counter-clockwisely when looking at inverse axis direction. Note that the
    input `ax` needs to be a 3x1 unit vector."""
    q = quatFromAxisAng(ax, theta)
    return rotVecByQuat(u, q)




old_vect = np.zeros(3)
# old_vect = np.array([1.0, 1.0, 1.0])
# old_vect = old_vect / np.linalg.norm(old_vect)
curr_vect = np.zeros(3)

def animate(i):
    # FOR WHEN READING THROUGH A .CSV
    # data = pd.read_csv('data.csv')
    # u is the unit vector for now, but will be the one that is aligned with the magnetic grid
                
    ###########################################
    # print(i)
    q = np.zeros(4)


    q[0] = w_vals[i]
    q[1] = i_vals[i]
    q[2] = j_vals[i]
    q[3] = k_vals[i]

    # print("pre norm quaternian")
    # print(q)
    q /= np.linalg.norm(q)
    # print("post norm quaternian")
    # print(q)
    
    # unit_vect = np.array([1.0, 1.0, 1.0])
    # init_vect = unit_vect
    
    init_vect = np.array([1.0, 1.0, 1.0])
    init_vect = init_vect / np.linalg.norm(init_vect)

    # clear previous plot
    #plt.cla()

    # Setting the axes properties
    ax.set_xlim3d([-1.0, 1.0])
    ax.set_xlabel('X')

    ax.set_ylim3d([-1.0, 1.0])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-1.0, 1.0])
    ax.set_zlabel('Z')

    ax.set_title('Position Displayed as a Unit Vector')
    # ax.plot_surface(x_vals, y_vals, z_vals, color='b')

    # calculate new vector to print
    new_vect = rotVecByQuat(init_vect, q)

    # print("new_vect")
    # print(new_vect)

    # plot the new vector
    ax.plot([0, new_vect[0]], [0, new_vect[1]], [0, new_vect[2]], 'y')
    #curr_vect = ax.plot(1000 * [0, new_vect[0]], 1000 * [0, new_vect[1]], 1000 * [0, new_vect[2]], 'y')

    # print("inside the graphing section")
   
    # line = old_vect.pop(0)
    # line.remove()


    # old_vect = curr_vect

    

    # plt.tight_layout()
    plt.show()
    plt.pause(0.5)
    # plt.draw()
    # plt.show()
    set_lock_graph_to_zero()

    return


# Rotation axis.
ax = np.array([1.0, 1.0, 1.0])
print("print pre normalization")
print(ax)
ax = ax / np.linalg.norm(ax)
print("print post normalization")
print(ax)

# Rotation angle.
theta = -5*np.pi/6

# Original vector.
u = [0.5, 0.6, np.sqrt(3)/2]
u /= np.linalg.norm(u)

# Draw the circle frame.
nSamples = 1000
t = np.linspace(-np.pi, np.pi, nSamples)
z = np.zeros(t.shape)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax_2 = fig.add_subplot(111, projection='3d')

ax.plot(np.cos(t), np.sin(t), z, 'b')
ax.plot(z, np.cos(t), np.sin(t), 'b')
ax.plot(np.cos(t), z, np.sin(t), 'b')

# # Draw rotation axis.
# ax.plot([0, ax[0]*2], [0, ax[1]*2], [0, ax[2]*2], 'r')

# # Rotate the `u` vector and draw results.
# ax.plot([0, u[0]], [0, u[1]], [0, u[2]], 'm')
# v = rotVecByAxisAng(u, ax, theta)
# ax.plot([0, v[0]], [0, v[1]], [0, v[2]], 'm')

# # Draw the circle that is all rotations of `u` across `ax` with different
# # angles.
# v = np.zeros((3, len(t)))
# for i,theta in enumerate(t):
#     v[:,i] = rotVecByAxisAng(u, ax, theta)
# ax.plot(v[0,:], v[1,:], v[2,:], 'm')

#########################################################################################ax.view_init(elev=8, azim=80)



# to get an initial vector, you need to align with the 
# for now, start off with the initial vector being the unit vector (3^1/2)/3 for i, j, and k components

# plt.show(block = False)
plt.pause(0.05)


init_vect = ax


print("Python sketch begin serial portion")
ser_bytes = ser.readline()
print(ser_bytes)
ser_bytes = ser.readline()
print(ser_bytes)
ser_bytes = ser.readline()
print(ser_bytes)
ser_bytes = ser.readline()
print(ser_bytes)
ser_bytes = ser.readline()
print(ser_bytes)
ser_bytes = ser.readline()
print(ser_bytes)
ser_bytes = ser.readline()
print(ser_bytes)
ser_bytes = ser.readline()
print(ser_bytes)
ser_bytes = ser.readline()
print(ser_bytes)
ser_bytes = ser.readline()
print(ser_bytes)
ser_bytes = ser.readline()
print(ser_bytes)
ser_bytes = ser.readline()
print(ser_bytes)



i = 0
num_chars_to_parse = 6

lock_acq_data = 0
lock_graph = 0

plt.ion() # enables interactive mode

while i < size_of_collection_array:
    ser_bytes = ser.read() 
    #print(ser_bytes)

    if ser_bytes == b'$': # begin acquiring the data until one full set of quaternian data is collected
        print("")
        print(i)

        set_lock_aqc_data_to_one() # lock to the data collection task only

        # print(lock_acq_data)
        while lock_acq_data == 1:
            ser_bytes = ser.read() 

            if ser_bytes == b'^':
                w_vals[i] = float(ser.read(num_chars_to_parse)) # read 4 bytes and cast it into a float
                # print("w")
                # print(w_vals[i])

            if ser_bytes == b'~':
                i_vals[i] = float(ser.read(num_chars_to_parse)) # read 4 bytes and cast it into a float
                # print("i")
                # print(i_vals[i])

            if ser_bytes == b'!':
                j_vals[i] = float(ser.read(num_chars_to_parse)) # read 4 bytes and cast it into a float
                # print("j")
                # print(j_vals[i])

            if ser_bytes == b'@':
                k_vals[i] = float(ser.read(num_chars_to_parse)) # read 4 bytes and cast it into a float
                # print("k")
                # print(k_vals[i])
            
            if ser_bytes == b'#':
                set_lock_aqc_data_to_zero() # set the lock to 0 to allow the while loop to break
                set_lock_graph_to_one()     # lock this portion of the function so that there can be a call to print out the most recent value to the graph
                

    
    
    while lock_graph == 1:
        print("graphing mode")
        plt.gcf()
        animate(i)                
        # plt.show(block = False)
        i += 1
        
print("done collecting data")

print("w_vals")
print(w_vals)
print("i_vals")
print(i_vals)
print("j_vals")
print(j_vals)
print("k_vals")
print(k_vals)

#################################################################################################

# q = np.zeros(4)

# for i in range(20):

#     print(i)
#     w_comp = w_vals[i]
#     i_comp = i_vals[i]
#     j_comp = j_vals[i]
#     k_comp = k_vals[i]

#     q[0] = w_comp
#     q[1] = i_comp
#     q[2] = j_comp
#     q[3] = k_comp

#     print("pre norm quaternian")
#     print(q)
#     q /= np.linalg.norm(q)
#     print("post norm quaternian")
#     print(q)


#     # clear previous plot
#     # plt.cla()

#     # ax.plot_surface(x_vals, y_vals, z_vals, color='b')

#     # calculate new vector to print
#     new_vect = rotVecByQuat(init_vect, q)

#     print("new_vect")
#     print(new_vect)

#     # plot the new vector
    
#     ax.plot(1000 * [0, new_vect[0]], 1000 * [0, new_vect[1]], 1000 * [0, new_vect[2]], 'y')

#     print("thhhhh")
#     #    ax.scatter(x,y,z,c=np.linalg.norm([x,y,z], axis=0))
#     plt.tight_layout()
#     plt.show()
    

##################################################################################################

# ani = FuncAnimation(plt.gcf(), animate, frames=size_of_collection_array, interval=1, blit=False, repeat=False)
print("Done")


# Comment out for now 
plt.show(block = True)
print("terminate")
