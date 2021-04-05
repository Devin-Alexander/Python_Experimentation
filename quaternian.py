#!/usr/bin/env python
#
# Author: Ying Xiong.
# Created: Mar 18, 2014.

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

from itertools import count
import pandas as pd
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from time import sleep




################This is just some temporary things to get me going
w_vals = [-0.66 , -0.66 , -0.66 , -0.78 , -0.89 , -0.86 , -0.77 , -0.55 , -0.52 , -0.54 , -0.68 , -0.64 , -0.59 , -0.37 , 0.31  , 0.25  , 0.21  , 0.39 , 0.13 , -0.27]
i_vals = [-0.60 ,-0.62  , -0.63 , -0.46 , -0.21 , 0.13  , 0.36  , 0.55  , 0.60  , 0.48  , 0.14  , -0.52 , -0.53 , -0.45 , 0.30  , -0.09 , -0.03 ,-0.16 , -0.40, -0.46]
j_vals = [0.32  , 0.30  , 0.28  , 0.19  , 0.06  , -0.15 , -0.29 , -0.46 , -0.45 , -0.40 , -0.27 , 0.24  , 0.33  , 0.58  , -0.71 , 0.71  , 0.73  , 0.63 , 0.58 , 0.54]
k_vals = [0.32  , 0.31  , 0.29  , 0.37  , 0.40  , 0.47  , 0.43  , 0.43  , 0.42  , 0.56  , 0.66  , 0.50  , 0.51  , 0.57  , -0.56 , 0.65  , 0.65  , 0.65 , 0.70 , 0.65]

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




def animate(i):
    # FOR WHEN READING THROUGH A .CSV
    # data = pd.read_csv('data.csv')
    # u is the unit vector for now, but will be the one that is aligned with the magnetic grid

    w_vals = [-0.66 , -0.66 , -0.66 , -0.78 , -0.89 , -0.86 , -0.77 , -0.55 , -0.52 , -0.54 , -0.68 , -0.64 , -0.59 , -0.37 , 0.31  , 0.25  , 0.21  , 0.39 , 0.13 , -0.27]
    i_vals = [-0.60 ,-0.62  , -0.63 , -0.46 , -0.21 , 0.13  , 0.36  , 0.55  , 0.60  , 0.48  , 0.14  , -0.52 , -0.53 , -0.45 , 0.30  , -0.09 , -0.03 ,-0.16 , -0.40, -0.46]
    j_vals = [0.32  , 0.30  , 0.28  , 0.19  , 0.06  , -0.15 , -0.29 , -0.46 , -0.45 , -0.40 , -0.27 , 0.24  , 0.33  , 0.58  , -0.71 , 0.71  , 0.73  , 0.63 , 0.58 , 0.54]
    k_vals = [0.32  , 0.31  , 0.29  , 0.37  , 0.40  , 0.47  , 0.43  , 0.43  , 0.42  , 0.56  , 0.66  , 0.50  , 0.51  , 0.57  , -0.56 , 0.65  , 0.65  , 0.65 , 0.70 , 0.65]

    init_vector = ax

    ###########################################
    print(i)
    w_comp = w_vals[i]
    i_comp = i_vals[i]
    j_comp = j_vals[i]
    k_comp = k_vals[i]

    q[0] = w_comp
    q[1] = i_comp
    q[2] = j_comp
    q[3] = k_comp

    print("pre norm quaternian")
    print(q)
    q /= np.linalg.norm(q)
    print("post norm quaternian")
    print(q)


    # clear previous plot
    # plt.cla()

    # ax.plot_surface(x_vals, y_vals, z_vals, color='b')

    # calculate new vector to print
    new_vect = rotVecByQuat(init_vect, q)

    print("new_vect")
    print(new_vect)

    # plot the new vector
    
    fig_ax.plot(1000 * [0, new_vect[0]], 1000 * [0, new_vect[1]], 1000 * [0, new_vect[2]], 'y')
    #    ax.scatter(x,y,z,c=np.linalg.norm([x,y,z], axis=0))
    plt.tight_layout()
    plt.show()

    #################################################3

        
    # ax = np.array([1.0, 1.0, 1.0])
    # ax = ax / np.linalg.norm(ax)

    # init_vector = ax

    # fig = plt.figure()
    # fig_ax = fig.add_subplot(111, projection='3d')

    # w_comp = w_vals[i]
    # i_comp = i_vals[i]
    # j_comp = j_vals[i]
    # k_comp = k_vals[i]

    # q = np.zeros(4)
    
    # q[0] = w_comp
    # q[1] = i_comp
    # q[2] = j_comp
    # q[3] = k_comp

    # print(q)

    # # clear previous plot
    # ######################333plt.cla()

    # # ax.plot_surface(x_vals, y_vals, z_vals, color='b')

    # # calculate new vector to print
    # new_vect = rotVecByQuat(init_vector, q)

    # print(new_vect)

    # # plot the new vector
    
    # fig_ax.plot([0, new_vect[0]], [0, new_vect[1]], [0, new_vect[2]], 'g')


    # plt.tight_layout()
    # plt.show()
    return



def quatDemo():
    
    
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
    fig_ax = fig.add_subplot(111, projection='3d')

    fig_ax.plot(np.cos(t), np.sin(t), z, 'b')
    fig_ax.plot(z, np.cos(t), np.sin(t), 'b')
    fig_ax.plot(np.cos(t), z, np.sin(t), 'b')

    # Draw rotation axis.
    fig_ax.plot([0, ax[0]*2], [0, ax[1]*2], [0, ax[2]*2], 'r')

    # Rotate the `u` vector and draw results.
    fig_ax.plot([0, u[0]], [0, u[1]], [0, u[2]], 'm')
    v = rotVecByAxisAng(u, ax, theta)
    fig_ax.plot([0, v[0]], [0, v[1]], [0, v[2]], 'm')

    # Draw the circle that is all rotations of `u` across `ax` with different
    # angles.
    v = np.zeros((3, len(t)))
    for i,theta in enumerate(t):
        v[:,i] = rotVecByAxisAng(u, ax, theta)
    fig_ax.plot(v[0,:], v[1,:], v[2,:], 'm')

    #########################################################################################fig_ax.view_init(elev=8, azim=80)
    
    

    # to get an initial vector, you need to align with the 
    # for now, start off with the initial vector being the unit vector (3^1/2)/3 for i, j, and k components
    
    #plt.show()
    print("jere")

    init_vect = ax
    ##################################################################################################
    w_vals = [-0.66 , -0.66 , -0.66 , -0.78 , -0.89 , -0.86 , -0.77 , -0.55 , -0.52 , -0.54 , -0.68 , -0.64 , -0.59 , -0.37 , 0.31  , 0.25  , 0.21  , 0.39 , 0.13 , -0.27]
    i_vals = [-0.60 ,-0.62  , -0.63 , -0.46 , -0.21 , 0.13  , 0.36  , 0.55  , 0.60  , 0.48  , 0.14  , -0.52 , -0.53 , -0.45 , 0.30  , -0.09 , -0.03 ,-0.16 , -0.40, -0.46]
    j_vals = [0.32  , 0.30  , 0.28  , 0.19  , 0.06  , -0.15 , -0.29 , -0.46 , -0.45 , -0.40 , -0.27 , 0.24  , 0.33  , 0.58  , -0.71 , 0.71  , 0.73  , 0.63 , 0.58 , 0.54]
    k_vals = [0.32  , 0.31  , 0.29  , 0.37  , 0.40  , 0.47  , 0.43  , 0.43  , 0.42  , 0.56  , 0.66  , 0.50  , 0.51  , 0.57  , -0.56 , 0.65  , 0.65  , 0.65 , 0.70 , 0.65]

    q = np.zeros(4)

    for i in range(20):

        print(i)
        w_comp = w_vals[i]
        i_comp = i_vals[i]
        j_comp = j_vals[i]
        k_comp = k_vals[i]

        q[0] = w_comp
        q[1] = i_comp
        q[2] = j_comp
        q[3] = k_comp

        print("pre norm quaternian")
        print(q)
        q /= np.linalg.norm(q)
        print("post norm quaternian")
        print(q)


        # clear previous plot
        # plt.cla()

        # ax.plot_surface(x_vals, y_vals, z_vals, color='b')

        # calculate new vector to print
        new_vect = rotVecByQuat(init_vect, q)

        print("new_vect")
        print(new_vect)

        # plot the new vector
        
        fig_ax.plot(1000 * [0, new_vect[0]], 1000 * [0, new_vect[1]], 1000 * [0, new_vect[2]], 'y')

        print("thhhhh")
        #    ax.scatter(x,y,z,c=np.linalg.norm([x,y,z], axis=0))
        plt.tight_layout()
        #usleep(250)
    
    ##################################################################################################
    #FuncAnimation(plt.gcf(), animate, interval=250)
    print("Done")

    
    # Comment out for now 
    plt.show()

if __name__ == "__main__":
    quatDemo()