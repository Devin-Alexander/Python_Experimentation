print("Hello World")

#You will need to install matplotlib to be able to install this

#from matplotlib import pyplot as plt 


#dev_x = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

#dev_y = [38496, 42000, 46752, 49320, 53200, 56000, 62316, 64928, 67317, 68748, 73752]

#dev_z = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

#plt.plot(dev_x, dev_y, dev_z)
#plt.show()

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

#%matplotlib notebook 
#setup figure size and DPI for screen demo
plt.rcParams['figure.figsize'] = (6,4)
plt.rcParams['figure.dpi'] = 150

from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

#plt.show()

x = np.random.normal(size=500)
y = np.random.normal(size=500)
z = np.random.normal(size=500)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


plt.legend()
plt.tight_layout()

plt.show()
