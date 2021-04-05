import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


testData = np.random.rand(100,2)-[0.5, 0.5]

fig, ax = plt.subplots()
circle1=plt.Circle(testData[20,:],0,color='r',fill=False, clip_on = False)
ax.add_artist(circle1)
# i is the radius
def animate(i):
    #init()
    #you don't want to redraw the same dots over and over again, do you?
    circle1.set_radius(i)
    return


def init():
    sctPlot, = ax.plot(testData[:,0], testData[:,1], ".")
    return sctPlot,

ani = animation.FuncAnimation(fig, animate, np.arange(0.4, 2, 0.1), init_func=init,
    interval=25, blit=False)
plt.show()





# fig, ax = plt.subplots()
# def init():
#     fig.clf()
#     sctPlot, = ax.plot(testData[:,0], testData[:,1], ".")

# # i is the radius
# def animate(i):
#     init()
#     q = 20 #initial index for the starting position
#     pos = testData[q,:]
#     circle1=plt.Circle(pos,i,color='r',fill=False, clip_on = False)
#     fig.gca().add_artist(circle1)

# def init():
#     sctPlot, = ax.plot(testData[:,0], testData[:,1], ".")
#     return sctPlot,

# ani = animation.FuncAnimation(fig, animate, np.arange(0.4, 2, 0.1), init_func=init,
#     interval=25, blit=False)
# plt.show()