from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

try:
    import numpy as np
except:
    exit()

from deap import benchmarks

'''
#NUMMAX = 5
#A = 10 * np.random.rand(NUMMAX, 2)
#C = np.random.rand(NUMMAX)

# A = [[0.5, 0.5], [0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]]
# C = [0.002, 0.005, 0.005, 0.005, 0.005]

A = [[50, 50], [25, 25], [25, 75], [75, 25], [75, 75]]
C = [30, 50, 40, 50, 60]

def shekel_arg0(sol):
    return benchmarks.shekel(sol, A, C)[0]

fig = plt.figure()
# ax = Axes3D(fig, azim = -29, elev = 50)
ax = Axes3D(fig)
# X = np.arange(0, 1, 0.01)
# Y = np.arange(0, 1, 0.01)
X = np.arange(0, 100, 1)
Y = np.arange(0, 100, 1)
X, Y = np.meshgrid(X, Y)
Z = np.fromiter(map(shekel_arg0, zip(X.flat,Y.flat)), dtype=np.float, count=X.shape[0]*X.shape[1]).reshape(X.shape)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1,  norm=LogNorm(), cmap=cm.jet, linewidth=0.2)

plt.xlabel("x")
plt.ylabel("y")

plt.show()
'''

###   movingpeak scenario 1

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt

try:
    import numpy as np
except:
    exit()

import random
rnd = random.Random()
rnd.seed(128)

from deap.benchmarks import movingpeaks

sc = movingpeaks.SCENARIO_3
sc["uniform_height"] = 0
sc["uniform_width"] = 0
sc["npeaks"] = 2
sc["period"] = 1000

mp = movingpeaks.MovingPeaks(dim=2, random=rnd, **sc)

mp.changePeaks()

print(mp.period)

X = np.arange(0, 100, 1.0)
Y = np.arange(0, 100, 1.0)
X, Y = np.meshgrid(X, Y)

for i in range(6):
    fig = plt.figure()
    ax = Axes3D(fig)
    Z = np.fromiter(map(lambda x: mp(x)[0], zip(X.flat,Y.flat)), dtype=np.float, count=X.shape[0]*X.shape[1]).reshape(X.shape)

    print(X, Y, Z)

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0.2)

    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()