import numpy as np
import matplotlib.pyplot as plt

t, v, pos = [], [], []
for i in range(1, 100):
    data = np.load("MonitorDEM{0:06d}.npz".format(i))
    pos.append(-data["pos_y"][0]+2.2)
    v.append(data["vel_y"][0])
    time=data["t_current"]
    t.append(time)

plt.scatter(t, v, marker='^', color='orange', label='Simulation: DEM_Taichi')
plt.xlabel("$t$ (s)")
plt.ylabel("$v$ (m/s)")
plt.legend(loc='best')
plt.show()
plt.scatter(t, pos, marker='^', color='orange', label='Simulation: DEM_Taichi')
plt.xlim([0,0.2])
plt.xlabel("$t$ (s)")
plt.ylabel("$x$ (m)")
plt.legend(loc='best')
plt.show()
