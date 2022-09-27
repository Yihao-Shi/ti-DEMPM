import numpy as np
import matplotlib.pyplot as plt

t, pos, energy = [], [], []
for i in range(1, 76):
    data = np.load("MonitorDEM{0:06d}.npz".format(i))
    pos.append(data["pos_x"][0]-1)
    vel=data["vel_x"][0]
    time=data["t_current"]
    energy.append(0.5 * (25 - vel ** 2)/0.5/9.8)
    t.append(time)

plt.scatter(t, pos, marker='^', color='orange', label='Simulation: DEM_Taichi')
plt.plot(t, energy, color='black', label='Theory')
theory = 0.5 * 25 /0.5/9.8
print("simulation: ", pos[-1])
print("theory: ", theory)
print("Error: ", np.abs(pos[-1] - theory) / theory)
plt.xlim([1,2.4])
plt.ylim([0,2.6])
plt.xlabel("$t$ (s)")
plt.ylabel("$x$ (m)")
plt.legend(loc='best')
plt.show()
