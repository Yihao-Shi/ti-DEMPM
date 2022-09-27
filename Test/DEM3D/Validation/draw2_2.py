import numpy as np
import math
import matplotlib.pyplot as plt

t, v, energy, vt = [], [], [], []
m = 4./3.*math.pi*0.025**3*2650
J = 2./5.*m*0.025**2
for i in range(1, 101):
    data = np.load("MonitorDEM{0:06d}.npz".format(i))
    if i==1: posz0 = data["pos_z"][0]    
    posz = data["pos_z"][0]
    velx=data["vel_x"][0]
    vely=data["vel_y"][0]
    velz=data["vel_z"][0]
    wx=data["w_x"][0]
    wy=data["w_y"][0]
    wz=data["w_z"][0]
    time=data["t_current"] - 0.5
    energy.append(0.5*m*(velx**2+vely**2+velz**2)+0.5*J*(wx**2+wy**2+wz**2)+9.8*m*(posz-posz0))
    v.append(math.sqrt(velx**2+vely**2+velz**2))
    vt.append(9.8*time*math.sqrt(2.)/2.*(1-0.2))
    t.append(time)


plt.scatter(t, v, marker='^', color='orange', label='Simulation: DEM_Taichi')
plt.plot(t, vt, color='black', label='Theory')
plt.xlim([0,2])
plt.ylim([0,11])
plt.xlabel("$t$ (s)")
plt.ylabel("$v$ (m/s)")
plt.legend(loc='best')
plt.show()
plt.scatter(t, energy)
plt.show()
