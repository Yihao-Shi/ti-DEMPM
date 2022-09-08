import taichi as ti
import DEMLib3D_v1.Quaternion as Quaternion
import DEMLib3D_v1.DEM_paricles as DEMParticle
import numpy 
import math


@ti.data_oriented
class SDEMParticle(DEMParticle):
    def __init__(self, max_particle_num):
        pass
