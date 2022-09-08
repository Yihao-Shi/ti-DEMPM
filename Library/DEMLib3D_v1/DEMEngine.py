import taichi as ti
from DEMLib3D_v1.Function import *
import DEMLib3D_v1.Quaternion as Quaternion
import numpy 


@ti.data_oriented
class IntegrationScheme:
    def __init__(self, gravity, dt, partList, contModel):
        self.gravity = gravity
        self.dt = dt
        self.partList = partList
        self.contModel = contModel

        
    # ================= M. P. Allen, D. J. Tildesley (1989) Computer simualtion of liquids =============== #
    @ti.kernel
    def UpdatePosition(self):
        for np in range(self.partList.particleNum[None]):
            self.partList.x[np] += self.partList.v[np] * self.dt + 0.5 * self.partList.av[np] * self.dt ** 2
                
            self.partList.Fc[np] = ti.Matrix.zero(float, 3)
            self.partList.Tc[np] = ti.Matrix.zero(float, 3)
            self.partList.Fd[np] = ti.Matrix.zero(float, 3)
            self.partList.Td[np] = ti.Matrix.zero(float, 3)
            # Todo: Periodic condition


    @ti.kernel
    def UpdateAngularVelocity(self, t: float):
        for np in range(self.partList.particleNum[None]):
            matID = self.partList.materialID[np]
            beta = self.contModel.TorqueLocalDamping[matID]
            
            generalTorque = (self.partList.Tex[np] + self.partList.Td[np] + self.partList.Tc[np]) * Zero2OneVector(self.partList.fixedW[np])
            Lt = self.partList.rotate[np] @ (self.partList.Am[np] + 0.5 * self.dt * generalTorque)
            Lmid = self.partList.rotate[np] @ (self.partList.Am[np] + self.dt * generalTorque)
            wt = self.partList.inv_I[np] @ Lt
            wmid = self.partList.inv_I[np] @ Lmid
            qt = self.partList.q[np] + 0.5 * self.dt * Quaternion.SetDQ(self.partList.q[np], wt)
            self.partList.q[np] += self.dt * Quaternion.SetDQ(qt, wmid)
            self.partList.w[np] = self.partList.rotate[np].inverse() @ wmid
            self.partList.rotate[np] = Quaternion.SetToRotate(self.partList.q[np])
            
            if t == 0:
                self.partList.Am[np] += 0.5 * self.dt * generalTorque 
            else:
                self.partList.Am[np] += self.dt * generalTorque 


    @ti.kernel
    def UpdateVelocity(self):
        for np in range(self.partList.particleNum[None]):
            matID = self.partList.materialID[np]
            alpha = self.contModel.ForceLocalDamping[matID]
            v, av0 = self.partList.v[np], self.partList.av[np]

            generalForce = (self.partList.Fd[np] + self.partList.Fc[np] + self.partList.Fex[np] + self.gravity * self.partList.m[np]) * Zero2OneVector(self.partList.fixedV[np])
            self.partList.av[np] = (generalForce - alpha * generalForce.norm() * Normalize(self.partList.v[np])) / self.partList.m[np]
            self.partList.v[np] += 0.5 * self.dt * (av0 + self.partList.av[np])


@ti.data_oriented
class Euler(IntegrationScheme):
    def __init__(self, gravity, dt, partList, contModel):
        super().__init__(gravity, dt, partList, contModel)


@ti.data_oriented
class Verlet(IntegrationScheme):
    def __init__(self, gravity, dt, partList, contModel):
        super().__init__(gravity, dt, partList, contModel)