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

    @ti.kernel
    def IntegrationInit(self):
        for np in range(self.partList.particleNum[None]):
            self.partList.av[np] = self.partList.Fex[np] / self.partList.m[np] + self.gravity
            self.partList.aw[np] = self.partList.Tex[np] * self.partList.inv_I[np]

    @ti.kernel
    def Reset(self):
        for np in range(self.partList.particleNum[None]):
            self.partList.Fc[np] = ti.Matrix.zero(float, 3)
            self.partList.Tc[np] = ti.Matrix.zero(float, 3)

    @ti.func
    def EulerEquations(self, np, w_local, T_local, aw_local):
        i_intertia = self.partList.inv_I[np]
        intertia = self.partList.inv_I[np].inverse()
        for d in ti.static(range(3)):
            aw_local[d] = T_local[d] - (aw_local[(d+1)%3] * intertia[(d+2)%3] * aw_local[(d+2)%3] - aw_local[(d+2)%3] * intertia[(d+1)%3] * aw_local[(d+1)%3]) * i_intertia[j]
        return aw_local


    # =============== M. P. Allen, D. J. Tildesley (1989) Computer Simualtion of Liquids ================= #
    @ti.func
    def UpdateQuanternion(self, np, wt, wmid):
        dq = Quaternion.SetDQ(self.partList.q[np], wt)
        qt = self.partList.q[np] + 0.5 * self.dt * dq
        self.partList.q[np] += self.dt * Quaternion.SetDQ(qt, wmid)
        self.partList.q[np] = Quaternion.Normalized(self.partList.q[np])
        self.partList.rotate[np] = Quaternion.SetToRotate(self.partList.q[np])


@ti.data_oriented
class Euler(IntegrationScheme):
    def __init__(self, gravity, dt, partList, contModel):
        super().__init__(gravity, dt, partList, contModel)
    
    @ti.func
    def ParticleTranslation(self, np):
        matID = self.partList.materialID[np]
        ForceLocalDamping = self.contModel.ForceLocalDamping[matID]

        generalForce = self.partList.Fc[np] + self.partList.Fex[np] + self.gravity * self.partList.m[np]
        force = (generalForce - ForceLocalDamping * generalForce.norm() * Normalize(self.partList.v[np]))
        self.partList.av[np] = force / self.partList.m[np] * Zero2OneVector(self.partList.fixedV[np])
        self.partList.v[np] += self.dt * self.partList.av[np]

        self.partList.x[np] += self.partList.v[np] * self.dt 
        self.partList.disp[np] += self.partList.v[np] * self.dt 

    @ti.func
    def SphereRotation(self, np):
        matID = self.partList.materialID[np]
        TorqueLocalDamping = self.contModel.TorqueLocalDamping[matID]

        generalTorque = self.partList.Tex[np] + self.partList.Tc[np]
        torque =  (generalTorque - TorqueLocalDamping * generalTorque.norm() * Normalize(self.partList.w[np]))
        self.partList.aw[np] = torque * self.partList.inv_I[np] * Zero2OneVector(self.partList.fixedW[np])
        self.partList.w[np] += self.dt * self.partList.aw[np]

        self.partList.theta[np] += (self.partList.w[np] * self.dt).norm() 

    @ti.kernel
    def Integration(self):
        for np in range(self.partList.particleNum[None]):
            self.ParticleTranslation(np)
            self.SphereRotation(np)  

            # Todo: Periodic condition         


@ti.data_oriented
class Verlet(IntegrationScheme):
    def __init__(self, gravity, dt, partList, contModel):
        super().__init__(gravity, dt, partList, contModel)

    @ti.kernel
    def HalfStep(self):
        for np in range(self.partList.particleNum[None]):
            am_local = self.rotate[np] @ self.partList.Am[np]
            T_local = self.partList.rotate[np] @ self.partList.Tex[np]
            w_local = self.partList.inv_I[np] @ am_local
            d_am_local = T_local - w_local.cross(am_local)
            am_local += 0.5 * self.dt * d_am_local * Zero2OneVector(self.partList.fixedW[np])

            d_qt = 0.5 * self.partList.q[np] * (self.partList.inv_I[np].inverse() @ self.partList.Am[np])
            self.partList.q[np] += 0.5 * d_qt * self.dt
            self.partList.Am[np] += 0.5 * self.partList.Tex[np] * self.dt


    @ti.kernel
    def IntegrationPredictor(self):
        for np in range(self.partList.particleNum[None]):
            self.partList.x[np] += self.partList.v[np] * self.dt + 0.5 * self.partList.av[np] * self.dt * self.dt
            # Todo: Periodic condition

            self.partList.disp[np] += self.partList.v[np] * self.dt + 0.5 * self.partList.av[np] * self.dt * self.dt
            # Predictor: v = v + 0.5 * av0 * t ^ 2
            
            self.partList.theta[np] += self.partList.w[np] * self.dt + 0.5 * self.partList.aw[np] * self.dt * self.dt
            # Predictor: w = w + 0.5 * aw0 * t ^ 2

    
    @ti.kernel
    def SphereRotation(self):
        for np in range(self.partList.particleNum[None]):
            matID = self.partList.materialID[np]
            TorqueLocalDamping = self.contModel.TorqueLocalDamping[matID]

            aw0 = self.partList.aw[np]
            generalTorque = self.partList.Tex[np] + self.partList.Tc[np]
            torque =  (generalTorque - TorqueLocalDamping * generalTorque.norm() * Normalize(self.partList.w[np]))
            aw = torque * self.partList.inv_I[np] * Zero2OneVector(self.partList.fixedW[np])
            self.partList.w[np] += 0.5 * self.dt * (aw + aw0)
            self.partList.aw[np] = aw


    @ti.kernel
    def ClumpRotation(self):
        for np in range(self.partList.particleNum[None]):
            matID = self.partList.materialID[np]
            TorqueLocalDamping = self.contModel.TorqueLocalDamping[matID]
            generalTorque = self.partList.Tex[np] + self.partList.Tc[np]
            torque =  (generalTorque - TorqueLocalDamping * generalTorque.norm() * Normalize(self.partList.w[np]))

            Lt = self.partList.rotate[np] @ (self.partList.Am[np] + 0.5 * self.dt * generalTorque)
            Lmid = self.partList.rotate[np] @ (self.partList.Am[np] + self.dt * generalTorque)
            wt = self.partList.inv_I[np] @ Lt
            wmid = self.partList.inv_I[np] @ Lmid
            self.partList.w[np] = self.partList.rotate[np].inverse() @ wmid
            self.partList.Am[np] += self.dt * generalTorque * Zero2OneVector(self.partList.fixedW[np])

            self.UpdateQuanternion(np, wt, wmid)


    @ti.kernel
    def IntegrationCorrector(self):
        for np in range(self.partList.particleNum[None]):
            matID = self.partList.materialID[np]
            ForceLocalDamping = self.contModel.ForceLocalDamping[matID]
            av0 = self.partList.av[np]
            generalForce = self.partList.Fc[np] + self.partList.Fex[np] + self.gravity * self.partList.m[np]
            force = (generalForce - ForceLocalDamping * generalForce.norm() * Normalize(self.partList.v[np]))
            av = force / self.partList.m[np] * Zero2OneVector(self.partList.fixedV[np])
            self.partList.v[np] += 0.5 * self.dt * (av0 + av)
            self.partList.av[np] = av


@ti.data_oriented
class RungeKutta(IntegrationScheme):
    def __init__(self, gravity, dt, partList, contModel):
        super().__init__(gravity, dt, partList, contModel)