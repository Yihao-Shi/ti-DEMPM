from DEMPMLib3D_v1.Function import *


@ti.data_oriented
class BarrierMethod:
    def __init__(self, DEMpartList, MPMpartList, DEMcontModel, MPMmatList, dt):
        self.dt = dt
        self.DEMpartList = DEMpartList
        self.MPMpartList = MPMpartList
        self.DEMcontModel = DEMcontModel
        self.MPMmatList = MPMmatList

    @ti.kernel
    def InitDEM(self, DEMmatID: int, DEMcontInfo: ti.template()):
        print("Coupling Scheme: Barrier Method")
        print('DEM Barrier Scalar: = ', DEMcontInfo[DEMmatID].scalar)
        print('DEM Maximum Microslip Displacement: = ', DEMcontInfo[DEMmatID].slipLim)

        self.DEMcontModel.scalar[DEMmatID] = DEMcontInfo[DEMmatID].scalar
        self.DEMcontModel.slipLim[DEMmatID] = DEMcontInfo[DEMmatID].slipLim

    @ti.kernel
    def InitMPM(self, MPMmatID: int, MPMcontInfo: ti.template()):
        print('MPM Barrier Scalar = ', MPMcontInfo[MPMmatID].scalar)
        print('MPM Maximum Microslip Displacement = ', MPMcontInfo[MPMmatID].slipLim)
        print('MPM Friction Coefficient = ', MPMcontInfo[MPMmatID].Mu, '\n')

        self.MPMmatList.scalar[MPMmatID] = MPMcontInfo[MPMmatID].scalar
        self.MPMmatList.slipLim[MPMmatID] = MPMcontInfo[MPMmatID].slipLim
        self.MPMmatList.Mu[MPMmatID] = MPMcontInfo[MPMmatID].Mu

    @ti.func
    def ComputeContactNormalForce(self, matID1, matID2, gapn, norm):
        scalar = EffectiveValue(self.DEMcontModel.scalar[matID1], self.MPMmatList.scalar[matID2])
        rad = ti.min(self.DEMpartList.rad[end1], self.MPMpartList.rad[end2])
        cnforce = scalar * (gapn_- rad) * (2 * ti.log(gapn / rad) - rad - gapn + 1) 
        cdnforce = ti.Matrix.zero(float, 3)
        return cnforce, cdnforce

    @ti.func
    def ComputeContactTangentialForce(self, ContactPair, end1, end2, matID1, matID2, v_rel, norm, cnforce, particle_num):
        slipLim = ti.min(self.DEMcontModel.slipLim[matID1], self.MPMmatList.slipLim[matID2])
        vs = v_rel - v_rel.dot(norm) * norm  
        trial_t = vs * self.dt 
        key  = HashValue(end1, particle_num + end2)
        for i in range(ContactPair.contactNum0[None]):
            if ContactPair.key[i] == key:
                t_ori = ContactPair.RelTranslate[i] - ContactPair.RelTranslate[i].dot(norm) * norm
                t_temp = ContactPair.RelTranslate[i].norm() * Normalize(t_ori)
                trial_t = trial_t + ft_temp
        
        miu = ti.min(self.DEMcontModel.Mu[matID1], self.MPMmatList.Mu[matID2])
        fric = miu * cnforce.norm()
        sslip = 1
        if trial_t > slipLim:
            sslip = -(trial_t.norm() / slipLim) ** 2 + 2 * (trial_t.norm() / slipLim)
        ctforce = sslip * fric * Normalize(vs)

        cdsforce = ti.Matrix.zero(float, 3)
        return ctforce, cdsforce
