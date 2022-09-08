from DEMPMLib3D_v1.Function import *


@ti.data_oriented
class PenaltyMethod:
    def __init__(self, DEMcontModel, MPMmatList, dt):
        self.dt = dt
        self.DEMcontModel = DEMcontModel
        self.MPMmatList = MPMmatList
    
    @ti.kernel
    def InitDEM(self, DEMmatID: int, DEMcontInfo: ti.template()):
        print("Coupling Scheme: Penalty Method")

    @ti.kernel
    def InitMPM(self, MPMmatID: int, MPMcontInfo: ti.template()):
        print('MPM Contact Normal Stiffness = ', MPMcontInfo[MPMmatID].Kn)
        print('MPM Contact Tangential Stiffness = ', MPMcontInfo[MPMmatID].Ks)
        print('MPM Friction Coefficient = ', MPMcontInfo[MPMmatID].Mu, '\n')

        self.MPMmatList.kn[MPMmatID] = MPMcontInfo[MPMmatID].Kn
        self.MPMmatList.ks[MPMmatID] = MPMcontInfo[MPMmatID].Ks
        self.MPMmatList.Mu[MPMmatID] = MPMcontInfo[MPMmatID].Mu

    @ti.func
    def ComputeContactNormalForce(self, matID1, matID2, gapn, norm):
        kn = EffectiveValue(self.DEMcontModel.kn[matID1], self.MPMmatList.kn[matID2])
        cnforce = kn * gapn * norm
        cdnforce = ti.Matrix.zero(float, 3)
        return cnforce, cdnforce

    @ti.func
    def ComputeContactTangentialForce(self, ContactPair, end1, end2, matID1, matID2, v_rel, norm, cnforce, particle_num):
        ks = EffectiveValue(self.DEMcontModel.ks[matID1], self.MPMmatList.ks[matID2])
        vs = v_rel - v_rel.dot(norm) * norm  
        trial_ft = -ks * vs * self.dt 
        key  = HashValue(end1, particle_num + end2)
        for i in range(ContactPair.contactNum0[None]):
            if ContactPair.key[i] == key:
                ft_ori = ContactPair.RelTranslate[i] - ContactPair.RelTranslate[i].dot(norm) * norm
                ft_temp = ContactPair.RelTranslate[i].norm() * Normalize(ft_ori)
                trial_ft = trial_ft + ft_temp
        
        ctforce = trial_ft 
        miu = ti.min(self.DEMcontModel.Mu[matID1], self.MPMmatList.Mu[matID2])
        fric = miu * cnforce.norm()
        if trial_ft.norm() > fric:
            ctforce = fric * trial_ft.normalized()

        cdsforce = ti.Matrix.zero(float, 3)
        return ctforce, cdsforce
