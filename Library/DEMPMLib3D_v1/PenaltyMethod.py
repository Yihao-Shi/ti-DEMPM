from DEMPMLib3D_v1.Function import *


@ti.data_oriented
class PenaltyMethod:
    def __init__(self, DEMcontModel, MPMmatList, dt):
        self.dt = dt
        self.DEMcontModel = DEMcontModel
        self.MPMmatList = MPMmatList
    
    @ti.kernel
    def InitDEM(self, DEMmatID: int, DEMcontInfo: ti.template()):
        print("Coupling Scheme: Penalty Method\n")

    @ti.kernel
    def InitMPM(self, MPMmatID: int, MPMcontInfo: ti.template()):
        print('MPM Contact Normal Stiffness = ', MPMcontInfo[MPMmatID].Kn)
        print('MPM Contact Tangential Stiffness = ', MPMcontInfo[MPMmatID].Ks)
        print('MPM Friction Coefficient = ', MPMcontInfo[MPMmatID].Mu, '\n')

        self.MPMmatList.kn[MPMmatID] = MPMcontInfo[MPMmatID].Kn
        self.MPMmatList.ks[MPMmatID] = MPMcontInfo[MPMmatID].Ks
        self.MPMmatList.Mu[MPMmatID] = MPMcontInfo[MPMmatID].Mu

    @ti.func
    def ComputeContactNormalForce(self, ContactPair, nc, matID1, matID2, gapn, norm):
        kn = EffectiveValue(self.DEMcontModel.kn[matID1], self.MPMmatList.kn[matID2])
        ContactPair.cnforce[nc] = kn * gapn * norm

    @ti.func
    def ComputeContactTangentialForce(self, ContactPair, nc, end1, end2, matID1, matID2, v_rel, norm, keyLoc):
        ks = EffectiveValue(self.DEMcontModel.ks[matID1], self.MPMmatList.ks[matID2])
        vs = v_rel - v_rel.dot(norm) * norm  
        trial_ft = -ks * vs * self.dt 

        if keyLoc >= 0:
            ft_ori = ContactPair.RelTranslate[keyLoc] - ContactPair.RelTranslate[keyLoc].dot(norm) * norm
            ft_temp = ContactPair.RelTranslate[keyLoc].norm() * Normalize(ft_ori)
            trial_ft = trial_ft + ft_temp
        
        miu = ti.min(self.DEMcontModel.Mu[matID1], self.MPMmatList.Mu[matID2])
        fric = miu * ContactPair.cnforce[nc].norm()
        if trial_ft.norm() > fric:
            ContactPair.ctforce[nc] = fric * trial_ft.normalized()
            #print(nc, end1, end2, keyLoc, ContactPair.cnforce[nc], ContactPair.ctforce[nc], ContactPair.cnforce[nc].norm(), ContactPair.ctforce[nc].norm(), 1, '\n')
        else:
            ContactPair.ctforce[nc] = trial_ft 
            #print(nc, end1, end2, keyLoc, ContactPair.cnforce[nc], ContactPair.ctforce[nc], ContactPair.cnforce[nc].norm(), ContactPair.ctforce[nc].norm(), 2)


