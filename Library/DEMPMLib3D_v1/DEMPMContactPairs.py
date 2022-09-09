import taichi as ti
import math
from DEMPMLib3D_v1.Function import *


@ti.data_oriented
class DEMPMContactPair:
    def __init__(self, max_dempm_contact_num, DEMpartList, MPMpartList, DEMPMcontModel):
        self.contactNum = ti.field(int, shape=())
        self.endID1 = ti.field(int, shape=(max_dempm_contact_num,))
        self.endID2 = ti.field(int, shape=(max_dempm_contact_num,))
        self.cnforce = ti.Vector.field(3, float, shape=(max_dempm_contact_num,))
        self.ctforce = ti.Vector.field(3, float, shape=(max_dempm_contact_num,))
        self.key = ti.field(int, shape=(max_dempm_contact_num,))
        self.RelTranslate = ti.Vector.field(3, float, shape=(max_dempm_contact_num,))
        self.contactNum0 = ti.field(int, shape=())

        self.DEMpartList = DEMpartList
        self.MPMpartList = MPMpartList
        self.DEMPMcontModel = DEMPMcontModel

    @ti.func
    def HashValue(self, i, j):
        return int((i + j) * (i + j + 1) / 2. + j)
    
    @ti.func
    def ResetContactList(self, nc):
        self.endID1[nc] = -1
        self.endID2[nc] = -1
        self.cnforce[nc] = ti.Matrix.zero(float, 3)
        self.ctforce[nc] = ti.Matrix.zero(float, 3)

    @ti.func
    def ResetFtIntegration(self, nc):
        if nc < self.contactNum0[None]:
            self.key[nc] = self.HashValue(self.endID1[nc], self.DEMpartList.particleNum[None] + self.endID2[nc])
            self.RelTranslate[nc] = self.ctforce[nc]
        elif self.contactNum0[None] <= nc < self.contactNum[None]:
            self.key[nc] = -1
            self.RelTranslate[nc] = ti.Matrix.zero(float, 3)

    @ti.kernel
    def Reset(self):
        for nc in range(self.contactNum[None]):
            self.ResetContactList(nc)
            self.ResetFtIntegration(nc)

        self.contactNum0[None] = self.contactNum[None]
        self.contactNum[None] = 0
        
    @ti.func
    def Contact(self, end1, end2, pos1, pos2, rad1, rad2):
        nc = ti.atomic_add(self.contactNum[None], 1)
        self.endID1[nc] = end1
        self.endID2[nc] = end2
        gapn = rad1 + rad2 - (pos1 - pos2).norm()
        norm = (pos1 - pos2).normalized()
        cpos = pos2 + (rad2 - 0.5 * gapn) * norm
        matID1 = self.DEMpartList.materialID[end1]
        matID2 = self.MPMpartList.materialID[end2]
        self.ForceAssemble(nc, end1, end2, matID1, matID2, gapn, norm, cpos)

    @ti.func
    def ForceAssemble(self, nc, end1, end2, matID1, matID2, gapn, norm, cpos):
        vel1 = self.DEMpartList.v[end1]
        w1 = self.DEMpartList.w[end1]
        pos1 = self.DEMpartList.x[end1]
        vel2 = self.MPMpartList.v[end2]
        pos2 = self.MPMpartList.x[end2]
        v_rel = vel1 + w1.cross(cpos - pos1) - vel2 

        m_eff = EffectiveValue(self.DEMpartList.m[end1], self.MPMpartList.m[end2])

        self.DEMPMcontModel.ComputeContactNormalForce(self, nc, matID1, matID2, gapn, norm)
        self.DEMPMcontModel.ComputeContactTangentialForce(self, nc, end1, end2, matID1, matID2, v_rel, norm, self.DEMpartList.particleNum[None])

        Ftotal = self.cnforce[nc] + self.ctforce[nc]
        self.DEMpartList.Fc[end1] += Ftotal
        self.DEMpartList.Tc[end1] += Ftotal.cross(self.DEMpartList.x[end1] - cpos) 
        self.MPMpartList.fc[end2] -= Ftotal

