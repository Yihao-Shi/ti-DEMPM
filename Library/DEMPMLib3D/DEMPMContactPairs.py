import taichi as ti
from Common import HashMap
from Common.Function import *


@ti.data_oriented
class DEMPMContactPair:
    def __init__(self, max_dempm_contact_num, DEMpartList, MPMpartList, DEMPMcontModel):
        self.contactNum = ti.field(int, shape=())
        self.endID1 = ti.field(int, shape=(max_dempm_contact_num,))
        self.endID2 = ti.field(int, shape=(max_dempm_contact_num,))
        self.cnforce = ti.Vector.field(3, float, shape=(max_dempm_contact_num,))
        self.ctforce = ti.Vector.field(3, float, shape=(max_dempm_contact_num,))

        self.TangForceMap = HashMap.HASHMAP(max_dempm_contact_num, 3, type=1)

        self.DEMpartList = DEMpartList
        self.MPMpartList = MPMpartList
        self.DEMPMcontModel = DEMPMcontModel
    
    @ti.func
    def ResetContactList(self, nc):
        self.endID1[nc] = -1
        self.endID2[nc] = -1
        self.cnforce[nc] = ti.Matrix.zero(float, 3)
        self.ctforce[nc] = ti.Matrix.zero(float, 3)

    @ti.func
    def copyHistory(self, nc):
        key = PairingFunction(self.endID1[nc], self.DEMpartList.particleNum[None] + self.endID2[nc])
        self.TangForceMap.Insert(key, self.ctforce[nc])

    @ti.kernel
    def Reset(self):
        for nc in self.TangForceMap.Key:
            self.TangForceMap.MapInit(nc)
        for nc in range(self.contactNum[None]):
            self.copyHistory(nc)
            #self.ResetContactList(nc)

        self.contactNum[None] = 0

    @ti.func
    def HistTangInfo(self, nc):
        key = PairingFunction(self.endID1[nc], self.DEMpartList.particleNum[None] + self.endID2[nc])
        return self.TangForceMap.Search(key)
        
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
        v_rel = vel1 + w1.cross(cpos - pos1) - vel2 

        keyLoc = self.HistTangInfo(nc)
        self.DEMPMcontModel.ComputeContactNormalForce(self, nc, matID1, matID2, gapn, norm)
        self.DEMPMcontModel.ComputeContactTangentialForce(self, nc, matID1, matID2, v_rel, norm, keyLoc)

        Ftotal = self.cnforce[nc] + self.ctforce[nc]
        self.DEMpartList.Fc[end1] += Ftotal
        self.DEMpartList.Tc[end1] += Ftotal.cross(self.DEMpartList.x[end1] - cpos) 
        self.MPMpartList.fc[end2] -= Ftotal

