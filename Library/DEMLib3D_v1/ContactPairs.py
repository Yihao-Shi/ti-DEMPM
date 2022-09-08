import taichi as ti
import math
from DEMLib3D_v1.Function import *


@ti.data_oriented
class ContactPairs:
    def __init__(self, max_contact_num, partList, wallList, contModel):
        self.endID1 = ti.field(int, shape=(max_contact_num,))
        self.endID2 = ti.field(int, shape=(max_contact_num,))
        self.TYPE = ti.field(int, shape=(max_contact_num,))
        self.contactNum = ti.field(int, shape=())

        self.gapn = ti.field(float, shape=(max_contact_num,))
        self.norm = ti.Vector.field(3, float, shape=(max_contact_num,))
        self.cnforce = ti.Vector.field(3, float, shape=(max_contact_num,))
        self.ctforce = ti.Vector.field(3, float, shape=(max_contact_num,))
        self.Tr = ti.Vector.field(3, float, shape=(max_contact_num,))
        self.Tt = ti.Vector.field(3, float, shape=(max_contact_num,))

        self.key = ti.field(int, shape=(max_contact_num,))
        self.RelTranslate = ti.Vector.field(3, float, shape=(max_contact_num,))
        self.contactNum0 = ti.field(int, shape=())

        self.partList = partList
        self.wallList = wallList
        self.contModel = contModel

    @ti.func
    def HashValue(self, i, j):
        return int((i + j) * (i + j + 1) / 2. + j)

    @ti.func
    def ResetContactList(self, nc):
        self.endID1[nc] = -1
        self.endID2[nc] = -1
        self.TYPE[nc] = -1
        self.gapn[nc] = 0.
        self.cnforce[nc] = ti.Matrix.zero(float, 3)
        self.ctforce[nc] = ti.Matrix.zero(float, 3)
        self.Tr[nc] = ti.Matrix.zero(float, 3)
        self.Tt[nc] = ti.Matrix.zero(float, 3)
        self.norm[nc] = ti.Matrix.zero(float, 3)

    @ti.func
    def ResetFtIntegration(self, nc):
        if nc < self.contactNum0[None]:
            self.key[nc] = self.HashValue(self.TYPE[nc] * self.partList.particleNum[None] + self.endID1[nc], self.endID2[nc])
            self.RelTranslate[nc] = self.ctforce[nc]
        elif self.contactNum0[None] <= nc < self.contactNum[None]:
            self.key[nc] = -1
            self.RelTranslate[nc] = ti.Matrix.zero(float, 3)

    @ti.kernel
    def Reset(self):
        for nc in range(self.contactNum[None]):
            self.ResetFtIntegration(nc)
            self.ResetContactList(nc)

        self.contactNum0[None] = self.contactNum[None]
        self.contactNum[None] = 0
        
    @ti.func
    def Contact(self, end1, end2, pos1, pos2, rad1, rad2, TYPE):
        nc = ti.atomic_add(self.contactNum[None], 1)
        self.endID1[nc] = end1
        self.endID2[nc] = end2
        self.TYPE[nc] = TYPE
        self.gapn[nc] = rad1 + rad2 - (pos1 - pos2).norm()
        self.norm[nc] = (pos1 - pos2).normalized()
        cpos = pos2 + (rad2 - 0.5 * self.gapn[nc]) * self.norm[nc]
        matID1 = -1
        if TYPE == 0:
            matID1 = self.partList.materialID[end1]
        elif TYPE == 1:
            matID1 = self.wallList.materialID[end1]
        matID2 = self.partList.materialID[end2]
        self.ForceAssemble(nc, end1, end2, matID1, matID2, cpos)

    @ti.func
    def ForceAssemble(self, nc):
        pass 

        
@ti.data_oriented
class Linear(ContactPairs):
    def __init__(self, max_contact_num, partcleList, wallList, contModel):
        print("Contact Model: Linear Contact Model\n")
        super().__init__(max_contact_num, partcleList, wallList, contModel)
        self.cdnforce = ti.Vector.field(3, float, shape=(max_contact_num,))
        self.cdsforce = ti.Vector.field(3, float, shape=(max_contact_num,))

    @ti.func
    def ForceAssemble(self, nc, end1, end2, matID1, matID2, cpos):
        vel1, w1, pos1 = ti.Matrix.zero(float, 3), ti.Matrix.zero(float, 3), ti.Matrix.zero(float, 3)
        if self.TYPE[nc] == 0:
            vel1 = self.partList.v[end1]
            w1 = self.partList.w[end1]
            pos1 = self.partList.x[end1]
        elif self.TYPE[nc] == 1:
            vel1 = self.wallList.v[end1]
            w1 = self.wallList.w[end1]
            pos1 = (self.wallList.p1[end1] + self.wallList.p2[end1] + self.wallList.p3[end1] + self.wallList.p4[end1]) / 4.
        vel2 = self.partList.v[end2]
        w2 = self.partList.w[end2]
        pos2 = self.partList.x[end2]
        v_rel = vel1 + w1.cross(cpos - pos1) - (vel2 + w2.cross(cpos - pos2))

        m_eff = 0.
        if self.TYPE[nc] == 0:
            m_eff = EffectiveValue(self.partList.m[end1], self.partList.m[end2])
        elif self.TYPE[nc] == 1:
            m_eff = EffectiveValue(1e12, self.partList.m[end2])

        self.contModel.ComputeContactNormalForce(self, nc, matID1, matID2, m_eff, v_rel)
        self.contModel.ComputeContactTangentialForce(self, nc, matID1, matID2, m_eff, v_rel, self.partList.particleNum[None])

        Ftotal = self.cnforce[nc] + self.ctforce[nc] + self.cdnforce[nc] + self.cdsforce[nc]
        if self.TYPE[nc] == 0:
            self.partList.Fc[end1] += Ftotal
            self.partList.Tc[end1] += Ftotal.cross(self.partList.x[end1] - cpos) 
        elif self.TYPE[nc] == 1:
            center = (self.wallList.p1[end1] + self.wallList.p2[end1] + self.wallList.p3[end1] + self.wallList.p4[end1]) / 4.
            self.wallList.Fc[end1] += Ftotal
            self.wallList.Tc[end1] += Ftotal.cross(center - cpos)
        self.partList.Fc[end2] -= Ftotal
        self.partList.Tc[end2] -= Ftotal.cross(self.partList.x[end2] - cpos) 


@ti.data_oriented
class HertzMindlin(ContactPairs):
    def __init__(self, max_contact_num, partcleList, wallList, contModel): 
        print("Contact Model: Hertz-Mindlin Contact Model\n")
        super().__init__(max_contact_num, partcleList, wallList, contModel)

    @ti.func
    def ForceAssemble(self, nc, end1, end2, matID1, matID2, cpos):
        G1 = self.contModel.modulus[matID1]
        G2 = self.contModel.modulus[matID2]
        mu1 = self.contModel.Mu[matID1]
        mu2 = self.contModel.Mu[matID2]
        E1 = 2 * G1 * (1 + mu1)
        E2 = 2 * G2 * (1 + mu2)

        m_eff, rad_eff = 0., 0.
        vel1, w1, pos1 = ti.Matrix.zero(float, 3), ti.Matrix.zero(float, 3), ti.Matrix.zero(float, 3)
        if self.TYPE[nc] == 0:
            m_eff = EffectiveValue(self.partList.m[end1], self.partList.m[end2])
            rad_eff = EffectiveValue(self.partList.rad[end1], self.partList.rad[end2])
            vel1 = self.partList.v[end1]
            w1 = self.partList.w[end1]
            pos1 = self.partList.x[end1]
        elif self.TYPE[nc] == 1:
            m_eff = self.partList.m[end2]
            rad_eff = self.partList.rad[end2]
            vel1 = self.wallList.v[end1]
            w1 = self.wallList.w[end1]
            pos1 = (self.wallList.p1[end1] + self.wallList.p2[end1] + self.wallList.p3[end1] + self.wallList.p4[end1]) / 4.
        vel2 = self.partList.v[end2]
        w2 = self.partList.w[end2]
        pos2 = self.partList.x[end2]
        v_rel = vel1 + w1.cross(cpos - pos1) - (vel2 + w2.cross(cpos - pos2))

        self.contModel.ComputeContactNormalForce(self, nc, matID1, matID2, E1, E2, mu1, mu2, m_eff, rad_eff, v_rel)
        self.contModel.ComputeContactTangentialForce(self, nc, matID1, matID2, E1, E2, mu1, mu2, m_eff, rad_eff, v_rel, self.partList.particleNum[None])

        Ftotal = self.cnforce[nc] + self.ctforce[nc]
        if self.TYPE[nc] == 0:
            self.partList.Fc[end1] += Ftotal
            self.partList.Tc[end1] += Ftotal.cross(self.partList.x[end1] - cpos) 
        elif self.TYPE[nc] == 1:
            center = (self.wallList.p1[end1] + self.wallList.p2[end1] + self.wallList.p3[end1] + self.wallList.p4[end1]) / 4.
            self.wallList.Fc[end1] += Ftotal
            self.wallList.Tc[end1] += Ftotal.cross(center - cpos)
        self.partList.Fc[end2] -= Ftotal
        self.partList.Tc[end2] -= Ftotal.cross(self.partList.x[end2] - cpos) 


@ti.data_oriented
class LinearRollingResistance(Linear):
    def __init__(self, max_contact_num, partcleList, wallList, contModel):
        print("Contact Model: Linear Rolling Resistance Contact Model\n")
        super().__init__(max_contact_num, partcleList, wallList, contModel)
        self.Tr = ti.Vector.field(3, float, shape=(max_contact_num,))
        self.Tt = ti.Vector.field(3, float, shape=(max_contact_num,))
        self.RelRolling = ti.Vector.field(3, float, shape=(max_contact_num,))
        self.RelTwist = ti.field(float, shape=(max_contact_num,))

    @ti.func
    def ResetFtIntegration(self, nc):
        if nc < self.contactNum0[None]:
            self.key[nc] = self.HashValue(self.TYPE[nc] * self.partList.particleNum[None] + self.endID1[nc], self.endID2[nc])
            self.RelTranslate[nc] = self.ctforce[nc]
            self.RelRolling[nc] = self.Tr[nc]
            self.RelTwist[nc] = self.Tt[nc].norm()
        elif self.contactNum0[None] <= nc < self.contactNum[None]:
            self.key[nc] = -1
            self.RelTranslate[nc] = ti.Matrix.zero(float, 3)
            self.RelRolling[nc] = ti.Matrix.zero(float, 3)
            self.RelTwist[nc] = 0.

    @ti.func
    def ForceAssemble(self, nc, end1, end2, matID1, matID2, cpos):
        vel1, w1, pos1 = ti.Matrix.zero(float, 3), ti.Matrix.zero(float, 3), ti.Matrix.zero(float, 3)
        if self.TYPE[nc] == 0:
            vel1 = self.partList.v[end1]
            w1 = self.partList.w[end1]
            pos1 = self.partList.x[end1]
        elif self.TYPE[nc] == 1:
            vel1 = self.wallList.v[end1]
            w1 = self.wallList.w[end1]
            pos1 = (self.wallList.p1[end1] + self.wallList.p2[end1] + self.wallList.p3[end1] + self.wallList.p4[end1]) / 4.
        vel2 = self.partList.v[end2]
        w2 = self.partList.w[end2]
        pos2 = self.partList.x[end2]
        v_rel = vel1 + w1.cross(cpos - pos1) - (vel2 + w2.cross(cpos - pos2))

        m_eff, rad_eff = 0., 0.
        if self.TYPE[nc] == 0:
            m_eff = EffectiveValue(self.partList.m[end1], self.partList.m[end2])
            rad_eff = EffectiveValue(self.partList.rad[end1], self.partList.rad[end2])
        elif self.TYPE[nc] == 1:
            m_eff = self.partList.m[end2]
            rad_eff = self.partList.rad[end2]

        self.contModel.ComputeContactNormalForce(self, nc, matID1, matID2, m_eff, v_rel)
        self.contModel.ComputeContactTangentialForce(self, nc, matID1, matID2, m_eff, v_rel, self.partList.particleNum[None])
        self.contModel.ComputeRollingFriction(self, nc, matID1, matID2, w1, w2, m_eff, rad_eff, self.partList.particleNum[None])
        self.contModel.ComputeTorsionFriction(self, nc, matID1, matID2, w1, w2, m_eff, rad_eff, self.partList.particleNum[None])

        Ftotal = self.cnforce[nc] + self.ctforce[nc]
        Ttotal = self.Tr[nc] + self.Tt[nc]
        if self.TYPE[nc] == 0:
            self.partList.Fc[end1] += Ftotal
            self.partList.Tc[end1] += Ftotal.cross(self.partList.x[end1] - cpos) + Ttotal
        elif self.TYPE[nc] == 1:
            center = (self.wallList.p1[end1] + self.wallList.p2[end1] + self.wallList.p3[end1] + self.wallList.p4[end1]) / 4.
            self.wallList.Fc[end1] += Ftotal
            self.wallList.Tc[end1] += Ftotal.cross(center - cpos) + Ttotal
        self.partList.Fc[end2] -= Ftotal
        self.partList.Tc[end2] -= Ftotal.cross(self.partList.x[end2] - cpos) + Ttotal


@ti.data_oriented
class LinearContactBond(Linear):
    def __init__(self, max_contact_num, partcleList, wallList, contModel):  
        print("Contact Model: Linear Bond Contact Model\n")
        super().__init__(max_contact_num, partcleList, wallList, contModel)


@ti.data_oriented
class LinearParallelBond(Linear):
    def __init__(self, max_contact_num, partcleList, wallList, contModel):      
        print("Contact Model: Linear Parallel Bond Contact Model\n") 
        super().__init__(max_contact_num, partcleList, wallList, contModel)     
