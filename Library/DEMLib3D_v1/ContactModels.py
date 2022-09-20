import taichi as ti
import math
from DEMLib3D_v1.Function import *


@ti.data_oriented
class ContactModel:
    def __init__(self, max_material_num, dt):
        self.dt = dt
        self.max_material_num = max_material_num
        self.rho = ti.field(float, shape=(max_material_num,))
        self.ForceLocalDamping = ti.field(float, shape=(max_material_num,))
        self.TorqueLocalDamping = ti.field(float, shape=(max_material_num,))
        self.Mu = ti.field(float, shape=(max_material_num,))

    def UpdateDt(self, dt):
        self.dt = dt

    def DEMPMBarrierMethod(self):
        self.scalar = ti.field(float, shape=(self.max_material_num,))
        self.slipLim = ti.field(float, shape=(self.max_material_num,))

@ti.data_oriented
class LinearContactModel(ContactModel):
    def __init__(self, max_material_num, dt):
        super().__init__(max_material_num, dt)
        self.NormalViscousDamping = ti.field(float, shape=(max_material_num,))
        self.TangViscousDamping = ti.field(float, shape=(max_material_num,))
        self.kn = ti.field(float, shape=(max_material_num,))
        self.ks = ti.field(float, shape=(max_material_num,))

    def ResetMaterialProperty(self, keyword, modified_num):
        pass

    @ti.kernel
    def ParticleMaterialInit(self, matID: int, MatInfo: ti.template()):
        print('Contact model: Linear contact Model')
        print('Contact normal stiffness: = ', MatInfo[matID].Kn)
        print('Contact tangential stiffness: = ', MatInfo[matID].Ks)
        print('Friction coefficient = ', MatInfo[matID].Mu)
        print('Local damping coefficient = ', MatInfo[matID].ForceLocalDamping)
        print('Local damping coefficient = ', MatInfo[matID].TorqueLocalDamping)
        print('Viscous damping coefficient = ', MatInfo[matID].NormalViscousDamping)
        print('Local damping coefficient = ', MatInfo[matID].TangViscousDamping)
        print('Particle density = ', MatInfo[matID].ParticleRho, '\n')

        self.rho[matID] = MatInfo[matID].ParticleRho
        self.ForceLocalDamping[matID] = MatInfo[matID].ForceLocalDamping
        self.TorqueLocalDamping[matID] = MatInfo[matID].TorqueLocalDamping
        self.NormalViscousDamping[matID] = MatInfo[matID].NormalViscousDamping
        self.TangViscousDamping[matID] = MatInfo[matID].TangViscousDamping
        self.kn[matID] = MatInfo[matID].Kn
        self.ks[matID] = MatInfo[matID].Ks
        self.Mu[matID] = MatInfo[matID].Mu

    @ti.func
    def ComputeContactNormalForce(self, ContactPair, nc, matID1, matID2, m_eff, v_rel):
        kn = EffectiveValue(self.kn[matID1], self.kn[matID2])
        ContactPair.cnforce[nc] = kn * ContactPair.gapn[nc] * ContactPair.norm[nc] 

        vn = v_rel.dot(ContactPair.norm[nc]) * ContactPair.norm[nc]
        ndratio = ti.min(self.NormalViscousDamping[matID1], self.NormalViscousDamping[matID2])
        ContactPair.cdnforce[nc] = -2 * ndratio * ti.sqrt(m_eff * kn) * vn

    @ti.func
    def ComputeContactTangentialForce(self, ContactPair, nc, matID1, matID2, m_eff, v_rel, particle_num):
        ks = EffectiveValue(self.ks[matID1], self.ks[matID2])
        vs = v_rel - v_rel.dot(ContactPair.norm[nc]) * ContactPair.norm[nc]  
        trial_ft = -ks * vs * self.dt 
        key = HashValue(ContactPair.TYPE[nc] * particle_num + ContactPair.endID1[nc], ContactPair.endID2[nc])
        for i in range(ContactPair.contactNum0[None]):
            if ContactPair.key[i] == key:
                ft_ori = ContactPair.RelTranslate[i] - ContactPair.RelTranslate[i].dot(ContactPair.norm[nc]) * ContactPair.norm[nc]
                ft_temp = ContactPair.RelTranslate[i].norm() * Normalize(ft_ori)
                trial_ft = trial_ft + ft_temp
                break
        
        miu = ti.min(self.Mu[matID1], self.Mu[matID2])
        fric = miu * ContactPair.cnforce[nc].norm()
        if trial_ft.norm() > fric:
            ContactPair.ctforce[nc] = fric * trial_ft.normalized()
        else:
            ContactPair.ctforce[nc] = trial_ft 

        nsratio = ti.min(self.TangViscousDamping[matID1], self.TangViscousDamping[matID2])
        ContactPair.cdsforce[nc] = -2 * nsratio * ti.sqrt(m_eff * ks) * vs


@ti.data_oriented
class HertzMindlinContactModel(ContactModel):
    def __init__(self, max_material_num, dt):
        super().__init__(max_material_num, dt)
        self.modulus = ti.field(float, shape=(max_material_num,))
        self.possion = ti.field(float, shape=(max_material_num,))
        self.Restitution = ti.field(float, shape=(max_material_num,))

    @ti.kernel
    def ParticleMaterialInit(self, matID: int, MatInfo: ti.template()):
        print('Contact model: Hertz contact Model')
        print('Shear modulus: = ', MatInfo[matID].Modulus)
        print('Possion ratio: = ', MatInfo[matID].possion)
        print('Friction coefficient = ', MatInfo[matID].Mu)
        print('Local damping coefficient = ', MatInfo[matID].ForceLocalDamping)
        print('Local damping coefficient = ', MatInfo[matID].TorqueLocalDamping)
        print('Restitution = ', MatInfo[matID].Restitution)
        print('Particle density = ', MatInfo[matID].ParticleRho, '\n')

        self.rho[matID] = MatInfo[matID].ParticleRho
        self.ForceLocalDamping[matID] = MatInfo[matID].ForceLocalDamping
        self.TorqueLocalDamping[matID] = MatInfo[matID].TorqueLocalDamping
        self.modulus[matID] = MatInfo[matID].Modulus
        self.possion[matID] = MatInfo[matID].possion
        self.Restitution[matID] = MatInfo[matID].Restitution
        self.Mu[matID] = MatInfo[matID].Mu

    @ti.func
    def DampingRatio(self, matID1, matID2):
        restitution = ti.min(self.Restitution[matID1], self.Restitution[matID2])
        res = 0.
        if restitution == 0:
            res = 0.
        else:
            res = ti.log(restitution) / ti.sqrt(math.pi * math.pi + ti.log(restitution) * ti.log(restitution))
        return res

    @ti.func
    def ComputeContactNormalForce(self, ContactPair, nc, matID1, matID2, E1, E2, mu1, mu2, m_eff, rad_eff, v_rel):
        E = 1. / ((1 - mu1 * mu1) / E1 + (1 - mu2 * mu2) / E2)
        kn = 2 * E * ti.sqrt(ContactPair.gapn[nc] * rad_eff)
        res = self.DampingRatio(matID1, matID2)
        ContactPair.cnforce[nc] = 2./3. * kn * ContactPair.gapn[nc] * ContactPair.norm[nc] + 1.8257 * res * v_rel.norm() * ti.sqrt(kn * m_eff) * sgn(v_rel.dot(ContactPair.norm[nc])) * ContactPair.norm[nc] 

    @ti.func
    def ComputeContactTangentialForce(self, ContactPair, nc, matID1, matID2, G1, G2, mu1, mu2, m_eff, rad_eff, v_rel, particle_num):
        res = self.DampingRatio(matID1, matID2)
        G = 1. / ((2 - mu1) / G1 + (2 - mu2) / G2)
        ks = 8 * G * ti.sqrt(ContactPair.gapn[nc] * rad_eff)
        vs = v_rel - v_rel.dot(ContactPair.norm[nc]) * ContactPair.norm[nc]  
        trial_ft = -ks * vs * self.dt + 1.8257 * res * vs * ti.sqrt(ks * m_eff)
        key = HashValue(ContactPair.TYPE[nc] * particle_num + ContactPair.endID1[nc], ContactPair.endID2[nc])
        for i in range(ContactPair.contactNum0[None]):
            if ContactPair.key[i] == key:
                ft_ori = ContactPair.RelTranslate[i] - ContactPair.RelTranslate[i].dot(ContactPair.norm[nc]) * ContactPair.norm[nc]
                ft_temp = ContactPair.RelTranslate[i].norm() * Normalize(ft_ori)
                trial_ft = trial_ft + ft_temp
                break
        
        miu = ti.min(self.Mu[matID1], self.Mu[matID2])
        fric = miu * ContactPair.cnforce[nc].norm()
        if trial_ft.norm() > fric:
            ContactPair.ctforce[nc] = fric * trial_ft.normalized()
        else:
            ContactPair.ctforce[nc] = trial_ft - 1.8257 * res * vs * ti.sqrt(ks * m_eff)


@ti.data_oriented
class LinearRollingResistanceContactModel(LinearContactModel):
    def __init__(self, max_material_num, dt):
        super().__init__(max_material_num, dt)
        self.kr = ti.field(float, shape=(max_material_num,))
        self.kt = ti.field(float, shape=(max_material_num,))
        self.RMu = ti.field(float, shape=(max_material_num,))
        self.TMu = ti.field(float, shape=(max_material_num,))

    @ti.kernel
    def ParticleMaterialInit(self, matID: int, MatInfo: ti.template()):
        print('Contact model: Linear Rolling Resistance Contact Model')
        print('Contact normal stiffness: = ', MatInfo[matID].Kn)
        print('Contact tangential stiffness: = ', MatInfo[matID].Ks)
        print('Contact rolling stiffness: = ', MatInfo[matID].Kr)
        print('Contact twisting stiffness: = ', MatInfo[matID].Kt)
        print('Friction coefficient = ', MatInfo[matID].Mu)
        print('Rolling Friction coefficient = ', MatInfo[matID].Rmu)
        print('Twisting Friction coefficient = ', MatInfo[matID].Tmu)
        print('Local damping coefficient = ', MatInfo[matID].ForceLocalDamping)
        print('Local damping coefficient = ', MatInfo[matID].TorqueLocalDamping)
        print('Viscous damping coefficient = ', MatInfo[matID].NormalViscousDamping)
        print('Local damping coefficient = ', MatInfo[matID].TangViscousDamping)
        print('Particle density = ', MatInfo[matID].ParticleRho, '\n')

        self.rho[matID] = MatInfo[matID].ParticleRho
        self.ForceLocalDamping[matID] = MatInfo[matID].ForceLocalDamping
        self.TorqueLocalDamping[matID] = MatInfo[matID].TorqueLocalDamping
        self.NormalViscousDamping[matID] = MatInfo[matID].NormalViscousDamping
        self.TangViscousDamping[matID] = MatInfo[matID].TangViscousDamping
        self.kn[matID] = MatInfo[matID].Kn
        self.ks[matID] = MatInfo[matID].Ks
        self.kr[matID] = MatInfo[matID].Kr
        self.kt[matID] = MatInfo[matID].Kt
        self.Mu[matID] = MatInfo[matID].Mu
        self.RMu[matID] = MatInfo[matID].Rmu
        self.TMu[matID] = MatInfo[matID].Tmu
    
    # Luding (2008) Introduction to discrete element method
    @ti.func
    def ComputeRollingFriction(self, ContactPair, nc, matID1, matID2, w1, w2, m_eff, rad_eff, particle_num):
        vr = -rad_eff * ContactPair.norm[nc].cross(w1 - w2)
        kr = EffectiveValue(self.kr[matID1], self.kr[matID2])
        trial_fr = -kr * vr * self.dt 
        key = HashValue(ContactPair.TYPE[nc] * particle_num + ContactPair.endID1[nc], ContactPair.endID2[nc])
        for i in range(ContactPair.contactNum0[None]):
            if ContactPair.key[i] == key:
                fr_pre = ContactPair.RelRolling[i] - ContactPair.RelRolling[i].dot(ContactPair.norm[nc]) * ContactPair.norm[nc]
                fr_temp = ContactPair.RelRolling[i].norm() * Normalize(fr_pre)
                trial_fr = trial_fr + fr_temp
                break
        
        rmiu = ti.min(self.RMu[matID1], self.RMu[matID2])
        fricRoll = rmiu * ContactPair.cnforce[nc].norm()
        if trial_fr.norm() > fricRoll:
            ContactPair.Fr[nc] = fricRoll * trial_fr.normalized()
        else:
            ContactPair.Fr[nc] = trial_fr
            

    # J. S. Marshall (2009) Discrete-element modeling of particulate aerosol flows /JCP/
    @ti.func
    def ComputeTorsionFriction(self, ContactPair, nc, matID1, matID2, w1, w2, m_eff, rad_eff, particle_num):
        vt = rad_eff * (w1 - w2).dot(ContactPair.norm[nc]) * ContactPair.norm[nc]
        kt = EffectiveValue(self.kt[matID1], self.kt[matID2])
        trial_ft = -kt * vt * self.dt
        key = HashValue(ContactPair.TYPE[nc] * particle_num + ContactPair.endID1[nc], ContactPair.endID2[nc])
        for i in range(ContactPair.contactNum0[None]):
            if ContactPair.key[i] == key:
                ft_pre = ContactPair.RelTwist[i].dot(ContactPair.norm[nc]) * ContactPair.norm[nc]
                ft_temp = ContactPair.RelTwist[i].norm() * Normalize(ft_pre)
                trial_ft = trial_ft + ft_temp
                break
        
        tmiu = ti.min(self.TMu[matID1], self.TMu[matID2])
        fricTwist = tmiu * ContactPair.cnforce[nc].norm()
        if trial_ft.norm() > fricTwist:
            ContactPair.Ft[nc] = fricTwist * trial_ft.normalized()
        else:
            ContactPair.Ft[nc] = trial_ft


@ti.data_oriented
class LinearBondContactModel(LinearContactModel):
    def __init__(self, max_material_num, dt):
        super().__init__(max_material_num, dt)

    @ti.kernel
    def ParticleMaterialInit(self, matID: int, MatInfo: ti.template()):
        print('Contact model: Linear Bond Model')
        print('Contact normal stiffness: = ', MatInfo[matID].Kn)
        print('Contact tangential stiffness: = ', MatInfo[matID].Ks)
        print('Friction coefficient = ', MatInfo[matID].Mu)
        print('Local damping coefficient = ', MatInfo[matID].ForceLocalDamping)
        print('Local damping coefficient = ', MatInfo[matID].TorqueLocalDamping)
        print('Viscous damping coefficient = ', MatInfo[matID].NormalViscousDamping)
        print('Local damping coefficient = ', MatInfo[matID].TangViscousDamping)
        print('Particle density = ', MatInfo[matID].ParticleRho, '\n')

        self.rho[matID] = MatInfo[matID].ParticleRho
        self.ForceLocalDamping[matID] = MatInfo[matID].ForceLocalDamping
        self.TorqueLocalDamping[matID] = MatInfo[matID].TorqueLocalDamping
        self.NormalViscousDamping[matID] = MatInfo[matID].NormalViscousDamping
        self.TangViscousDamping[matID] = MatInfo[matID].TangViscousDamping
        self.modulus[matID] = MatInfo[matID].Modulus
        self.possion[matID] = MatInfo[matID].possion
        self.kn[matID] = MatInfo[matID].Kn
        self.ks[matID] = MatInfo[matID].Ks
        self.Mu[matID] = MatInfo[matID].Mu
        self.Rmu[matID] = MatInfo[matID].Rmu

    
@ti.data_oriented
class LinearParallelBondContactModel(LinearContactModel):
    def __init__(self, max_material_num, dt):
        super().__init__(max_material_num, dt)

    @ti.kernel
    def ParticleMaterialInit(self, matID: int, MatInfo: ti.template()):
        print('Contact model: Linear Parallel Bond Model')
        print('Contact normal stiffness: = ', MatInfo[matID].Kn)
        print('Contact tangential stiffness: = ', MatInfo[matID].Ks)
        print('Friction coefficient = ', MatInfo[matID].Mu)
        print('Local damping coefficient = ', MatInfo[matID].ForceLocalDamping)
        print('Local damping coefficient = ', MatInfo[matID].TorqueLocalDamping)
        print('Viscous damping coefficient = ', MatInfo[matID].NormalViscousDamping)
        print('Local damping coefficient = ', MatInfo[matID].TangViscousDamping)
        print('Particle density = ', MatInfo[matID].ParticleRho, '\n')

        self.rho[matID] = MatInfo[matID].ParticleRho
        self.ForceLocalDamping[matID] = MatInfo[matID].ForceLocalDamping
        self.TorqueLocalDamping[matID] = MatInfo[matID].TorqueLocalDamping
        self.NormalViscousDamping[matID] = MatInfo[matID].NormalViscousDamping
        self.TangViscousDamping[matID] = MatInfo[matID].TangViscousDamping
        self.modulus[matID] = MatInfo[matID].Modulus
        self.possion[matID] = MatInfo[matID].possion
        self.kn[matID] = MatInfo[matID].Kn
        self.ks[matID] = MatInfo[matID].Ks
        self.Mu[matID] = MatInfo[matID].Mu
        self.Rmu[matID] = MatInfo[matID].Rmu
