import taichi as ti
import math


@ti.data_oriented
class MaterialList:
    def __init__(self, max_material_num):
        self.max_material_num = int(max_material_num)
        self.matType = ti.field(int, max_material_num)                               # Material type
        self.young = ti.field(float, max_material_num)                               # Young's modulus
        self.Ep = ti.field(float, max_material_num)                                  # Plastic Modulus
        self.possion = ti.field(float, max_material_num)                             # Possion ratio
        self.g = ti.field(float, max_material_num)                                   # Shear modulus / Viscosity parameters(miu)
        self.la = ti.field(float, max_material_num)                                  # Lame's first parameter
        self.k = ti.field(float, max_material_num)                                   # Lame's second parameter / Viscosity parameters(lamda)
        self.h = ti.field(float, max_material_num)                                   # Linear hardening modulus
        self.c = ti.field(float, max_material_num)                                   # Cohesion coefficient
        self.fai = ti.field(float, max_material_num)                                 # Angle of internal friction
        self.psi = ti.field(float, max_material_num)                                 # Angle of dilatation
        self.lodeT = ti.field(float, max_material_num)                               # Transition angle
        self.tensile = ti.field(float, max_material_num)                             # Tension stress
        self.yield0 = ti.field(float, max_material_num)                              # Yield stress
        self.q_fai = ti.field(float, max_material_num)                               # Drucker-Prager parameter
        self.k_fai = ti.field(float, max_material_num)                               # Drucker-Prager parameter
        self.q_psi = ti.field(float, max_material_num)                               # Drucker-Prager parameter
        self.vis = ti.field(float, max_material_num)                                 # Viscosity
        self.soundSpeed = ti.field(float, max_material_num)                          # Sound speed
        self.rhop = ti.field(float, max_material_num)                                # Particle density
        self.miu_s = ti.field(float, max_material_num)                               # Rheology parameters
        self.miu_2 = ti.field(float, max_material_num)                               # Rheology parameters
        self.I0 = ti.field(float, max_material_num)                                  # Rheology parameters
        self.diamp = ti.field(float, max_material_num)                               # Particle diatmeter

    def DEMPMPenaltyMethod(self):
        self.kn = ti.field(float, shape=(self.max_material_num,))
        self.ks = ti.field(float, shape=(self.max_material_num,))
        self.Mu = ti.field(float, shape=(self.max_material_num,))
        
    def DEMPMBarrierMethod(self):
        self.scalar = ti.field(float, shape=(self.max_material_num,))
        self.slipLim = ti.field(float, shape=(self.max_material_num,))
        self.Mu = ti.field(float, shape=(self.max_material_num,))

    # Set Material Parameters
    @ti.kernel
    def ElasticModel(self, nm: int, matInfo: ti.template()):
        print('Constitutive model: Elastic Model')
        print('Young Modulus = ', matInfo[nm].Modulus)
        print('Possion Ratio = ', matInfo[nm].mu)
        print('Hardening = ', matInfo[nm].h, '\n')

        self.young[nm] = matInfo[nm].Modulus
        self.possion[nm] = matInfo[nm].mu
        self.h[nm] = matInfo[nm].h
        self.g[nm] = 0.5 * matInfo[nm].Modulus / (1. + matInfo[nm].mu)
        self.la[nm] = matInfo[nm].Modulus * matInfo[nm].mu / (1. + matInfo[nm].mu) / (1. - 2. * matInfo[nm].mu)
        self.k[nm] = self.la[nm] + 2. / 3. * self.g[nm]

    @ti.kernel
    def MohrCoulombModel(self, nm: int, matInfo: ti.template()):
        print('Constitutive model = Mohr-Coulomb Model')
        print('Young Modulus = ', matInfo[nm].Modulus)
        print('Possion Ratio = ', matInfo[nm].mu)
        print('Hardening = ', matInfo[nm].h)
        print('Cohesion Coefficient = ', matInfo[nm].Cohesion)
        print('Angle of Internal Friction = ', matInfo[nm].InternalFric)
        print('Angle of Dilatation = ', matInfo[nm].Dilation, '\n')

        self.young[nm] = matInfo[nm].Modulus
        self.possion[nm] = matInfo[nm].mu
        self.h[nm] = matInfo[nm].h
        self.g[nm] = 0.5 * matInfo[nm].Modulus / (1. + matInfo[nm].mu)
        self.la[nm] = matInfo[nm].Modulus * matInfo[nm].mu / (1. + matInfo[nm].mu) / (1. - 2. * matInfo[nm].mu)
        self.k[nm] = self.la[nm] + 2. / 3. * self.g[nm]
        self.c[nm] = matInfo[nm].Cohesion
        self.fai[nm] = matInfo[nm].InternalFric
        self.psi[nm] = matInfo[nm].Dilation
        self.lodeT[nm] = matInfo[nm].lodeT

    @ti.kernel
    def DruckerPragerModel(self, nm: int, matInfo: ti.template()):
        print('Constitutive model = Drucker-Prager Model')
        print('Young Modulus = ', matInfo[nm].Modulus)
        print('Possion Ratio = ', matInfo[nm].mu)
        print('Hardening = ', matInfo[nm].h)
        print('Cohesion Coefficient = ', matInfo[nm].Cohesion)
        print('Angle of Internal Friction = ', matInfo[nm].InternalFric)
        print('Angle of Dilatation = ', matInfo[nm].Dilation, '\n')

        self.young[nm] = matInfo[nm].Modulus
        self.possion[nm] = matInfo[nm].mu
        self.h[nm] = matInfo[nm].h
        self.g[nm] = 0.5 * matInfo[nm].Modulus / (1. + matInfo[nm].mu)
        self.la[nm] = matInfo[nm].Modulus * matInfo[nm].mu / (1. + matInfo[nm].mu) / (1. - 2. * matInfo[nm].mu)
        self.k[nm] = self.la[nm] + 2. / 3. * self.g[nm]
        self.c[nm] = matInfo[nm].Cohesion
        self.fai[nm] = matInfo[nm].InternalFric
        self.psi[nm] = matInfo[nm].Dilation

        if matInfo[nm].dpType == 0:
            self.q_fai[nm] = 6. * ti.sin(matInfo[nm].InternalFric) / ti.sqrt(3) * (3 + ti.sin(matInfo[nm].InternalFric))
            self.k_fai[nm] = 6. * ti.cos(matInfo[nm].InternalFric) * matInfo[nm].Cohesion / ti.sqrt(3) * (3 + ti.sin(matInfo[nm].InternalFric))
            self.q_psi[nm] = 6. * ti.sin(matInfo[nm].Dilation) / ti.sqrt(3) * (3 + ti.sin(matInfo[nm].Dilation))
        elif matInfo[nm].dpType == 1:
            self.q_fai[nm] = 6. * ti.sin(matInfo[nm].InternalFric) / ti.sqrt(3) * (3 - ti.sin(matInfo[nm].InternalFric))
            self.k_fai[nm] = 6. * ti.cos(matInfo[nm].InternalFric) * matInfo[nm].Cohesion / ti.sqrt(3) * (3 - ti.sin(matInfo[nm].InternalFric))
            self.q_psi[nm] = 6. * ti.sin(matInfo[nm].Dilation) / ti.sqrt(3) * (3 - ti.sin(matInfo[nm].Dilation))
        elif matInfo[nm].dpType == 2:
            self.q_fai[nm] = 3. * ti.tan(matInfo[nm].InternalFric) / ti.sqrt(9. + 12 * ti.tan(matInfo[nm].InternalFric) ** 2)
            self.k_fai[nm] = 3. * matInfo[nm].Cohesion / ti.sqrt(9. + 12 * ti.tan(matInfo[nm].InternalFric) ** 2)
            self.q_psi[nm] = 3. * ti.tan(matInfo[nm].Dilation) / ti.sqrt(9. + 12 * ti.tan(matInfo[nm].Dilation) ** 2)

        if matInfo[nm].InternalFric == 0:
            self.tensile[nm] = 0.
        else:
            self.tensile[nm] = ti.min(self.tensile[nm], self.k_fai[nm] / self.q_fai[nm])

    @ti.kernel
    def NewtonianModel(self, nm: int, matInfo: ti.template()):
        print('Constitutive model = Newtonian Model')
        print('Bulk Modulus = ', matInfo[nm].Modulus)
        print('Viscosity = ', matInfo[nm].Viscosity)
        print('Speed of sound = ', matInfo[nm].SoundSpeed)

        self.k[nm] = matInfo[nm].Modulus
        self.vis[nm] = matInfo[nm].Viscosity
        self.soundSpeed[nm] = matInfo[nm].SoundSpeed

    @ti.kernel
    def ViscoplasticModel(self, nm: int, matInfo: ti.template()):
        print('Constitutive model = Viscoplastic Model')
        print('Miu_s = ', matInfo[nm].miu_s)
        print('Miu_2 = ', matInfo[nm].miu_2)
        print('I0 = ', matInfo[nm].I0)
        print('Particle density = ', matInfo[nm].ParticleRho)
        print('Particle Diameter = ', matInfo[nm].ParticleDiameter)

        self.miu_s[nm] = matInfo[nm].miu_s
        self.miu_2[nm] = matInfo[nm].miu_2
        self.I0[nm] = matInfo[nm].I0
        self.rhop[nm] = matInfo[nm].ParticleRho
        self.diamp[nm] = matInfo[nm].ParticleDiameter

    @ti.kernel
    def ElasticViscoplasticModel(self, nm: int, matInfo: ti.template()):
        print('Constitutive model = Elastic-Viscoplastic Model')
        print('Young Modulus = ', matInfo[nm].Modulus)
        print('Possion Ratio = ', matInfo[nm].mu)
        print('Hardening = ', matInfo[nm].h)
        print('Cohesion Coefficient = ', matInfo[nm].Cohesion)
        print('Angle of Internal Friction = ', matInfo[nm].InternalFric)
        print('Miu_s = ', matInfo[nm].miu_s)
        print('Miu_2 = ', matInfo[nm].miu_2)
        print('I0 = ', matInfo[nm].I0)
        print('Particle density = ', matInfo[nm].ParticleRho)
        print('Particle Diameter = ', matInfo[nm].ParticleDiameter)

        self.young[nm] = matInfo[nm].Modulus
        self.possion[nm] = matInfo[nm].mu
        self.h[nm] = matInfo[nm].h
        self.g[nm] = 0.5 * matInfo[nm].Modulus / (1. + matInfo[nm].mu)
        self.la[nm] = matInfo[nm].Modulus * matInfo[nm].mu / (1. + matInfo[nm].mu) / (1. - 2. * matInfo[nm].mu)
        self.k[nm] = self.la[nm] + 2. / 3. * self.g[nm]
        self.c[nm] = matInfo[nm].Cohesion
        self.fai[nm] = matInfo[nm].InternalFric
        self.miu_s[nm] = matInfo[nm].miu_s
        self.miu_2[nm] = matInfo[nm].miu_2
        self.I0[nm] = matInfo[nm].I0
        self.rhop[nm] = matInfo[nm].ParticleRho
        self.diamp[nm] = matInfo[nm].ParticleDiameter

        self.q_fai[nm] = 3. * ti.tan(matInfo[nm].InternalFric) / ti.sqrt(9. + 12 * ti.tan(matInfo[nm].InternalFric) ** 2)
        self.k_fai[nm] = 3. * matInfo[nm].Cohesion / ti.sqrt(9. + 12 * ti.tan(matInfo[nm].InternalFric) ** 2)
