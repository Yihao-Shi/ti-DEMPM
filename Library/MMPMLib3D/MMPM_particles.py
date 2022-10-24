import taichi as ti
from Common.Function import *
import MPMLib2D.MPM_particles as MPMParticle
import MPMLib2D.Shape_functions as ShFunc
import math


@ti.data_oriented
class ParticleList(MPMParticle.ParticleList):
    def __init__(self, dx, domain, max_particle_num):
        super().__init__(dx, domain, max_particle_num)
        self.poros = ti.field(float, max_particle_num)
        self.Se = ti.field(float, max_particle_num)
        self.kh = ti.field(float, max_particle_num)

    def StoreInitialPoros(self, max_particle_num):
        self.poros0 = ti.field(float, max_particle_num)
        self.kr = ti.field(float, max_particle_num)
        self.C1 = ti.field(float, max_particle_num)

    def StoreHydroProperties(self, max_particle_num):
        self.Solidvel = ti.Vector.field(2, float, max_particle_num)
        self.SolidStrainRate = ti.Matrix.field(2, 2, float, max_particle_num)
        self.Sr = ti.field(float, max_particle_num)
    
    # ====================================================== MPM Particle Initialization ========================================================== #
    @ti.kernel
    def SolidPhaseInit(self, bodyInfo: ti.template()):
        for np in range(self.particleNum[None]):
            bodyID = self.bodyID[np]
            self.kh[np] = bodyInfo[bodyID].permeability
            self.poros0[np] = self.poros[np] = bodyInfo[bodyID].porosity
            self.C1[np] = self.kh[np] * (1 - self.poros[np]) ** 2 / self.poros[np] ** 3
            self.m[np] *= (1 - self.poros[np])

    @ti.kernel
    def FluidPhaseInit(self, bodyInfo: ti.template()):
        for np in range(self.particleNum[None]):
            bodyID = self.bodyID[np]
            self.Sr[np] = bodyInfo[bodyID].saturation

    def SetMaxPoros(self, poros_max):
        self.poros_max = poros_max

    
    # ============================================= Solve ======================================================= #
    # ========================================================= #
    #                                                           #
    #              Compute GIMP Shape Functions                 #
    #                                                           #
    # ========================================================= #
    @ti.func
    def ComputeGIMP3(self, np, activeID, nodeID, xg):
        SF, GS = ShFunc.GIMP3(self.x[np], xg, self.Dx, self.pSize[np])
        if SF > 0.:
            self.LnID[np, activeID] = nodeID
            self.LnShape[np, activeID] = SF
            self.LnDShape[np, activeID] = GS
            self.ComputeBMatrix(np, activeID, GS)

    @ti.func
    def UpdateGIMP(self, np, activeID, nodeID, xg, stablization, phase):
        if phase == 0:
            if stablization == 2: 
                self.ComputeGIMP2(np, activeID, nodeID, xg)
            else: 
                self.ComputeGIMP1(np, activeID, nodeID, xg)
        elif phase == 1: 
            self.ComputeGIMP3(np, activeID, nodeID, xg)
        
    # ========================================================= #
    #                                                           #
    #        Calculate Porosity at water material point         #
    #                                                           #
    # ========================================================= #
    @ti.func
    def PorosityTransfer(self, np, gridList):
        self.poros[np] = 0.
        for ln in range(self.LnID.shape[1]):
            if self.LnID[np, ln] >= 0:
                nodeID = self.LnID[np, ln]
                SF = self.LnShape[np, ln]
                self.poros[np] += SF * gridList.poros[nodeID]
        self.poros[np] = Clamp(0.00001, 0.99999, self.poros[np])

    @ti.func
    def PermeabilityTransfer(self, np, gridList):
        self.kh[np] = 0.
        for ln in range(self.LnID.shape[1]):
            if self.LnID[np, ln] >= 0:
                nodeID = self.LnID[np, ln]
                SF = self.LnShape[np, ln]
                self.kh[np] += SF * gridList.kh[nodeID]

    @ti.func
    def ComputePermeability(self, np, gravity):
        return self.m[np] * self.poros[np] * gravity.norm()

    # ========================================================= #
    #                                                           #
    #             Calculate velocity gradient                   #
    #                                                           #
    # ========================================================= #
    @ti.func
    def CalLocalDvFluid(self, gridList, np):
        temp = ti.Matrix.zero(float, 2, 2)
        for ln in range(self.LnID.shape[1]):
            if self.LnID[np, ln] >= 0:
                nodeID = self.LnID[np, ln]
                vw = gridList.vw[nodeID]
                GS = self.LnDShape[np, ln]

                temp[0, 0] += vw[0] * GS[0]
                temp[0, 1] += vw[0] * GS[1]
                temp[1, 0] += vw[1] * GS[0]
                temp[1, 1] += vw[1] * GS[1]
        self.gradv[np] = temp

    # ========================================================= #
    #                                                           #
    #        Update strain increment and strain rate            #
    #                                                           #
    # ========================================================= #
    @ti.func
    def UpdateStrainFluid(self, np, gridList):
        temp = ti.Matrix.zero(float, 2, 2)
        for ln in range(self.LnID.shape[1]):
            if self.LnID[np, ln] >= 0:
                nodeID = self.LnID[np, ln]
                gv = gridList.vw[nodeID]
                bmatrix = self.Bmatrix[np, ln]
                temp[0, 0] += bmatrix[0, 0] * gv[0] + bmatrix[0, 1] * gv[1]
                temp[0, 1] += 0.5 * (bmatrix[2, 0] * gv[0] + bmatrix[2, 1] * gv[1])
                temp[1, 0] += 0.5 * (bmatrix[2, 0] * gv[0] + bmatrix[2, 1] * gv[1])
                temp[1, 1] += bmatrix[1, 0] * gv[0] + bmatrix[1, 1] * gv[1]
        self.StrainRate[np] = temp

    @ti.func
    def UpdateSolidStrainFluid(self, np, gridList):
        temp = ti.Matrix.zero(float, 2, 2)
        for ln in range(self.LnID.shape[1]):
            if self.LnID[np, ln] >= 0:
                nodeID = self.LnID[np, ln]
                gv = gridList.v[nodeID]
                bmatrix = self.Bmatrix[np, ln]
                temp[0, 0] += (bmatrix[0, 0] * gv[0] + bmatrix[0, 1] * gv[1])
                temp[0, 1] += 0.5 * (bmatrix[2, 0] * gv[0] + bmatrix[2, 1] * gv[1]) 
                temp[1, 0] += 0.5 * (bmatrix[2, 0] * gv[0] + bmatrix[2, 1] * gv[1])
                temp[1, 1] += (bmatrix[1, 0] * gv[0] + bmatrix[1, 1] * gv[1])
        self.SolidStrainRate[np] = temp
    
    # ========================================================= #
    #                                                           #
    #                   Update Pore Pressure                    #
    #                                                           #
    # ========================================================= #
    @ti.func
    def ComputePorePressure(self, np, dt, threshold, matList):
        matID = self.materialID[np]
        self.P[np] -= matList.k[matID] * dt / self.poros[np] * (self.poros[np] * self.StrainRate[np].trace() + (1 - self.poros[np]) * self.SolidStrainRate[np].trace())
        if self.poros[np] < self.poros_max:
            self.stress[np] = self.P[np] * ti.Matrix.identity(float, 2)
        elif self.poros[np] > self.poros_max and self.P[np] > 0:
            self.stress[np] = self.P[np] * ti.Matrix.identity(float, 2) + 2 * matList.vis[matID] * (self.StrainRate[np] - self.StrainRate[np].trace() * ti.Matrix.identity(float, 2) / 2.)

    # ========================================================= #
    #                                                           #
    #                Update Particle Properity                  #
    #                                                           #
    # ========================================================= #
    @ti.func
    def UpdatePartProperties(self, np):
        self.vol[np] = self.J[np] * self.vol0[np]
        self.rad[np] = (self.vol[np] / math.pi) ** 0.5
        self.poros[np] = 1 - (1 - self.poros0[np]) / self.J[np]
        self.kh[np] = self.C1[np] * self.poros[np] ** 3 / (1 - self.poros[np]) ** 2

    @ti.func
    def UpdatePartPropertiesFluid(self, np):
        self.vol[np] = self.J[np] * self.vol0[np]
        self.rad[np] = (self.vol[np] / math.pi) ** 0.5
        self.m[np] = self.poros[np] * self.rho[np] * self.vol[np]

    # ============================================= Transfer Algorithm ======================================================= #
    # ========================================================= #
    #                                                           #
    #                        PIC & FLIP                         #
    #                                                           #
    # ========================================================= #
    @ti.func
    def LinearPICFLIPFluid(self, np, gridList, alphaPIC, dt):
        vPIC, vFLIP = ti.Matrix.zero(float, 2, 1), self.v[np]
        pos = self.x[np]
        for ln in range(self.LnID.shape[1]):
            if self.LnID[np, ln] >= 0:
                nodeID = self.LnID[np, ln]
                SF = self.LnShape[np, ln]
                vPIC += SF * gridList.vw[nodeID]
                vFLIP += SF * gridList.fw[nodeID] / gridList.mw[nodeID] * dt
                pos += SF * gridList.vw[nodeID] * dt
        vel = alphaPIC * vPIC + (1 - alphaPIC) * vFLIP
        self.v[np] = vel * Zero2OneVector(self.fixVel[np]) + vFLIP * self.fixVel[np]
        self.x[np] = pos * Zero2OneVector(self.fixVel[np]) + vFLIP * dt * self.fixVel[np]
        if isnan(self.x[np]) or self.x[np][0] < 0. or self.x[np][0] > self.Domain[0] or self.x[np][1] < 0. or self.x[np][1] > self.Domain[1]: 
            print("ERROR FLUID POSITION: Particle ID:", self.ID[np], self.x[np])
            #assert 0
