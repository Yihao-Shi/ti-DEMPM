import taichi as ti
from Common.Function import *
import MPMLib2D.ShapeFunctions as ShFunc
import math


@ti.data_oriented
class ParticleList:
    # =============================================== Set Up Taichi Field ============================================================ #
    def __init__(self, dx, domain, max_particle_num):
        self.particleNum = ti.field(int, shape=())                                   # Total particle Number
        self.max_particle_num = max_particle_num                                     # Max particle number
        self.Dx = dx
        self.Domain = domain

        self.ID = ti.field(int, max_particle_num)                                    # Particle ID
        self.bodyID = ti.field(int, max_particle_num)                                # Body ID
        self.materialID = ti.field(int, max_particle_num)                            # Material ID
        self.vol0 = ti.field(float, max_particle_num)                                # Initial volume
        self.vol = ti.field(float, max_particle_num)                                 # Volume
        self.m = ti.field(float, max_particle_num)                                   # Particle mass
        self.rho0 = ti.field(float, max_particle_num)                                # Initial density of granular particles
        self.rho = ti.field(float, max_particle_num)                                 # Density of granular particles
        self.rad = ti.field(float, max_particle_num)                                 # Radius of granular particles
        self.P = ti.field(float, max_particle_num)                                   # Pressure of fluid
        self.J = ti.field(float, max_particle_num)                                   # Volume ratio
        self.cs = ti.field(float, max_particle_num)                                  # Material sound speed

        self.x = ti.Vector.field(2, float, max_particle_num)                         # Position
        self.v = ti.Vector.field(2, float, max_particle_num)                         # Velocity
        self.fixVel = ti.Vector.field(2, int, max_particle_num)                      # fix velocity
        self.fc = ti.Vector.field(2, float, max_particle_num)                        # Contact force

        self.strain = ti.Matrix.field(2, 2, float, max_particle_num)                 # Strain tensor
        self.StrainRate = ti.Matrix.field(2, 2, float, max_particle_num)             # Shear strain rate
        self.plasticStrainEff = ti.field(float, max_particle_num)                    # Norm of deviatoric plastic strain tensor
        self.stress = ti.Matrix.field(2, 2, float, max_particle_num)                 # Stress tensor
        self.gradv = ti.Matrix.field(2, 2, float, max_particle_num)                  # Velocity gradient tensor
        self.td = ti.Matrix.field(2, 2, float, max_particle_num)                     # Deformation gradient tensor
    
    def StoreShapeFuncs(self, max_particle_num, BasisType):
        self.LnID = ti.field(int, (max_particle_num, BasisType))                     # List of node index
        self.LnShape = ti.field(float, (max_particle_num, BasisType))                # List of shape function
        self.LnDShape = ti.Vector.field(2, float, (max_particle_num, BasisType))  # List of gradient of shape function
        self.Bmatrix = ti.Matrix.field(3, 2, float, (max_particle_num, BasisType))   # List of B Matrix

    def StoreParticleLen(self, max_particle_num):
        self.pSize0 = ti.Vector.field(2, float, max_particle_num)                    # Initial half length of particle domain for GIMP
        self.pSize = ti.Vector.field(2, float, max_particle_num)                     # Half length of particle domain for GIMP

    def MPMCell(self, cellList):
        self.cellID = ti.field(int, self.max_particle_num)                           # The cell ID of mpm particles
        self.PartcleInCellInit(cellList)
    
    # ====================================== MPM Particle Initialization ====================================== #
    @ti.func
    def ParticleInit(self, np, npic, nb, bodyInfo, pos):
        self.ID[np] = np
        self.bodyID[np] = bodyInfo[nb].ID
        self.materialID[np] = bodyInfo[nb].Mat
        self.rho0[np] = bodyInfo[nb].MacroRho
        self.rho[np] = self.rho0[np]
        self.vol0[np] = (0.5 * (self.Dx[0] + self.Dx[1]) / npic) ** 2
        self.vol[np] = self.vol0[np]
        self.m[np] = self.vol0[np] * bodyInfo[nb].MacroRho
        self.rad[np] = (self.vol0[np] / math.pi) ** 0.5

        self.x[np] = pos
        self.v[np] = bodyInfo[nb].v0
        self.fixVel[np] = bodyInfo[nb].fixedV

        self.td[np] = ti.Matrix.identity(float, 2)
        self.J[np] = 1.

    @ti.kernel
    def GIMPInit(self, npic: int):
        for np in range(self.particleNum[None]):
            self.pSize0[np] = 0.5 * self.Dx / npic
            self.pSize[np] = self.pSize0[np]

    @ti.kernel
    def PartcleInCellInit(self, cellList: ti.template()):
        for np in range(self.particleNum[None]):
            self.UpdateParticleInCell(np, cellList)
            cellID = self.cellID[np]

    # ============================================ Reset ====================================================== #
    @ti.func
    def ResetShapeFuncs(self, np):
        for i in range(self.LnID.shape[1]):
            self.LnID[np, i] = -1
            self.LnShape[np, i] = 0.
            self.LnDShape[np, i] = ti.Matrix.zero(float, 2)
            self.Bmatrix[np, i] = ti.Matrix.zero(float, 3, 2)

    @ti.func
    def ResetParticleForce(self, np):
        self.fc[np] = ti.Matrix.zero(float, 2)

    @ti.func
    def AdaptiveParticle(self, np):
        alpha = 0.5
        partID = ti.atomic_add(self.particleNum[None], 1)
        for d in ti.static(range(2)):
            if self.pSize[np][d] > alpha * self.Dx[d]:
                self.m[np] /= 2
                self.vol0[np] /= 2
                self.vol[np] /= 2
                self.rad[np] /= ti.sqrt(2)
                self.pSize[np][d] /= 2
                self.pSize0[np][d] /= 2
                self.strain[np][0, 0] = ti.sqrt(2)/2. * (1 + self.strain[np][0, 0]) - 1
                self.strain[np][1, 1] = ti.sqrt(2) * (1 + self.strain[np][1, 1]) - 1

                self.ID[partID] = self.ID[np]
                self.bodyID[partID] = self.bodyID[np]
                self.materialID[partID] = self.materialID[np]
                self.m[partID] = self.m[np]
                self.vol0[partID] = self.vol0[np]
                self.vol[partID] = self.vol[np]
                self.rad[partID] = self.rad[np]
                self.P[partID] = self.P[np]
                self.cs[partID] = self.cs[np]
                self.rho[partID] = self.rho[np]
                self.v[partID] = self.v[np]
                self.fixVel[partID] = self.fixVel[np]
                self.td[partID] = self.td[np]
                self.stress[partID] = self.stress[np]
                self.strain[partID] = self.strain[np]
                self.plasticStrainEff[partID] = self.plasticStrainEff[np]
                for i in range(self.LnID.shape[1]):
                    self.LnID[partID, i] = self.LnID[np, i]
                    self.LnShape[partID, i] = self.LnShape[np, i]
                    self.Bmatrix[partID, i] = self.Bmatrix[np, i]

                xmax = ti.min(self.x[np][d] + 0.25 * self.pSize[np][d], self.Domain[d])
                xmin = ti.max(self.x[np][d] - 0.25 * self.pSize[np][d], 0.)
                if d == 0:
                    self.x[partID] = [xmax, self.x[np][1]]
                    self.x[np] = [xmin, self.x[np][1]]
                else:
                    self.x[partID] = [self.x[np][0], xmax]
                    self.x[np] = [self.x[np][0], xmin]

    # ======================================== Add Particles ================================================= #
    # ========================================================= #
    #                                                           #
    #                     Add Rectangle                         #
    #                                                           #
    # ========================================================= #
    @ti.kernel
    def AddRectangle(self, nb: int, bodyInfo: ti.template(), matList: ti.template(), npic: int):
        pnum = (bodyInfo[nb].len * npic / self.Dx).cast(int)
        new_particle_num = pnum[0] * pnum[1]
        assert self.particleNum[None] + new_particle_num <= self.max_particle_num
        print("Body Shape: Rectangle")
        print("Material ID = ", bodyInfo[nb].Mat)
        print("Macro Density = ", bodyInfo[nb].MacroRho)
        print("Initial Velocity = ", bodyInfo[nb].v0)
        print("Fixed Velocity = ", bodyInfo[nb].fixedV)
        print("Particle Number = ", pnum, '\n')
        for np in range(new_particle_num):
            ip = (np % (pnum[0] * pnum[1])) % pnum[0]
            jp = (np % (pnum[0] * pnum[1])) // pnum[0]
            pos = (ti.Vector([ip, jp]) + 0.5) * self.Dx / npic + bodyInfo[nb].pos0
            self.ParticleInit(self.particleNum[None] + np, npic, nb, bodyInfo, pos)
            self.UpdateSoundSpeed(matList, self.particleNum[None] + np)
        self.particleNum[None] += new_particle_num
    
    # ========================================================= #
    #                                                           #
    #                     Add Triangle                          #
    #                                                           #
    # ========================================================= #
    @ti.func
    def RegionTriangle(self, xpos, ypos, x0, x1, l0, l1):
        return l1 * xpos + l0 * ypos - (x0 * l1 + x1 * l0 + l0 * l1)

    @ti.kernel
    def AddTriangle(self, nb: int, bodyInfo: ti.template(), matList: ti.template(), npic: int, dx: ti.template()):
        pnum = (bodyInfo[nb].len * npic / self.Dx).cast(int)
        new_particle_num_trial = pnum[0] * pnum[1]
        assert self.particleNum[None] + 0.5 * new_particle_num_trial <= self.max_particle_num
        print("Body Shape: Triangle")
        print("Material ID = ", bodyInfo[nb].Mat)
        print("Macro Density = ", bodyInfo[nb].MacroRho)
        print("Initial Velocity = ", bodyInfo[nb].v0)
        print("Fixed Velocity = ", bodyInfo[nb].fixedV)
        print("Particle Number = ", pnum, '\n')
        for np in range(new_particle_num_trial):
            ip = (np % (pnum[0] * pnum[1])) % pnum[0]
            jp = (np % (pnum[0] * pnum[1])) // pnum[0]
            pos = (ti.Vector([ip, jp]) + 0.5) * self.Dx / npic + bodyInfo[nb].pos0
            if self.RegionTriangle(pos[0], pos[1], bodyInfo[nb].pos0[0], bodyInfo[nb].pos0[1], bodyInfo[nb].len[0], bodyInfo[nb].len[1]) < 0:
                temp = ti.atomic_add(self.particleNum[None], 1)
                self.ParticleInit(temp, npic, nb, bodyInfo, pos)
                self.UpdateSoundSpeed(matList, temp)
    
    # ========================================================= #
    #                                                           #
    #                       Add Disk                            #
    #                                                           #
    # ========================================================= #
    @ti.func
    def RegionCircle(self, xpos, ypos, x0, x1, rad):
        return (xpos - x0) ** 2 + (ypos - x1) ** 2 - rad ** 2

    @ti.kernel
    def AddCircle(self, nb: int, bodyInfo: ti.template(), matList: ti.template(), npic: int, dx: ti.template()):
        pnum = (bodyInfo[nb].len * npic / self.Dx).cast(int)
        new_particle_num_trial = pnum[0] * pnum[1]
        assert self.particleNum[None] + new_particle_num_trial <= self.max_particle_num
        print("Body Shape: Circle")
        print("Material ID = ", bodyInfo[nb].Mat)
        print("Macro Density = ", bodyInfo[nb].MacroRho)
        print("Initial Velocity = ", bodyInfo[nb].v0)
        print("Fixed Velocity = ", bodyInfo[nb].fixedV)
        print("Particle Number = ", pnum, '\n')
        for np in range(new_particle_num_trial):
            ip = (np % (pnum[0] * pnum[1])) % pnum[0]
            jp = (np % (pnum[0] * pnum[1])) // pnum[0]
            pos = (ti.Vector([ip, jp]) + 0.5) * self.Dx / npic + bodyInfo[nb].pos0
            if self.RegionCircle(pos[0], pos[1], bodyInfo[nb].pos0[0], bodyInfo[nb].pos0[1], bodyInfo[nb].rad) < 0:
                temp = ti.atomic_add(self.particleNum[None], 1)
                self.ParticleInit(temp, npic, nb, bodyInfo, pos)
                self.UpdateSoundSpeed(matList, temp)

    # ============================================= Solve ======================================================= #
    # ========================================================= #
    #                                                           #
    #                 Information Exchange                      #
    #                                                           #
    # ========================================================= #
    @ti.func
    def Projection(self, np, ln, val):
        return self.LnShape[np, ln] * val

    # ========================================================= #
    #                                                           #
    #                   Compute B Matrix                        #
    #                                                           #
    # ========================================================= #
    @ti.func
    def ComputeBMatrix(self, np, activeID, GS):
        self.Bmatrix[np, activeID][0, 0] = GS[0]
        self.Bmatrix[np, activeID][1, 1] = GS[1]
        self.Bmatrix[np, activeID][2, 0] = GS[1]
        self.Bmatrix[np, activeID][2, 1] = GS[0]
    
    # ========================================================= #
    #                                                           #
    #               Anti-Locking (B-Bar Method)                 #
    #                                                           #
    # ========================================================= #
    @ti.func
    def BbarMethod(self, np, activeID, GS, GSC):
        self.Bmatrix[np, activeID][0, 0] += (GSC[0] - GS[0]) / 2.
        self.Bmatrix[np, activeID][0, 1] += (GSC[1] - GS[1]) / 2.
        self.Bmatrix[np, activeID][1, 0] += (GSC[0] - GS[0]) / 2.
        self.Bmatrix[np, activeID][1, 1] += (GSC[1] - GS[1]) / 2.
    
    # ========================================================= #
    #                                                           #
    #              Compute GIMP Shape Functions                 #
    #                                                           #
    # ========================================================= #
    @ti.func
    def ComputeGIMP1(self, np, activeID, nodeID, xg):
        SF, GS = ShFunc.GIMP1(self.x[np], xg, self.Dx, self.pSize[np])
        if SF > 0.:
            self.LnID[np, activeID] = nodeID
            self.LnShape[np, activeID] = SF
            self.LnDShape[np, activeID] = GS
            self.ComputeBMatrix(np, activeID, GS)

    @ti.func
    def ComputeGIMP2(self, np, activeID, nodeID, xg):
        SF, GS, GSC = ShFunc.GIMP2(self.x[np], xg, self.Dx, self.pSize[np])
        if SF > 0.:
            self.LnID[np, activeID] = nodeID
            self.LnShape[np, activeID] = SF
            self.LnDShape[np, activeID] = GS
            self.ComputeBMatrix(np, activeID, GS)
            self.BbarMethod(np, activeID, GS, GSC)

    @ti.func
    def UpdateGIMP(self, np, activeID, nodeID, xg, stablization):
        if stablization == 2: self.ComputeGIMP2(np, activeID, nodeID, xg)
        else: self.ComputeGIMP1(np, activeID, nodeID, xg)

    # ========================================================= #
    #                                                           #
    #                 Compute Shape Functions                   #
    #                                                           #
    # ========================================================= #
    @ti.func
    def ComputeShapeFuncs1(self, sType, np, activeID, nodeID, xg):
        SF, GS = 0., ti.Matrix.zero(float, 2)
        if sType == 0: SF, GS = ShFunc.Linear1(self.x[np], xg, self.Dx)
        elif sType == 2: SF, GS = ShFunc.BsplineQ1(self.x[np], xg, self.Dx, self.Domain)
        elif sType == 3: SF, GS = ShFunc.BsplineC1(self.x[np], xg, self.Dx, self.Domain)
        if SF > 0.:
            self.LnID[np, activeID] = nodeID
            self.LnShape[np, activeID] = SF
            self.LnDShape[np, activeID] = GS
            self.ComputeBMatrix(np, activeID, GS)

    @ti.func
    def ComputeShapeFuncs2(self, sType, np, activeID, nodeID, xg):
        SF, GS, GSC = 0., ti.Matrix.zero(float, 2), ti.Matrix.zero(float, 2)
        if sType == 0: SF, GS, GSC = ShFunc.Linear2(self.x[np], xg, self.Dx)
        elif sType == 2: SF, GS, GSC = ShFunc.BsplineQ2(self.x[np], xg, self.Dx, self.Domain)
        elif sType == 3: SF, GS, GSC = ShFunc.BsplineC2(self.x[np], xg, self.Dx, self.Domain)
        if SF > 0.:
            self.LnID[np, activeID] = nodeID
            self.LnShape[np, activeID] = SF
            self.LnDShape[np, activeID] = GS
            self.ComputeBMatrix(np, activeID, GS)
            self.BbarMethod(np, activeID, GS, GSC)

    @ti.func
    def UpdateShapeFuncs(self, sType, np, activeID, nodeID, xg, stablization):
        if ti.static(stablization == 2): self.ComputeShapeFuncs2(sType, np, activeID, nodeID, xg)
        else: self.ComputeShapeFuncs1(sType, np, activeID, nodeID, xg)

    # ========================================================= #
    #                                                           #
    #                Compute Particle Velocity                  #
    #                                                           #
    # ========================================================= # 
    @ti.func
    def ComputeParticleVel(self, np, xg):
        return self.v[np] - self.gradv[np] @ (self.x[np] - xg)
    
    # ========================================================= #
    #                                                           #
    #                 Compute Internal Force                    #
    #                                                           #
    # ========================================================= #    
    @ti.func
    def ComputeInternalForce(self, np, ln, fInt):
        sigma = ti.Matrix.zero(float, 2)
        bmatrix = self.Bmatrix[np, ln]
        sigma[0] = bmatrix[0, 0] * fInt[0, 0] + bmatrix[1, 0] * fInt[1, 1] + 0.5 * bmatrix[2, 0] * (fInt[0, 1] + fInt[1, 0])
        sigma[1] = bmatrix[0, 1] * fInt[0, 0] + bmatrix[1, 1] * fInt[1, 1] + 0.5 * bmatrix[2, 1] * (fInt[0, 1] + fInt[1, 0])
        return sigma
    
    # ========================================================= #
    #                                                           #
    #             Calculate velocity gradient                   #
    #                                                           #
    # ========================================================= #
    @ti.func
    def CalLocalDv(self, gridList, np):
        self.gradv[np] = ti.Matrix.zero(float, 2, 2)
        for ln in range(self.LnID.shape[1]):
            if self.LnID[np, ln] >= 0:
                nodeID = self.LnID[np, ln]
                v = gridList.v[nodeID]
                GS = self.LnDShape[np, ln]

                self.gradv[np][0, 0] += v[0] * GS[0]
                self.gradv[np][0, 1] += v[0] * GS[1]
                self.gradv[np][1, 0] += v[1] * GS[0]
                self.gradv[np][1, 1] += v[1] * GS[1]
    
    # ========================================================= #
    #                                                           #
    #        Update strain increment and strain rate            #
    #                                                           #
    # ========================================================= #
    @ti.func
    def UpdateStrain(self, np, dt, gridList):
        self.StrainRate[np] = ti.Matrix.zero(float, 2, 2)
        for ln in range(self.LnID.shape[1]):
            if self.LnID[np, ln] >= 0:
                nodeID = self.LnID[np, ln]
                gv = gridList.v[nodeID]
                bmatrix = self.Bmatrix[np, ln]
                self.StrainRate[np][0, 0] += bmatrix[0, 0] * gv[0] + bmatrix[0, 1] * gv[1]
                self.StrainRate[np][0, 1] += 0.5 * (bmatrix[2, 0] * gv[0] + bmatrix[2, 1] * gv[1])
                self.StrainRate[np][1, 0] += 0.5 * (bmatrix[2, 0] * gv[0] + bmatrix[2, 1] * gv[1])
                self.StrainRate[np][1, 1] += bmatrix[1, 0] * gv[0] + bmatrix[1, 1] * gv[1]
        dw = 0.5 * dt * (self.gradv[np] - self.gradv[np].transpose())
        de = self.StrainRate[np] * dt
        return de, dw
    
    # ========================================================= #
    #                                                           #
    #             Update deformation gradient                   #
    #                                                           #
    # ========================================================= #
    @ti.func
    def ComputeDeformationGrad0(self, np, dt, matList):
        self.td[np] = (ti.Matrix.identity(float, 2) + self.gradv[np] * dt) @ self.td[np]
        matID = self.materialID[np]
        if matList.matType[matID] == 3: self.UpdateJacobianFromDet(np, dt)
        else: self.UpdateJacobianFromGrad(np, dt)
    
    # ========================================================= #
    #                                                           #
    #                   Update Jacobian                         #
    #                                                           #
    # ========================================================= #
    @ti.func
    def UpdateJacobianFromGrad(self, np, dt):
        self.J[np] = self.td[np].determinant()

    @ti.func
    def UpdateJacobianFromDet(self, np, dt):
        self.J[np] = (1 + dt * self.gradv[np].trace()) * self.J[np]

    # ========================================================= #
    #                                                           #
    #               Anti-Locking (F-Bar Method)                 #
    #                                                           #
    # ========================================================= #
    @ti.func
    def ComputeDeformationGrad1(self, cellList, np, dt):
        self.UpdateParticleInCell(np, cellList)
        cellID = self.cellID[np]
        vol0 = self.vol0[np]
        vol = vol0 * self.J[np]
        cellList.CellVolumeInit(cellID, vol0)
        cellList.UpdateCellVolume(cellID, vol)

    @ti.func
    def ComputeDeformationGrad2(self, cellList, np, dt):
        cellID = self.cellID[np]
        self.td[np] *= (cellList.J[cellID] / self.J[np]) ** (1./3.)

    @ti.func
    def UpdateDeformationGrad(self, np, dt, matList, cellList, stablization, mode):
        if mode == 0: self.ComputeDeformationGrad0(np, dt, matList)
        if mode == 1 and stablization == 3:
            self.ComputeDeformationGrad1(cellList, np, dt)
        if mode == 2 and stablization == 3:
            self.ComputeDeformationGrad2(cellList, np, dt)
            self.UpdateJacobianFromGrad(np, dt)

    # ========================================================= #
    #                                                           #
    #                Update Particle Properity                  #
    #                                                           #
    # ========================================================= #
    @ti.func
    def UpdatePartProperties(self, np):
        self.vol[np] = self.J[np] * self.vol0[np]
        self.rad[np] = (self.vol[np] / math.pi) ** 0.5
        self.rho[np] = self.rho0[np] / self.J[np]
    
    # ========================================================= #
    #                                                           #
    #                   Update Sound Speed                      #
    #                                                           #
    # ========================================================= #
    @ti.func
    def UpdateSoundSpeed(self, matList, np):
        matID = self.materialID[np]
        if matList.matType[matID] != 3:
            young = matList.young[matID]
            possion = matList.possion[matID]
            self.cs[np] = ti.sqrt(young * (1 - possion) / (1 + possion) / (1 - 2 * possion) / self.rho[np])
    
    # ========================================================= #
    #                                                           #
    #                 Update Particle Domain                    #
    #                                                           #
    # ========================================================= #
    @ti.func
    def CalPSizeCP(self, np):
        self.pSize[np][0] = self.pSize0[np][0] * self.td[np][0, 0]
        self.pSize[np][1] = self.pSize0[np][1] * self.td[np][1, 1]

    @ti.func
    def CalPSizeR(self, np):
        self.pSize[np][0] = self.pSize0[np][0] * ti.sqrt(self.td[np][0, 0] ** 2 + self.td[np][1, 0] ** 2)
        self.pSize[np][1] = self.pSize0[np][1] * ti.sqrt(self.td[np][0, 1] ** 2 + self.td[np][1, 1] ** 2)
    
    # ========================================================= #
    #                                                           #
    #              Update Particle in which cell                #
    #                                                           #
    # ========================================================= #
    @ti.func
    def UpdateParticleInCell(self, np, cellList):
        i = self.x[np][0] // self.Dx[0]
        j = self.x[np][1] // self.Dx[1]
        self.cellID[np] = cellList.GetCellID(i, j)

    # ========================================================= #
    #                                                           #
    #              Update Particle in which cell                #
    #                                                           #
    # ========================================================= #
    @ti.func
    def ReComputePorePressure(self, np, p):
        self.P[np] = p
        self.stress[np] = p * ti.Matrix.identity(float, 2)
        
    # ============================================= Transfer Algorithm ======================================================= #
    # ========================================================= #
    #                                                           #
    #                        PIC & FLIP                         #
    #                                                           #
    # ========================================================= #
    @ti.func
    def LinearPICFLIP(self, np, gridList, alphaPIC, dt):
        vPIC, vFLIP = ti.Matrix.zero(float, 2, 1), self.v[np]
        pos = self.x[np]
        for ln in range(self.LnID.shape[1]):
            if self.LnID[np, ln] >= 0:
                nodeID = self.LnID[np, ln]
                SF = self.LnShape[np, ln]
                vPIC += SF * gridList.v[nodeID]
                vFLIP += SF * gridList.f[nodeID] / gridList.m[nodeID] * dt
                pos += SF * gridList.v[nodeID] * dt
        vel = alphaPIC * vPIC + (1 - alphaPIC) * vFLIP
        self.v[np] = vel * Zero2OneVector(self.fixVel[np]) + vFLIP * self.fixVel[np]
        self.x[np] = pos * Zero2OneVector(self.fixVel[np]) + vFLIP * dt * self.fixVel[np]
        if isnan(self.x[np]) or self.x[np][0] < 0. or self.x[np][0] > self.Domain[0] or self.x[np][1] < 0. or self.x[np][1] > self.Domain[1]: 
            print("ERROR POSITION: Particle ID:", self.ID[np], self.x[np])
            # assert 0

    # ========================================================= #
    #                                                           #
    #                           APIC                            #
    #                                                           #
    # ========================================================= #
    @ti.func
    def APICTransfer(self, np, gridList, alphaPIC, dt):
        pass

    # ========================================================= #
    #                                                           #
    #                         PolyPIC                           #
    #                                                           #
    # ========================================================= #
    @ti.func
    def PolyPICTransfer(self, np, gridList, alphaPIC, dt):
        pass
