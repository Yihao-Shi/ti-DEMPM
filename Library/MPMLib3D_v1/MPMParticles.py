import taichi as ti
from MPMLib3D_v1.Function import *
import MPMLib3D_v1.ShapeFunctions as ShFunc
import math


@ti.data_oriented
class ParticleList:
    # =============================================== Set Up Taichi Field ============================================================ #
    def __init__(self, dx, domain, threshold, max_particle_num, shapefunction, stablization, dt, gridList, matList, cellList):
        self.particleNum = ti.field(int, shape=())                                   # Total particle Number
        self.max_particle_num = max_particle_num                                     # Max particle number
        self.Dx = dx
        self.Domain = domain
        self.threshold = threshold
        self.shapefunction = shapefunction
        self.stablization = stablization
        self.dt = dt
        self.gridList = gridList
        self.matList = matList
        self.cellList = cellList

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

        self.x = ti.Vector.field(3, float, max_particle_num)                         # Position
        self.v = ti.Vector.field(3, float, max_particle_num)                         # Velocity
        self.fixVel = ti.Vector.field(3, int, max_particle_num)                      # fix velocity
        self.fc = ti.Vector.field(3, float, max_particle_num)                        # Contact force

        self.strain = ti.Matrix.field(3, 3, float, max_particle_num)                 # Strain tensor
        self.StrainRate = ti.Matrix.field(3, 3, float, max_particle_num)             # Shear strain rate
        self.plasticStrainEff = ti.field(float, max_particle_num)                    # Norm of deviatoric plastic strain tensor
        self.stress = ti.Matrix.field(3, 3, float, max_particle_num)                 # Stress tensor
        self.gradv = ti.Matrix.field(3, 3, float, max_particle_num)                  # Velocity gradient tensor
        self.td = ti.Matrix.field(3, 3, float, max_particle_num)                     # Deformation gradient tensor
    
    def StoreShapeFuncs(self, BasisType):
        self.LnID = ti.field(int, (self.max_particle_num, BasisType))                     # List of node index
        self.LnShape = ti.field(float, (self.max_particle_num, BasisType))                # List of shape function
        self.LnDShape = ti.Vector.field(3, float, (self.max_particle_num, BasisType))  # List of gradient of shape function
        self.Bmatrix = ti.Matrix.field(6, 3, float, (self.max_particle_num, BasisType))   # List of B Matrix

    def StoreParticleLen(self):
        self.pSize0 = ti.Vector.field(3, float, self.max_particle_num)                    # Initial half length of particle domain for GIMP
        self.pSize = ti.Vector.field(3, float, self.max_particle_num)                     # Half length of particle domain for GIMP

    def MPMCell(self):
        self.cellID = ti.field(int, self.max_particle_num)                           # The cell ID of mpm particles
        self.PartcleInCellInit()
    
    # ====================================== MPM Particle Initialization ====================================== #
    @ti.func
    def ParticleInit(self, np, npic, nb, bodyInfo, pos):
        self.ID[np] = np
        self.bodyID[np] = bodyInfo[nb].ID
        self.materialID[np] = bodyInfo[nb].Mat
        self.rho0[np] = bodyInfo[nb].MacroRho
        self.rho[np] = self.rho0[np]
        self.vol0[np] = (1./3. * (self.Dx[0] + self.Dx[1] + self.Dx[2]) / npic) ** 3
        self.vol[np] = self.vol0[np]
        self.m[np] = self.vol0[np] * bodyInfo[nb].MacroRho
        self.rad[np] = 1./3. * (self.Dx[0] + self.Dx[1] + self.Dx[2]) / npic
        self.x[np] = pos
        self.v[np] = bodyInfo[nb].v0
        self.fixVel[np] = bodyInfo[nb].fixedV
        self.td[np] = ti.Matrix.identity(float, 3)
        self.J[np] = 1.

    @ti.kernel
    def GIMPInit(self, npic: int):
        for np in range(self.particleNum[None]):
            self.pSize0[np] = 0.5 * self.Dx / npic
            self.pSize[np] = self.pSize0[np]

    @ti.kernel
    def PartcleInCellInit(self):
        for np in range(self.particleNum[None]):
            self.UpdateParticleInCell(np)
            cellID = self.cellID[np]

    # ============================================ Reset ====================================================== #
    @ti.func
    def ResetShapeFuncs(self, np):
        for i in range(self.LnID.shape[1]):
            self.LnID[np, i] = -1
            self.LnShape[np, i] = 0.
            self.LnDShape[np, i] = ti.Matrix.zero(float, 3)
            self.Bmatrix[np, i] = ti.Matrix.zero(float, 6, 3)

    @ti.func
    def ResetParticleForce(self, np):
        self.fc[np] = ti.Matrix.zero(float, 3)

    @ti.func
    def AdaptiveParticle(self, np):
        pass

    @ti.func
    def CheckDt(self, dt):
        self.dt = dt

    # ======================================== Add Particles ================================================= #
    # ========================================================= #
    #                                                           #
    #                     Add Rectangle                         #
    #                                                           #
    # ========================================================= #
    @ti.kernel
    def AddRectangle(self, nb: int, bodyInfo: ti.template(), npic: int):
        pnum = (bodyInfo[nb].len * npic / self.Dx).cast(int)
        new_particle_num = pnum[0] * pnum[1] * pnum[2]
        if self.particleNum[None] + new_particle_num > self.max_particle_num:
            print("The MPM particles should be set as: ", self.particleNum[None] + new_particle_num)
            assert 0
        print("Body Shape: Rectangle")
        print("Material ID = ", bodyInfo[nb].Mat)
        print("Macro Density = ", bodyInfo[nb].MacroRho)
        print("Initial Velocity = ", bodyInfo[nb].v0)
        print("Fixed Velocity = ", bodyInfo[nb].fixedV)
        print("Particle Number = ", pnum, '\n')
        for np in range(new_particle_num):
            ip = (np % (pnum[0] * pnum[1])) % pnum[0]
            jp = (np % (pnum[0] * pnum[1])) // pnum[0]
            kp = np // (pnum[0] * pnum[1])
            pos = (ti.Vector([ip, jp, kp]) + 0.5) * self.Dx / npic + bodyInfo[nb].pos0
            self.ParticleInit(self.particleNum[None] + np, npic, nb, bodyInfo, pos)
            self.UpdateSoundSpeed(self.particleNum[None] + np)
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
    def AddTriangle(self, nb: int, bodyInfo: ti.template(), npic: int):
        pnum = (bodyInfo[nb].len * npic / self.Dx).cast(int)
        new_particle_num_trial = pnum[0] * pnum[1] * pnum[2]
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
            kp = np // (pnum[0] * pnum[1])
            pos = (ti.Vector([ip, jp, kp]) + 0.5) * self.Dx / npic + bodyInfo[nb].pos0
            if self.RegionTriangle(pos[0], pos[2], bodyInfo[nb].pos0[0], bodyInfo[nb].pos0[2], bodyInfo[nb].len[0], bodyInfo[nb].len[2]) < 0:
                temp = ti.atomic_add(self.particleNum[None], 1)
                self.ParticleInit(temp, npic, nb, bodyInfo, pos)
                self.UpdateSoundSpeed(temp)
    
    # ========================================================= #
    #                                                           #
    #                      Add Sphere                           #
    #                                                           #
    # ========================================================= #
    @ti.func
    def RegionCircle(self, xpos, ypos, zpos, x0, x1, x2, rad):
        return (xpos - x0) ** 2 + (ypos - x1) ** 2 + (zpos - x2) ** 2 - rad ** 2

    @ti.kernel
    def AddCircle(self, nb: int, bodyInfo: ti.template(), npic: int):
        pnum = (bodyInfo[nb].len * npic / self.Dx).cast(int)
        new_particle_num_trial = pnum[0] * pnum[1] * pnum[2]
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
            kp = np // (pnum[0] * pnum[1])
            pos = (ti.Vector([ip, jp, kg]) + 0.5) * self.Dx / npic + bodyInfo[nb].pos0
            if self.RegionCircle(pos[0], pos[1], pos[2], bodyInfo[nb].pos0[0], bodyInfo[nb].pos0[1], bodyInfo[nb].pos0[2], bodyInfo[nb].rad) < 0:
                temp = ti.atomic_add(self.particleNum[None], 1)
                self.ParticleInit(temp, npic, nb, bodyInfo, pos)
                self.UpdateSoundSpeed(temp)

    # ============================================= Solve ======================================================= #
    # ========================================================= #
    #                                                           #
    #                   Compute B Matrix                        #
    #                                                           #
    # ========================================================= #
    @ti.func
    def ComputeBMatrix(self, np, activeID, GS):
        temp = ti.Matrix.zero(float, 6, 3)
        temp[0, 0] = GS[0]
        temp[1, 1] = GS[1]
        temp[2, 2] = GS[2]
        temp[3, 0], temp[3, 1] = GS[1], GS[0]
        temp[4, 1], temp[4, 2] = GS[2], GS[1]
        temp[5, 0], temp[5, 2] = GS[2], GS[0]
        self.Bmatrix[np, activeID] = temp
    
    # ========================================================= #
    #                                                           #
    #               Anti-Locking (B-Bar Method)                 #
    #                                                           #
    # ========================================================= #
    @ti.func
    def BbarMethod(self, np, activeID, GS, GSC):
        temp = ti.Matrix.zero(float, 6, 3)
        temp[0, 0] = (GSC[0] - GS[0]) / 3.
        temp[0, 1] = (GSC[1] - GS[1]) / 3.
        temp[0, 2] = (GSC[2] - GS[2]) / 3.
        temp[1, 0] = (GSC[0] - GS[0]) / 3.
        temp[1, 1] = (GSC[1] - GS[1]) / 3.
        temp[1, 2] = (GSC[2] - GS[2]) / 3.
        temp[2, 0] = (GSC[0] - GS[0]) / 3.
        temp[2, 1] = (GSC[1] - GS[1]) / 3.
        temp[2, 2] = (GSC[2] - GS[2]) / 3.
        temp[2, 0] = (GSC[0] - GS[0]) / 3.
        temp[2, 1] = (GSC[1] - GS[1]) / 3.
        temp[2, 2] = (GSC[2] - GS[2]) / 3.
        self.Bmatrix[np, activeID] += temp
    
    # ========================================================= #
    #                                                           #
    #              Compute GIMP Shape Functions                 #
    #                                                           #
    # ========================================================= #
    @ti.func
    def ComputeGIMP1(self, np, activeID: ti.template(), nodeID):
        SF, GS = ShFunc.GIMP1(self.x[np], self.gridList.x[nodeID], self.Dx, self.pSize[np])
        if SF > self.threshold:
            self.LnID[np, activeID] = nodeID
            self.LnShape[np, activeID] = SF
            self.LnDShape[np, activeID] = GS
            self.ComputeBMatrix(np, activeID, GS)
            activeID += 1

    @ti.func
    def ComputeGIMP2(self, np, activeID: ti.template(), nodeID):
        SF, GS, GSC = ShFunc.GIMP2(self.x[np], self.gridList.x[nodeID], self.Dx, self.pSize[np])
        if SF > self.threshold:
            self.LnID[np, activeID] = nodeID
            self.LnShape[np, activeID] = SF
            self.LnDShape[np, activeID] = GS
            self.ComputeBMatrix(np, activeID, GS)
            self.BbarMethod(np, activeID, GS, GSC)
            activeID += 1  

    @ti.func
    def UpdateGIMP(self, np, activeID: ti.template(), nodeID):
        if ti.static(self.stablization == 0): self.ComputeGIMP1(np, activeID, nodeID)
        elif ti.static(self.stablization == 2): self.ComputeGIMP2(np, activeID, nodeID)

    # ========================================================= #
    #                                                           #
    #                 Compute Shape Functions                   #
    #                                                           #
    # ========================================================= #
    @ti.func
    def ComputeShapeFuncs1(self, np, activeID, nodeID):
        SF, GS = 0., ti.Matrix.zero(float, 3)
        if ti.static(self.shapeFunction == 0): SF, GS = ShFunc.Linear1(self.x[np], self.gridList.x[nodeID], self.Dx)
        elif ti.static(self.shapeFunction == 2): SF, GS = ShFunc.BsplineQ1(self.x[np], self.gridList.x[nodeID], self.Dx, self.Domain)
        elif ti.static(self.shapeFunction == 3): SF, GS = ShFunc.BsplineC1(self.x[np], self.gridList.x[nodeID], self.Dx, self.Domain)
        if SF > 0.:
            self.LnID[np, activeID] = nodeID
            self.LnShape[np, activeID] = SF
            self.LnDShape[np, activeID] = GS
            self.ComputeBMatrix(np, activeID, GS)

    @ti.func
    def ComputeShapeFuncs2(self, np, activeID, nodeID):
        SF, GS, GSC = 0., ti.Matrix.zero(float, 3), ti.Matrix.zero(float, 3)
        if ti.static(self.shapeFunction == 0): SF, GS, GSC = ShFunc.Linear2(self.x[np], self.gridList.x[nodeID], self.Dx)
        elif ti.static(self.shapeFunction == 2): SF, GS, GSC = ShFunc.BsplineQ2(self.x[np], self.gridList.x[nodeID], self.Dx, self.Domain)
        elif ti.static(self.shapeFunction == 3): SF, GS, GSC = ShFunc.BsplineC2(self.x[np], self.gridList.x[nodeID], self.Dx, self.Domain)
        if SF > 0.:
            self.LnID[np, activeID] = nodeID
            self.LnShape[np, activeID] = SF
            self.LnDShape[np, activeID] = GS
            self.ComputeBMatrix(np, activeID, GS)
            self.BbarMethod(np, activeID, GS, GSC)

    @ti.func
    def UpdateShapeFuncs(self, np, activeID, nodeID):
        if ti.static(self.stablization == 0): self.ComputeShapeFuncs1(np, activeID, nodeID)
        elif ti.static(self.stablization == 2): self.ComputeShapeFuncs2(np, activeID, nodeID)

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
        sigma = ti.Matrix.zero(float, 3)
        bmatrix = self.Bmatrix[np, ln]
        sigma[0] = bmatrix[0, 0] * fInt[0, 0] + bmatrix[1, 0] * fInt[1, 1] + bmatrix[2, 0] * fInt[2, 2] + 0.5 * bmatrix[3, 0] * (fInt[0, 1] + fInt[1, 0]) + 0.5 * bmatrix[5, 0] * (fInt[1, 2] + fInt[2, 1])
        sigma[1] = bmatrix[0, 1] * fInt[0, 0] + bmatrix[1, 1] * fInt[1, 1] + bmatrix[2, 1] * fInt[2, 2] + 0.5 * bmatrix[3, 1] * (fInt[0, 1] + fInt[1, 0]) + 0.5 * bmatrix[4, 1] * (fInt[0, 2] + fInt[2, 0])
        sigma[2] = bmatrix[0, 2] * fInt[0, 0] + bmatrix[1, 2] * fInt[1, 1] + bmatrix[2, 2] * fInt[2, 2] + 0.5 * bmatrix[4, 2] * (fInt[0, 2] + fInt[2, 0]) + 0.5 * bmatrix[5, 2] * (fInt[1, 2] + fInt[2, 1])
        return sigma
    
    # ========================================================= #
    #                                                           #
    #             Calculate velocity gradient                   #
    #                                                           #
    # ========================================================= #
    @ti.func
    def CalLocalDv(self, np):
        temp = ti.Matrix.zero(float, 3, 3)
        for ln in range(self.LnID.shape[1]):
            if self.LnID[np, ln] >= 0:
                nodeID = self.LnID[np, ln]
                gv = self.gridList.v[nodeID]
                GS = self.LnDShape[np, ln]
                temp[0, 0] += gv[0] * GS[0]
                temp[0, 1] += gv[0] * GS[1]
                temp[0, 2] += gv[0] * GS[2]
                temp[1, 0] += gv[1] * GS[0]
                temp[1, 1] += gv[1] * GS[1]
                temp[1, 2] += gv[1] * GS[2]
                temp[2, 0] += gv[2] * GS[0]
                temp[2, 1] += gv[2] * GS[1]
                temp[2, 2] += gv[2] * GS[2]
        self.gradv[np] = temp
    
    # ========================================================= #
    #                                                           #
    #        Update strain increment and strain rate            #
    #                                                           #
    # ========================================================= #
    @ti.func
    def UpdateStrain(self, np):
        temp = ti.Matrix.zero(float, 3, 3)
        for ln in range(self.LnID.shape[1]):
            if self.LnID[np, ln] >= 0:
                nodeID = self.LnID[np, ln]
                bmatrix = self.Bmatrix[np, ln]
                gv = self.gridList.v[nodeID]
                temp[0, 0] += bmatrix[0, 0] * gv[0] + bmatrix[0, 1] * gv[1] + bmatrix[0, 2] * gv[2]
                temp[1, 1] += bmatrix[1, 0] * gv[0] + bmatrix[1, 1] * gv[1] + bmatrix[1, 2] * gv[2]
                temp[2, 2] += bmatrix[1, 0] * gv[0] + bmatrix[1, 1] * gv[1] + bmatrix[2, 2] * gv[2]
                temp[0, 1] += 0.5 * (bmatrix[3, 0] * gv[0] + bmatrix[3, 1] * gv[1]) 
                temp[0, 2] += 0.5 * (bmatrix[4, 1] * gv[1] + bmatrix[4, 2] * gv[2]) 
                temp[2, 1] += 0.5 * (bmatrix[5, 0] * gv[0] + bmatrix[5, 2] * gv[2]) 
        temp[1, 0] = temp[0, 1]
        temp[2, 0] = temp[0, 2]
        temp[1, 2] = temp[2, 1]
        dw = 0.5 * self.dt * (self.gradv[np] - self.gradv[np].transpose())
        self.StrainRate[np] = temp
        de = self.StrainRate[np] * self.dt
        return de, dw
    
    # ========================================================= #
    #                                                           #
    #             Update deformation gradient                   #
    #                                                           #
    # ========================================================= #
    @ti.func
    def ComputeDeformationGrad0(self, np):
        self.td[np] = (ti.Matrix.identity(float, 3) + self.gradv[np] * self.dt) @ self.td[np]
        matID = self.materialID[np]
        if self.matList.matType[matID] == 3: self.UpdateJacobianFromDet(np)
        else: self.UpdateJacobianFromGrad(np)
    
    # ========================================================= #
    #                                                           #
    #                   Update Jacobian                         #
    #                                                           #
    # ========================================================= #
    @ti.func
    def UpdateJacobianFromGrad(self, np):
        self.J[np] = self.td[np].determinant()

    @ti.func
    def UpdateJacobianFromDet(self, np):
        self.J[np] = (1 + self.dt * self.gradv[np].trace()) * self.J[np]

    # ========================================================= #
    #                                                           #
    #               Anti-Locking (F-Bar Method)                 #
    #                                                           #
    # ========================================================= #
    @ti.func
    def ComputeDeformationGrad1(self, np):
        self.UpdateParticleInCell(np)
        cellID = self.cellID[np]
        vol0 = self.vol0[np]
        vol = vol0 * self.J[np]
        self.cellList.CellVolumeInit(cellID, vol0)
        self.cellList.UpdateCellVolume(cellID, vol)

    @ti.func
    def ComputeDeformationGrad2(self, np):
        cellID = self.cellID[np]
        self.td[np] *= (self.cellList.J[cellID] / self.J[np]) ** (1./3.)

    @ti.func
    def UpdateDeformationGrad(self, np, mode):
        if mode == 0: self.ComputeDeformationGrad0(np)
        elif ti.static(mode == 1 and self.stablization == 3):
            self.ComputeDeformationGrad1(np)
        elif ti.static(mode == 2 and self.stablization == 3):
            self.ComputeDeformationGrad2(np)
            self.UpdateJacobianFromGrad(np)

    # ========================================================= #
    #                                                           #
    #                Update Particle Properity                  #
    #                                                           #
    # ========================================================= #
    @ti.func
    def UpdatePartProperties(self, np):
        self.vol[np] = self.J[np] * self.vol0[np]
        self.rho[np] = self.rho0[np] / self.J[np]
    
    # ========================================================= #
    #                                                           #
    #                   Update Sound Speed                      #
    #                                                           #
    # ========================================================= #
    @ti.func
    def UpdateSoundSpeed(self, np):
        matID = self.materialID[np]
        if self.matList.matType[matID] != 3:
            young = self.matList.young[matID]
            possion = self.matList.possion[matID]
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
        self.pSize[np][2] = self.pSize0[np][2] * self.td[np][2, 2] 

    @ti.func
    def CalPSizeR(self, np):
        self.pSize[np][0] = self.pSize0[np][0] * ti.sqrt(self.td[np][0, 0] ** 2 + self.td[np][1, 0] ** 2 + self.td[np][0, 2] ** 2)
        self.pSize[np][1] = self.pSize0[np][1] * ti.sqrt(self.td[np][0, 1] ** 2 + self.td[np][1, 1] ** 2 + self.td[np][1, 2] ** 2)
        self.pSize[np][2] = self.pSize0[np][2] * ti.sqrt(self.td[np][0, 2] ** 2 + self.td[np][1, 2] ** 2 + self.td[np][2, 2] ** 2)

    # ========================================================= #
    #                                                           #
    #              Update Particle in which cell                #
    #                                                           #
    # ========================================================= #
    @ti.func
    def UpdateParticleInCell(self, np):
        i = self.x[np][0] // self.Dx[0]
        j = self.x[np][1] // self.Dx[1]
        k = self.x[np][2] // self.Dx[2]
        self.cellID[np] = self.cellList.GetCellID(i, j, k)
