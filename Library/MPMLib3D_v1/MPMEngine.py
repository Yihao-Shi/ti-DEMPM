import taichi as ti
from MPMLib3D_v1.Function import *
import MPMLib3D_v1.ConsitutiveModel as cm


@ti.data_oriented
class MPMEngine:
    def __init__(self, gravity, threshold, damp, alphaPIC, dt, partList, gridList, matList):
        self.gravity = gravity
        self.threshold = threshold
        self.alphaPIC = alphaPIC
        self.damp = damp
        self.dt = dt
        self.partList = partList
        self.gridList = gridList
        self.matList = matList
    
    @ti.kernel
    def Reset(self):
        for np in range(self.partList.particleNum[None]):
            self.partList.ResetParticleForce(np)

    @ti.kernel
    def GridReset(self):
        for ng in self.gridList.id:
            self.gridList.GridReset(ng)


    @ti.func
    def CalNDN(self, np):
        # Reset shape function (N) and gradient of shape function (DN)
        self.partList.ResetShapeFuncs(np)
        
        # Find min position of nodes which are influenced by this particle
        minLocalNodeId, maxLocalNodeId = self.gridList.FindNode(self.partList.x[np])

        # Find nodes within the influence range
        activeID = 0
        for i in range(minLocalNodeId[0], maxLocalNodeId[0] + 1):
            for j in range(minLocalNodeId[1], maxLocalNodeId[1] + 1):
                for k in range(minLocalNodeId[2], maxLocalNodeId[2] + 1):
                    nodeID = self.gridList.GetNodeID(i, j, k)
                    self.partList.UpdateShapeFuncs(np, activeID, nodeID)
                    

    @ti.kernel
    def ParticleToGrid_Momentum(self):
        for np in range(self.partList.particleNum[None]):
            self.CalNDN(np)
            for ln in range(self.partList.LnID.shape[1]):
                if self.partList.LnID[np, ln] >= 0:
                    nodeID = self.partList.LnID[np, ln]
                    nm = self.partList.LnShape[np, ln] * self.partList.m[np]
                    self.gridList.m[nodeID] += nm
                    self.gridList.mv[nodeID] += nm * self.partList.v[np]


    @ti.kernel
    def ParticleToGrid_Force(self):
        for np in range(self.partList.particleNum[None]):
            fInt = -self.partList.vol[np] * self.partList.stress[np]
            fex = self.partList.m[np] * self.gravity + self.partList.fc[np] 
            for ln in range(self.partList.LnID.shape[1]):
                if self.partList.LnID[np, ln] >= 0:
                    nodeID = self.partList.LnID[np, ln]
                    df = self.partList.LnShape[np, ln] * fex + self.partList.ComputeInternalForce(np, ln, fInt)
                    self.gridList.f[nodeID] += df


    @ti.kernel
    def GridMomentum(self):
        for ng in self.gridList.id:
            if self.gridList.m[ng] > self.threshold:
                self.gridList.f[ng] -= self.damp * self.gridList.f[ng].norm() * Normalize(self.gridList.mv[ng])
                self.gridList.mv[ng] += self.gridList.f[ng] * self.dt
                self.gridList.ApplyBoundaryCondition(ng)
                self.gridList.v[ng] = self.gridList.mv[ng] / self.gridList.m[ng]


    @ti.kernel
    def GridToParticle(self):
        for np in range(self.partList.particleNum[None]):
            vPIC, vFLIP = ti.Matrix.zero(float, 3, 1), self.partList.v[np]
            pos = self.partList.x[np]
            for ln in range(self.partList.LnID.shape[1]):
                if self.partList.LnID[np, ln] >= 0:
                    nodeID = self.partList.LnID[np, ln]
                    SF = self.partList.LnShape[np, ln]
                    vPIC += SF * self.gridList.v[nodeID] * Zero2OneVector(self.partList.fixVel[np])
                    vFLIP += SF * self.gridList.f[nodeID] / self.gridList.m[nodeID] * self.dt * Zero2OneVector(self.partList.fixVel[np])
                    pos += SF * self.gridList.v[nodeID] * self.dt * Zero2OneVector(self.partList.fixVel[np])
            self.partList.v[np] = self.alphaPIC * vPIC + (1 - self.alphaPIC) * vFLIP
            self.partList.x[np] = pos 


    @ti.kernel
    def UpdateStressStrain(self):
        for np in range(self.partList.particleNum[None]):
            self.partList.CalLocalDv(np)
            self.partList.UpdateDeformationGrad(np, mode=0)

            self.partList.UpdatePartProperties(np)
            de, dw = self.partList.UpdateStrain(np)
            cm.CalStress(np, de, dw, self.partList, self.matList, self.threshold)


@ti.data_oriented
class USF(MPMEngine):
    def __init__(self, gravity, threshold, damp, alphaPIC, dt, partList, gridList, matList):
        super().__init__(gravity, threshold, damp, alphaPIC, dt, partList, gridList, matList)

    @ti.kernel
    def GridVelocity(self):
        for ng in self.gridList.id:
            if self.gridList.m[ng] > self.threshold:
                self.gridList.ApplyBoundaryCondition(ng)
                self.gridList.ComputeNodalVelocity(ng)


@ti.data_oriented
class USL(MPMEngine):
    def __init__(self, gravity, threshold, damp, alphaPIC, dt, partList, gridList, matList):
        super().__init__(gravity, threshold, damp, alphaPIC, dt, partList, gridList, matList)


@ti.data_oriented
class MUSL(MPMEngine):
    def __init__(self, gravity, threshold, damp, alphaPIC, dt, partList, gridList, matList):
        super().__init__(gravity, threshold, damp, alphaPIC, dt, partList, gridList, matList)

    @ti.kernel
    def NodalMomentumMUSL(self):    
        for ng in self.gridList.id:
            self.gridList.mv[ng] = ti.Matrix.zero(float, 3)

        for np in range(self.partList.particleNum[None]):
            for ln in range(self.partList.LnID.shape[1]):
                if self.partList.LnID[np, ln] >= 0:
                    nodeID = self.partList.LnID[np, ln]
                    nm = self.partList.Projection(np, ln, self.partList.m[np])
                    self.gridList.UpdateNodalMass(nodeID, nm)
                    self.gridList.UpdateNodalMomentumPIC(nodeID, nm * self.partList.v[np])

        for ng in self.gridList.id:
            if self.gridList.m[ng] > self.threshold:
                self.gridList.ApplyBoundaryCondition(ng)
                self.gridList.ComputeNodalVelocity(ng)  


@ti.data_oriented
class GIMP(MPMEngine):
    def __init__(self, gravity, threshold, damp, alphaPIC, dt, partList, gridList, matList):
        super().__init__(gravity, threshold, damp, alphaPIC, dt, partList, gridList, matList)

    @ti.func
    def CalNDN(self, np):
        # Reset shape function (N) and gradient of shape function (DN)
        self.partList.ResetShapeFuncs(np)
        
        # Find min position of nodes which are influenced by this particle
        minLocalNodeId, maxLocalNodeId = self.gridList.FindNodeGIMP(self.partList.x[np], self.partList.pSize0[np])

        # Find nodes within the influence range
        activeID = 0
        for i in range(minLocalNodeId[0], maxLocalNodeId[0] + 1):
            for j in range(minLocalNodeId[1], maxLocalNodeId[1] + 1):
                for k in range(minLocalNodeId[2], maxLocalNodeId[2] + 1):
                    nodeID = self.gridList.GetNodeID(i, j, k)
                    self.partList.UpdateGIMP(np, activeID, nodeID)


@ti.data_oriented
class MLSMPM(MPMEngine):
    def __init__(self, gravity, threshold, damp, alphaPIC, dt, partList, gridList, matList):
        super().__init__(gravity, threshold, damp, alphaPIC, dt, partList, gridList, matList)

    


    


    


    


    

