import taichi as ti
from MPMLib2D_v1.Function import *


@ti.data_oriented
class GridList:
    def __init__(self, domain, dx, ContactDetection):
        self.gnum = ti.Vector([int(domain[0] / dx[0] + 1), int(domain[1] / dx[1] + 1)])       # Grid Number
        self.gridSum = self.gnum[0] * self.gnum[1]

        self.id = ti.field(int, self.gridSum)                                                 # ID of grids
        self.x = ti.Vector.field(2, float, self.gridSum)                                      # Position
        self.m = ti.field(float, self.gridSum)                                            # Mass
        self.v = ti.Vector.field(2, float, self.gridSum)                                  # Velocity
        self.mv = ti.Vector.field(2, float, self.gridSum)                                 # Momentum
        self.f = ti.Vector.field(2, float, self.gridSum)                                  # Node force

        self.BCType = ti.field(int, shape=(self.gridSum, 4))                                  # Boundary condition type
        self.BoundaryNum = ti.field(int, shape=(self.gridSum,))                               # Boundary condition Number
        self.Norm = ti.Vector.field(2, float, shape=(self.gridSum, 4))                        # Normal direction for boundary
        self.miu = ti.field(float, shape=(self.gridSum, 4))                                   # Friction coefficient
    
    # ========================================================= #
    #                                                           #
    #                  Get Node ID & Index                      #
    #                                                           #
    # ========================================================= # 
    @ti.func
    def GetGridIndex(self, ng):
        ig = (ng % (self.gnum[0] * self.gnum[1])) % self.gnum[0]
        jg = (ng % (self.gnum[0] * self.gnum[1])) // self.gnum[0]
        return ig, jg

    @ti.func
    def GetNodeID(self, i, j):
        return int(i + j * self.gnum[0])
    
    # ======================================== MPM Grid Initialization ======================================== #
    @ti.kernel
    def GridInit(self, dx: ti.template()):
        for ng in self.id:
            ig, jg = self.GetGridIndex(ng)
            pos = ti.Vector([ig, jg]) * dx
            self.id[ng] = ng
            self.x[ng] = pos

    # =========================================== MPM Grid Reset ============================================== #
    @ti.kernel
    def GridReset(self, ng):
        for ng in self.m:
            if self.m[ng] > 0:
                self.m[ng] = 0.
                self.v[ng] = ti.Matrix.zero(float, 2)
                self.mv[ng] = ti.Matrix.zero(float, 2)
                self.f[ng] = ti.Matrix.zero(float, 2)

    # ========================================== Boundary Condition =========================================== #
    @ti.func
    def ApplyBoundaryCondition(self, ng, dt):
        for b in range(self.BoundaryNum[ng]):
            if self.BCType[ng, b] == 1:
                self.NonSlippingBC(ng)
            elif self.BCType[ng, b] == 2:
                self.SlippingBC(ng, b)
            elif self.BCType[ng, b] == 3:
                self.FrictionBC(ng, dt, b)

    @ti.kernel
    def SetNonSlippingBC(self, i: int, j: int):
        nodeId = int(i + j * self.gnum[0])
        self.BCType[nodeId, self.BoundaryNum[nodeId]] = 1
        self.BoundaryNum[nodeId] += 1

    @ti.func
    def NonSlippingBC(self, ng):
        self.f[ng] = ti.Matrix.zero(float, 2)
        self.mv[ng] = ti.Matrix.zero(float, 2)

    @ti.kernel
    def SetSlippingBC(self, i: int, j: int, norm: ti.types.vector(2, int)):
        nodeId = int(i + j * self.gnum[0])
        self.BCType[nodeId, self.BoundaryNum[nodeId]] = 2
        self.Norm[nodeId, self.BoundaryNum[nodeId]] = norm
        self.BoundaryNum[nodeId] += 1

    @ti.func
    def SlippingBC(self, ng, b):
        if self.mv[ng].dot(self.Norm[ng, b]) > 0:
            self.f[ng] = self.f[ng] - self.f[ng].dot(self.Norm[ng, b]) * self.Norm[ng, b]
            self.mv[ng] = self.mv[ng] - self.mv[ng].dot(self.Norm[ng, b]) * self.Norm[ng, b]
   
    @ti.kernel
    def SetFrictionBC(self, i: int, j: int, miu: float, norm: ti.types.vector(2, int)):
        nodeId = int(i + j * self.gnum[0])
        self.BCType[nodeId, self.BoundaryNum[nodeId]] = 3
        self.Norm[nodeId, self.BoundaryNum[nodeId]] = norm
        self.miu[nodeId, self.BoundaryNum[nodeId]] = miu
        self.BoundaryNum[nodeId] += 1

    @ti.func
    def FrictionBC(self, ng, dt, b):
        mvNorm = self.mv[ng].dot(self.Norm[ng, b])
        if mvNorm > 0:
            vt0 = self.v[ng] - self.v[ng].dot(self.Norm[ng, b]) * self.Norm[ng, b]
            mvn0 = mvNorm * self.Norm[ng, b]
            mvt0 = self.mv[ng] - mvn0
            tang = mvt0.normalized()
            fmiu = -sign(vt0.dot(tang)) * self.miu[ng, b] * self.f[ng].dot(self.Norm[ng, b]) * tang
            mvt1 = mvt0 + fmiu * dt

            if mvt0.dot(mvt1) > 0:
                self.f[ng] = self.f[ng].dot(tang) * tang + fmiu
                self.mv[ng] = mvt1
            else:
                self.f[ng] = ti.Matrix.zero(float, 2)
                self.mv[ng] = ti.Matrix.zero(float, 2)
    
    @ti.kernel
    def SetNewmanBC(self, field: ti.template(), f: ti.types.vector(2, float)):
        for i in field:
            ID = field[i][1]
            self.lp.fex[ID] = f

    @ti.kernel
    def FindTractionBoundary(self, idxmin: int, idxmax: int, idymin: int, idymax: int, field: ti.template(), nb: int):
        xmin = (idxmin + 0.5) * self.Dx[0] / self.Npic + self.BodyInfo[nb].pos0[0]
        xmax = (idxmax + 0.5) * self.Dx[0] / self.Npic + self.BodyInfo[nb].pos0[0]
        ymin = (idymin + 0.5) * self.Dx[1] / self.Npic + self.BodyInfo[nb].pos0[1]
        ymax = (idymax + 0.5) * self.Dx[1] / self.Npic + self.BodyInfo[nb].pos0[1]
        row = 0
        for np in self.lp.ID:
            if 0.99 * xmin < self.lp.x[np][0] < 1.01 * xmax and 0.99 * ymin < self.lp.x[np][1] < 1.01 * ymax:
                field[ti.atomic_add(row, 1)] = [nb, self.lp.ID[np]]

    @ti.kernel
    def SetFreeSurfaceBC(self, field: ti.template()):
        for i in field:
            ID = field[i][1]
            self.lp.coeff[ID] = 0
    
    # ================================================ Solve =================================================== #
    @ti.func
    def FindNode(self, sType, xp, dx):
        minLocalNodeId, maxLocalNodeId = ti.Matrix.zero(int, 2), ti.Matrix.zero(int, 2)
        if sType == 0: minLocalNodeId, maxLocalNodeId = self.FindNodeLinear(xp, dx)
        elif sType == 2: minLocalNodeId, maxLocalNodeId = self.FindNodeBSplineQ(xp, dx)
        elif sType == 3: minLocalNodeId, maxLocalNodeId = self.FindNodeBSplineC(xp, dx)
        return minLocalNodeId, maxLocalNodeId

    @ti.func
    def FindNodeLinear(self, xp, dx):
        minLocalNodeId = ti.max(int(xp / dx), 0)
        maxLocalNodeId = int(ti.min(minLocalNodeId + 1, self.gnum - 1))
        return minLocalNodeId, maxLocalNodeId

    @ti.func
    def FindNodeGIMP(self, xp, psize, dx):
        minLocalNodeId = int(ti.max(ti.ceil((xp - psize - dx) / dx), 0.))
        maxLocalNodeId = int(ti.min(ti.floor((xp + psize + dx) / dx), self.gnum - 1))
        return minLocalNodeId, maxLocalNodeId

    @ti.func
    def FindNodeBSplineQ(self, xp, dx):
        minLocalNodeId = ti.max(int(xp / dx - 0.5), 0)
        maxLocalNodeId = int(ti.min(minLocalNodeId + 2, self.gnum - 1))
        return minLocalNodeId, maxLocalNodeId

    @ti.func
    def FindNodeBSplineC(self, xp, dx):
        minLocalNodeId = ti.max(int(xp / dx) - 1, 0)
        maxLocalNodeId = int(ti.min(minLocalNodeId + 3, self.gnum - 1))
        return minLocalNodeId, maxLocalNodeId

    @ti.func
    def UpdateNodalMass(self, ng, mp):
        self.m[ng] += mp

    @ti.func
    def UpdateNodalMomentumPIC(self, ng, mvp):
        self.mv[ng] += mvp

    @ti.func
    def UpdateNodalMomentumAPIC(self, ng, mvp):
        pass

    @ti.func
    def UpdateNodalMomentumPolyPIC(self, ng, mvp):
        pass

    @ti.func
    def UpdateNodalForce(self, ng, fp):
        self.f[ng] += fp

    @ti.func
    def ApplyGlobalDamping(self, ng, damp):
        self.f[ng] -= damp * self.f[ng].norm() * Normalize(self.mv[ng])

    @ti.func
    def ComputeNodalMomentum(self, ng, dt):
        self.mv[ng] += self.f[ng] * dt

    @ti.func
    def ComputeNodalVelocity(self, ng):
        self.v[ng] = self.mv[ng] / self.m[ng]

