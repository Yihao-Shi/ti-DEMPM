import taichi as ti
from MPMLib3D_v1.Function import *


@ti.data_oriented
class GridList:
    def __init__(self, domain, threshold, shapeFunction, dx, dt, ContactDetection):
        self.gnum = ti.Vector([int(domain[0] / dx[0] + 1), int(domain[1] / dx[1] + 1), int(domain[2] / dx[2] + 1)])       # Grid Number
        self.gridSum = int(self.gnum[0] * self.gnum[1] * self.gnum[2])
        self.threshold = threshold
        self.shapeFunction = shapeFunction
        self.dx = dx
        self.dt = dt

        self.id = ti.field(int, self.gridSum)                                                 # ID of grids
        self.x = ti.Vector.field(3, float, self.gridSum)                                      # Position
        self.m = ti.field(float, self.gridSum)                                            # Mass
        self.v = ti.Vector.field(3, float, self.gridSum)                                  # Velocity
        self.mv = ti.Vector.field(3, float, self.gridSum)                                 # Momentum
        self.f = ti.Vector.field(3, float, self.gridSum)                                  # Node force

        self.BCType = ti.field(int, shape=(self.gridSum, 6))                                  # Boundary condition type
        self.BoundaryNum = ti.field(int, shape=(self.gridSum,))                               # Boundary condition Number
        self.Norm = ti.Vector.field(3, float, shape=(self.gridSum, 6))                        # Normal direction for boundary
        self.miu = ti.field(float, shape=(self.gridSum, 6))                                   # Friction coefficient
    
    # ========================================================= #
    #                                                           #
    #                  Get Node ID & Index                      #
    #                                                           #
    # ========================================================= # 
    @ti.func
    def GetGridIndex(self, ng):
        ig = (ng % (self.gnum[0] * self.gnum[1])) % self.gnum[0]
        jg = (ng % (self.gnum[0] * self.gnum[1])) // self.gnum[0]
        kg = ng // (self.gnum[0] * self.gnum[1])
        return ig, jg, kg

    @ti.func
    def GetNodeID(self, i, j, k):
        return int(i + j * self.gnum[0] + k * self.gnum[0] * self.gnum[1])
    
    # ======================================== MPM Grid Initialization ======================================== #
    @ti.kernel
    def GridInit(self):
        for ng in self.id:
            ig, jg, kg = self.GetGridIndex(ng)
            pos = ti.Vector([ig, jg, kg]) * self.dx
            self.id[ng] = ng
            self.x[ng] = pos

    @ti.func
    def GridResetContact(self, ng, nb):
        self.m[ng, nb] = 0.
        self.v0[ng, nb] = self.v[ng]
        self.v[ng, nb] = ti.Matrix.zero(float, 3)
        self.mv[ng, nb] = ti.Matrix.zero(float, 3)
        self.f[ng, nb] = ti.Matrix.zero(float, 3)
    
    # =========================================== MPM Grid Reset ============================================== #
    @ti.func
    def GridReset(self, ng):
        if self.m[ng] > self.threshold:
            self.m[ng] = 0.
            self.v[ng] = ti.Matrix.zero(float, 3)
            self.mv[ng] = ti.Matrix.zero(float, 3)
            self.f[ng] = ti.Matrix.zero(float, 3)

    @ti.func
    def CheckDt(self, dt):
        self.dt = dt

    # ========================================== Boundary Condition =========================================== #
    @ti.func
    def ApplyBoundaryCondition(self, ng):
        for b in range(self.BoundaryNum[ng]):
            if self.BCType[ng, b] == 1:
                self.NonSlippingBC(ng)
            elif self.BCType[ng, b] == 2:
                self.SlippingBC(ng, b)
            elif self.BCType[ng, b] == 3:
                self.FrictionBC(ng, b)

    @ti.kernel
    def SetNonSlippingBC(self, i: int, j: int, k: int):
        nodeId = self.GetNodeID(i, j, k)
        self.BCType[nodeId, self.BoundaryNum[nodeId]] = 1
        self.BoundaryNum[nodeId] += 1

    @ti.func
    def NonSlippingBC(self, ng):
        self.f[ng] = ti.Matrix.zero(float, 3)
        self.mv[ng] = ti.Matrix.zero(float, 3)

    @ti.kernel
    def SetSlippingBC(self, i: int, j: int, k: int, norm: ti.types.vector(3, int)):
        nodeId = self.GetNodeID(i, j, k)
        self.BCType[nodeId, self.BoundaryNum[nodeId]] = 2
        self.Norm[nodeId, self.BoundaryNum[nodeId]] = norm
        self.BoundaryNum[nodeId] += 1

    @ti.func
    def SlippingBC(self, ng, b):
        if self.mv[ng].dot(self.Norm[ng, b]) > 0:
            self.f[ng] = self.f[ng] - self.f[ng].dot(self.Norm[ng, b]) * self.Norm[ng, b]
            self.mv[ng] = self.mv[ng] - self.mv[ng].dot(self.Norm[ng, b]) * self.Norm[ng, b]
   
    @ti.kernel
    def SetFrictionBC(self, i: int, j: int, k: int, miu: float, norm: ti.types.vector(3, int)):
        nodeId = self.GetNodeID(i, j, k)
        self.BCType[nodeId, self.BoundaryNum[nodeId]] = 3
        self.Norm[nodeId, self.BoundaryNum[nodeId]] = norm
        self.miu[nodeId, self.BoundaryNum[nodeId]] = miu
        self.BoundaryNum[nodeId] += 1

    @ti.func
    def FrictionBC(self, ng, b):
        mvNorm = self.mv[ng].dot(self.Norm[ng, b])
        if mvNorm > 0:
            vt0 = self.v[ng] - self.v[ng].dot(self.Norm[ng, b]) * self.Norm[ng, b]
            mvn0 = mvNorm * self.Norm[ng, b]
            mvt0 = self.mv[ng] - mvn0
            tang = mvt0.normalized()
            fmiu = -sign(vt0.dot(tang)) * self.miu[ng, b] * self.f[ng].dot(self.Norm[ng, b]) * tang
            mvt1 = mvt0 + fmiu * self.dt

            if mvt0.dot(mvt1) > 0:
                self.f[ng] = self.f[ng].dot(tang) * tang + fmiu
                self.mv[ng] = mvt1
            else:
                self.f[ng] = ti.Matrix.zero(float, 3)
                self.mv[ng] = ti.Matrix.zero(float, 3)
    
    @ti.kernel
    def SetNewmanBC(self, field: ti.template(), f: ti.types.vector(3, float)):
        for i in field:
            ID = field[i][1]
            self.lp.fex[ID] = f

    @ti.kernel
    def FindTractionBoundary(self, idxmin: int, idxmax: int, idymin: int, idymax: int, idzmin: int, idzmax: int, field: ti.template(), nb: int):
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
    def FindNode(self, xp):
        minLocalNodeId, maxLocalNodeId = ti.Matrix.zero(int, 3), ti.Matrix.zero(int, 3)
        if ti.static(self.shapeFunction == 0): minLocalNodeId, maxLocalNodeId = self.FindNodeLinear(xp)
        elif ti.static(self.shapeFunction == 2): minLocalNodeId, maxLocalNodeId = self.FindNodeBSplineQ(xp)
        elif ti.static(self.shapeFunction == 3): minLocalNodeId, maxLocalNodeId = self.FindNodeBSplineC(xp)
        return minLocalNodeId, maxLocalNodeId

    @ti.func
    def FindNodeLinear(self, xp):
        minLocalNodeId = ti.max(int(xp / self.dx), 0)
        maxLocalNodeId = int(ti.min(minLocalNodeId + 1, self.gnum - 1))
        return minLocalNodeId, maxLocalNodeId

    @ti.func
    def FindNodeGIMP(self, xp, psize):
        minLocalNodeId = ti.max(ti.ceil((xp - psize - self.dx) / self.dx, int), 0)
        maxLocalNodeId = ti.min(ti.floor((xp + psize + self.dx) / self.dx, int), self.gnum - 1)
        return minLocalNodeId, maxLocalNodeId

    @ti.func
    def FindNodeBSplineQ(self, xp):
        minLocalNodeId = ti.max(int(xp / self.dx - 0.5), 0)
        maxLocalNodeId = int(ti.min(minLocalNodeId + 2, self.gnum - 1))
        return minLocalNodeId, maxLocalNodeId

    @ti.func
    def FindNodeBSplineC(self, xp):
        minLocalNodeId = ti.max(int(xp / self.dx) - 1, 0)
        maxLocalNodeId = int(ti.min(minLocalNodeId + 3, self.gnum - 1))
        return minLocalNodeId, maxLocalNodeId

