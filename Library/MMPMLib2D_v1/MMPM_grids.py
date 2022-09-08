import taichi as ti
from MPMLib2D_v1.Function import *
import MPMLib2D_v1.MPM_grids as MPMGrid


@ti.data_oriented
class GridList(MPMGrid.GridList):
    def __init__(self, domain, dx, ContactDetection):
        super().__init__(domain, dx, ContactDetection)
        self.mw = ti.field(float, self.gridSum)
        self.vw = ti.Vector.field(2, float, self.gridSum)
        self.mvw = ti.Vector.field(2, float, self.gridSum)
        self.fw = ti.Vector.field(2, float, self.gridSum)

        self.fd = ti.Vector.field(2, float, self.gridSum)
        self.fIntf = ti.Vector.field(2, float, self.gridSum)
        self.vol = ti.field(float, self.gridSum)
        self.poros = ti.field(float, self.gridSum)
        self.kh = ti.field(float, self.gridSum)
        self.P = ti.field(float, self.gridSum)

        self.BCTypeFluid = ti.field(int, shape=(self.gridSum, 4))                               
        self.BoundaryNumFluid = ti.field(int, shape=(self.gridSum,))                             
        self.NormFluid = ti.Vector.field(2, float, shape=(self.gridSum, 4))                       
        self.miuFluid = ti.field(float, shape=(self.gridSum, 4))                                 

    # =========================================== MPM Grid Reset ============================================== #                                   
    @ti.func
    def GridReset(self, ng):
        if self.m[ng] > 0:
            self.m[ng] = 0.
            self.v[ng] = ti.Matrix.zero(float, 2)
            self.mv[ng] = ti.Matrix.zero(float, 2)
            self.f[ng] = ti.Matrix.zero(float, 2)

            self.fd[ng] = ti.Matrix.zero(float, 2)
            self.fIntf[ng] = ti.Matrix.zero(float, 2)
            self.P[ng] = 0.
            self.poros[ng] = 0.
            self.kh[ng] = 0.
        
        if self.mw[ng] > 0:
            self.mw[ng] = 0.
            self.vw[ng] = ti.Matrix.zero(float, 2)
            self.mvw[ng] = ti.Matrix.zero(float, 2)
            self.fw[ng] = ti.Matrix.zero(float, 2)

    # ========================================== Boundary Condition =========================================== #
    @ti.func
    def ApplyBoundaryConditionFluid(self, ng, dt):
        for b in range(self.BoundaryNumFluid[ng]):
            if self.BCTypeFluid[ng, b] == 1:
                self.NonSlippingBCFluid(ng)
            elif self.BCTypeFluid[ng, b] == 2:
                self.SlippingBCFluid(ng, b)
            elif self.BCTypeFluid[ng, b] == 3:
                self.FrictionBCFluid(ng, dt, b)
    
    @ti.kernel
    def SetNonSlippingBCFluid(self, i: int, j: int):
        nodeId = int(i + j * self.gnum[0])
        self.BCTypeFluid[nodeId, self.BoundaryNumFluid[nodeId]] = 1
        self.BoundaryNumFluid[nodeId] += 1

    @ti.func
    def NonSlippingBCFluid(self, ng):    
        self.fw[ng] = ti.Matrix.zero(float, 2)
        self.mvw[ng] = ti.Matrix.zero(float, 2)

    @ti.kernel
    def SetSlippingBCFluid(self, i: int, j: int, norm: ti.types.vector(2, int)):
        nodeId = int(i + j * self.gnum[0])
        self.BCTypeFluid[nodeId, self.BoundaryNumFluid[nodeId]] = 2
        self.NormFluid[nodeId, self.BoundaryNumFluid[nodeId]] = norm
        self.BoundaryNumFluid[nodeId] += 1

    @ti.func
    def SlippingBCFluid(self, ng, b):
        if self.mvw[ng].dot(self.Norm[ng, b]) > 0:
            self.fw[ng] = self.fw[ng] - self.fw[ng].dot(self.Norm[ng, b]) * self.Norm[ng, b]
            self.mvw[ng] = self.mvw[ng] - self.mvw[ng].dot(self.Norm[ng, b]) * self.Norm[ng, b]
    
    @ti.kernel
    def SetFrictionBCFluid(self, i: int, j: int, miu: float, norm: ti.types.vector(2, int)):
        nodeId = int(i + j * self.gnum[0])
        self.BCTypeFluid[nodeId, self.BoundaryNumFluid[nodeId]] = 3
        self.NormFluid[nodeId, self.BoundaryNumFluid[nodeId]] = norm
        self.miuFluid[nodeId, self.BoundaryNumFluid[nodeId]] = miu
        self.BoundaryNumFluid[nodeId] += 1

    @ti.func
    def FrictionBCFluid(self, ng, dt, b):
        mvwNorm = self.mvw[ng].dot(self.Norm[ng, b])
        if mvwNorm > 0:
            vt0 = self.vw[ng] - self.vw[ng].dot(self.Norm[ng, b]) * self.Norm[ng, b]
            mvn0 = mvwNorm * self.Norm[ng, b]
            mvt0 = self.mvw[ng] - mvn0
            tang = mvt0.normalized()
            fmiu = -sign(vt0.dot(tang)) * self.miu[ng, b] * self.fw[ng].dot(self.Norm[ng, b]) * tang
            mvt1 = mvt0 + fmiu * dt

            if mvt0.dot(mvt1) > 0:
                self.fw[ng] = self.fw[ng].dot(tang) * tang + fmiu
                self.mvw[ng] = mvt1
            else:
                self.fw[ng] = ti.Matrix.zero(float, 2)
                self.mvw[ng] = ti.Matrix.zero(float, 2)
    
    # ================================================ Solve =================================================== #
    @ti.func
    def UpdateFluidMass(self, ng, mp):
        self.mw[ng] += mp

    @ti.func
    def UpdateFluidMomentumPIC(self, ng, mvp):
        self.mvw[ng] += mvp

    @ti.func
    def UpdateNodalPorosity(self, ng, nporos):
        self.poros[ng] += nporos

    @ti.func
    def UpdateNodalPermeability(self, ng, kh):
        self.kh[ng] += kh

    @ti.func
    def UpdateFluidForce(self, ng, fp):
        self.fw[ng] += fp

    @ti.func
    def ApplyGlobalDampingFluid(self, ng, damp):
        self.fw[ng] -= damp * self.fw[ng].norm() * Normalize(self.mvw[ng])

    @ti.func
    def ComputeFluidMomentum(self, ng, dt):
        self.mvw[ng] += self.fw[ng] * dt

    @ti.func
    def ComputeFluidVelocity(self, ng):
        self.vw[ng] = self.mvw[ng] / self.mw[ng]

    @ti.func
    def ComputeNodalPorosity(self, ng):
        self.poros[ng] = 1 - self.poros[ng] / self.m[ng]

    @ti.func
    def ComputeNodalPermeability(self, ng):
        self.kh[ng] = self.kh[ng] / self.m[ng]
    
    @ti.func
    def ComputeDragForce(self, ng, grav):
        self.fd[ng] = (self.vw[ng] - self.v[ng]) * self.m[ng] * self.poros[ng] * grav / self.kh[ng]

    @ti.func
    def FluidForceAssemble(self, ng):
        self.fw[ng] -= self.fd[ng]

    @ti.func
    def SolidForceAssemble(self, ng, grav):
        self.f[ng] += self.fIntf[ng] + self.fd[ng]

    @ti.func
    def StorePorePressure(self, ng, dfInt):
        self.fIntf[ng] += dfInt
