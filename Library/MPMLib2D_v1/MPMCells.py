import taichi as ti


@ti.data_oriented
class CellList:
    def __init__(self, domain, dx):
        self.Dx = dx
        self.cnum = ti.Vector([int(domain[0] / dx[0]), int(domain[1] / dx[1])])              # Grid Number
        self.cellSum = self.cnum[0] * self.cnum[1]
        self.cvol = dx[0] * dx[1]

        self.id = ti.field(int, self.cellSum)                                                 # ID of grids
        self.x = ti.Vector.field(2, float, self.cellSum)                                      # Position
        self.vol0 = ti.field(float, self.cellSum)                                             # Initial Volume
        self.vol = ti.field(float, self.cellSum)                                              # Volume
        self.J = ti.field(float, self.cellSum)                                                # Jacobian
        self.P = ti.field(float, self.cellSum)                                                # Pore Pressure

    # ========================================================= #
    #                                                           #
    #                  Get Cell ID & Index                      #
    #                                                           #
    # ========================================================= # 
    @ti.func
    def GetCellIndex(self, nc):
        ig = (nc % (self.cnum[0] * self.cnum[1])) % self.cnum[0]
        jg = (nc % (self.cnum[0] * self.cnum[1])) // self.cnum[0]
        return ig, jg

    @ti.func
    def GetCellID(self, i, j):
        return int(i + j * self.cnum[0])
    
    # ======================================== MPM Cell Initialization ======================================== #
    @ti.kernel
    def CellInit(self):
        for nc in self.id:
            ig, jg = self.GetCellIndex(nc)
            pos = (ti.Vector([ig, jg]) + 0.5) * self.Dx
            self.id[nc] = nc
            self.x[nc] = pos

    # =========================================== MPM Cell Reset ============================================== #
    @ti.func
    def CellReset(self, nc):
        if self.vol0[nc] > 0.:
            self.vol0[nc] = 0.
            self.vol[nc] = 0.
            self.P[nc] = 0.
    
    # ================================================ Solve =================================================== #
    # ========================================================= #
    #                                                           #
    #                       Cell Volume                         #
    #                                                           #
    # ========================================================= #
    @ti.func
    def CellVolumeInit(self, nc, pvol):
        self.vol0[nc] += pvol

    @ti.func
    def UpdateCellVolume(self, nc, pvol):
        self.vol[nc] += pvol
    
    # ========================================================= #
    #                                                           #
    #                  Cell Average Jacobian                    #
    #                                                           #
    # ========================================================= #
    @ti.func
    def ComputeCellJacobian(self, nc):
        if self.vol0[nc] > 0.: self.J[nc] = self.vol[nc] / self.vol0[nc]
    
    # ========================================================= #
    #                                                           #
    #                         Cell P                            #
    #                                                           #
    # ========================================================= #
    @ti.func
    def UpdatePorePressure(self, nc, pvol):
        self.P[nc] += pvol
    
    @ti.func
    def AveragePorePressure(self, nc):
        frac = self.vol[nc] / self.cvol
        if self.vol[nc] > 0. and frac > 0.2: 
            self.P[nc] /= self.vol[nc]
        elif frac <= 0.2:
            self.P[nc] = 0.
