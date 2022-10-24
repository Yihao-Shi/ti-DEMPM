import taichi as ti
import MPMLib2D.MPM_cells as MPMCell


@ti.data_oriented
class CellList(MPMCell.CellList):
    def __init__(self, domain, dx):
        super().__init__(domain, dx)
        self.isFluid = ti.field(int, self.cellSum)                                            # flags
        self.porosity = ti.field(float, self.cellSum)                                         # Porosity
    
    # ======================================== MPM Cell Initialization ======================================== #
    @ti.kernel
    def CellInit(self):
        for nc in self.id:
            ig, jg = self.GetCellIndex(nc)
            pos = (ti.Vector([ig, jg]) + 0.5) * self.Dx
            self.id[nc] = nc
            self.x[nc] = pos
            self.isFluid[nc] = 1

    # =========================================== MPM Cell Reset ============================================== #
    @ti.func
    def CellReset(self, nc):
        self.isFluid[nc] = 1
        self.P[nc] = 0.
        self.porosity[nc] = 0.
        self.vol[nc] = 0.

    # ========================================================= #
    #                                                           #
    #                      Free Surface                         #
    #                                                           #
    # ========================================================= #

