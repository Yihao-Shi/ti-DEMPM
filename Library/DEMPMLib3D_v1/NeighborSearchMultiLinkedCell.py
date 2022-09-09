import taichi as ti


@ti.data_oriented
class NeighborSearchMultiLinkedCell:
    def __init__(self, MPMdomain, MPMgridsize, DEMgridsize, SmoothRange, max_potential_dempm_particle_pairs, MPMpartList, DEMpartList, DEMneighborList, DEMPMcontPair):
        self.MPMgridsize = MPMgridsize
        self.DEMgridsize = DEMgridsize
        self.SmoothRange = SmoothRange
        self.MPMcnum = ti.ceil(ti.Vector([MPMdomain[0] / self.MPMgridsize, MPMdomain[1] / self.MPMgridsize, MPMdomain[2] / self.MPMgridsize]))             
        if not all(MPMdomain % self.MPMgridsize == 0): print("Warning: The computational domain is suggested to be an integer multiple of the grid size")

        self.MPMcellSum = self.MPMcnum[0] * self.MPMcnum[1] * self.MPMcnum[2]
        if self.MPMcnum[2] == 0:
            self.MPMcellSum = self.MPMcnum[0] * self.MPMcnum[1]
            if self.MPMcnum[1] == 0:
                self.MPMcellSum = self.MPMcnum[0]

        self.id = ti.field(int, self.MPMcellSum)                                                       
        self.x = ti.Vector.field(3, float, self.MPMcellSum)                                       
        self.MPMpartList = MPMpartList
        self.DEMpartList = DEMpartList
        self.DEMneighborList = DEMneighborList
        self.DEMPMcontPair = DEMPMcontPair

        # ==================================================== Particles ============================================================ #
        self.MPMListHead = ti.field(int, self.MPMcellSum)
        self.MPMListCur = ti.field(int, self.MPMcellSum)
        self.MPMListTail = ti.field(int, self.MPMcellSum)

        self.MPMGrainCount = ti.field(int, self.MPMcellSum)
        self.MPMColumnSum = ti.field(int, (self.MPMcnum[1] * self.MPMcnum[2]))
        self.MPMPrefixSum = ti.field(int, self.MPMcellSum) 
        self.MPMParticleID = ti.field(int, self.MPMpartList.particleNum[None])

        self.potentialListP2M = ti.Struct.field({                                 # List of potential particle list
            "end1": int,                                                        
            "end2": int,                                                                                      
        }, shape=(max_potential_dempm_particle_pairs,))
        self.p2m = ti.field(int, ())

        # ==================================================== Neighbor Cell ============================================================ #
        self.target_cell = ti.Vector.field(3, int, 13)
        self.target_cell[0], self.target_cell[1], self.target_cell[2] = ti.Vector([0, 0, -1]), ti.Vector([-1, 0, -1]), ti.Vector([-1, 0, 0])
        self.target_cell[3], self.target_cell[4], self.target_cell[5] = ti.Vector([-1, 0, 1]), ti.Vector([0, -1, -1]), ti.Vector([0, -1, 0])
        self.target_cell[6], self.target_cell[7], self.target_cell[8] = ti.Vector([0, -1, 1]), ti.Vector([-1, -1, -1]), ti.Vector([-1, -1, 0])
        self.target_cell[9], self.target_cell[10], self.target_cell[11], self.target_cell[12]= ti.Vector([-1, -1, 1]), ti.Vector([1, -1, -1]), ti.Vector([1, -1, 0]), ti.Vector([1, -1, 1])

        
    # ========================================================= #
    #                                                           #
    #                  Get Cell ID & Index                      #
    #                                                           #
    # ========================================================= # 
    @ti.func
    def GetCellIndex(self, nc):
        ig = (nc % (self.MPMcnum[0] * self.MPMcnum[1])) % self.MPMcnum[0]
        jg = (nc % (self.MPMcnum[0] * self.MPMcnum[1])) // self.MPMcnum[0]
        kg = nc // (self.MPMcnum[0] * self.MPMcnum[1])
        return ig, jg, kg

    @ti.func
    def GetCellID(self, i, j, k):
        return int(i + j * self.MPMcnum[0] + k * self.MPMcnum[0] * self.MPMcnum[1])

    @ti.func
    def GetCellIDyz(self, j, k):
        return int(j + k * self.MPMcnum[1])
    
    # ======================================== DEM Cell Initialization ======================================== #
    @ti.kernel
    def CellInit(self):
        for nc in self.id:
            ig, jg, kg = self.GetCellIndex(nc)
            pos = (ti.Vector([ig, jg, kg]) + 0.5) * self.MPMgridsize
            self.id[nc] = nc
            self.x[nc] = pos

    @ti.kernel
    def SumMPMParticles(self):
        for cellID in range(self.MPMcellSum):
            self.MPMGrainCount[cellID] = 0

        for np in range(self.MPMpartList.particleNum[None]):
            cellID = self.GetCellID(self.MPMpartList.x[np][0] // self.MPMgridsize, self.MPMpartList.x[np][1] // self.MPMgridsize, self.MPMpartList.x[np][2] // self.MPMgridsize)
            self.MPMGrainCount[cellID] += 1
            self.MPMpartList.cellID[np] = cellID

        for celly in range(self.MPMcnum[1]):
            for cellz in range(self.MPMcnum[2]):
                ParticleInRow = 0
                for cellx in range(self.MPMcnum[0]):
                    cellID = self.GetCellID(cellx, celly, cellz)
                    ParticleInRow += self.MPMGrainCount[cellID]
                cellIDyz = self.GetCellIDyz(celly, cellz)
                self.MPMColumnSum[cellIDyz] = ParticleInRow

    @ti.kernel
    def BoardMPMNeighborList(self):
        ti.loop_config(serialize=True)
        self.MPMPrefixSum[0] = 0
        for cellz in range(self.MPMcnum[2]):
            for celly in range(self.MPMcnum[1]):
                cellID = self.GetCellID(0, celly, cellz)
                cellIDyz = self.GetCellIDyz(celly, cellz)
                if cellID > 0 and cellIDyz > 0:
                    self.MPMPrefixSum[cellID] = self.MPMPrefixSum[cellID - self.MPMcnum[0]] + self.MPMColumnSum[cellIDyz - 1]
                   

        for cellx in range(self.MPMcnum[0]):
            for celly in range(self.MPMcnum[1]):
                for cellz in range(self.MPMcnum[2]): 
                    cellID = self.GetCellID(cellx, celly, cellz)
                    if cellx == 0:
                        self.MPMPrefixSum[cellID] += self.MPMGrainCount[cellID]
                    else:
                        self.MPMPrefixSum[cellID] = self.MPMPrefixSum[cellID - 1] + self.MPMGrainCount[cellID]
                    
                    self.MPMListHead[cellID] = self.MPMPrefixSum[cellID] - self.MPMGrainCount[cellID]
                    self.MPMListCur[cellID] = self.MPMListHead[cellID]
                    self.MPMListTail[cellID] = self.MPMPrefixSum[cellID]

        for np in range(self.MPMpartList.particleNum[None]):
            cellID = self.MPMpartList.cellID[np]
            grain_location = ti.atomic_add(self.MPMListCur[cellID], 1)
            self.MPMParticleID[grain_location] = np


    @ti.kernel
    def BoardSearchP2M(self):
        self.p2m[None] = 0
        for partDEM in range(self.DEMpartList.particleNum[None]):
            grid_idx = ti.floor(self.DEMpartList.x[partDEM] / self.MPMgridsize, int)
            extended = ti.ceil(self.SmoothRange / self.MPMgridsize, int)
            ratio = ti.ceil(self.DEMgridsize / self.MPMgridsize, int)
            neighbor_range = extended + ratio

            x_begin = max(grid_idx[0] - neighbor_range, 0)
            x_end = min(grid_idx[0] + neighbor_range + 1, self.MPMcnum[0])
            y_begin = max(grid_idx[1] - neighbor_range, 0)
            y_end = min(grid_idx[1] + neighbor_range + 1, self.MPMcnum[1])
            z_begin = max(grid_idx[2] - neighbor_range, 0)
            z_end = min(grid_idx[2] + neighbor_range + 1, self.MPMcnum[2])

            for neigh_i in range(x_begin, x_end):
                for neigh_j in range(y_begin, y_end): 
                    for neigh_k in range(z_begin, z_end): 
                        cellID = self.GetCellID(neigh_i, neigh_j, neigh_k)
                        for p_idx in range(self.MPMListHead[cellID], self.MPMListTail[cellID]):
                            partMPM = self.MPMParticleID[p_idx]
                            count_pairs = ti.atomic_add(self.p2m[None], 1)
                            self.potentialListP2M[count_pairs].end1 = partDEM
                            self.potentialListP2M[count_pairs].end2 = partMPM

    @ti.kernel
    def FineSearchP2M(self): 
        for p2m in range(self.p2m[None]):
            end1 = self.potentialListP2M[p2m].end1
            end2 = self.potentialListP2M[p2m].end2        
            pos1 = self.DEMpartList.x[end1]
            pos2 = self.MPMpartList.x[end2]  
            rad1 = self.DEMpartList.rad[end1]  
            rad2 = self.MPMpartList.rad[end2]    
            if (pos2 - pos1).norm() < rad1 + rad2:                
                self.DEMPMcontPair.Contact(end1, end2, pos1, pos2, rad1, rad2)

