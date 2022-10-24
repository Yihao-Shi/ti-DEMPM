import taichi as ti
import MPMLib2D.MPM_materials as MPMMaterial
import math


@ti.data_oriented
class MaterialList(MPMMaterial.MaterialList):
    def __init__(self, max_material_num):
        super().__init__(max_material_num)
        
