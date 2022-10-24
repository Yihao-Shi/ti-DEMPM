import os
from pyevtk.hl import pointsToVTK, gridToVTK
import numpy as np


def WriteFileVTK_MPM(Solid, Fluid, gridList, printNum, vtkPath):
    print('---------------------', 'Writing MPM Graphic files ', printNum, '---------------------')

    if printNum == 0:
        if not os.path.exists(vtkPath):
            os.mkdir(vtkPath)

    pos_x = np.ascontiguousarray(Solid.x.to_numpy()[0:Solid.particleNum[None], 0])
    pos_y = np.ascontiguousarray(Solid.x.to_numpy()[0:Solid.particleNum[None], 1])
    pos_z = np.zeros(Solid.particleNum[None], dtype=np.float32)
    ID = np.ascontiguousarray(Solid.ID.to_numpy()[0:Solid.particleNum[None]])
    body = np.ascontiguousarray(Solid.bodyID.to_numpy()[0:Solid.particleNum[None]])
    x = np.ascontiguousarray(Solid.x.to_numpy()[0:Solid.particleNum[None]])
    vel_x = np.ascontiguousarray(Solid.v.to_numpy()[0:Solid.particleNum[None], 0])
    vel_y = np.ascontiguousarray(Solid.v.to_numpy()[0:Solid.particleNum[None], 1])
    rad = np.ascontiguousarray(Solid.rad.to_numpy()[0:Solid.particleNum[None]])
    epeff = np.ascontiguousarray(Solid.plasticStrainEff.to_numpy()[0:Solid.particleNum[None]])
    porosity = np.ascontiguousarray(Solid.poros.to_numpy()[0:Solid.particleNum[None]])
    Darcy_k = np.ascontiguousarray(Solid.kh.to_numpy()[0:Solid.particleNum[None]])
    vol = np.ascontiguousarray(Solid.vol.to_numpy()[0:Solid.particleNum[None]])
    Stress = Solid.stress.to_numpy()[0:Solid.particleNum[None]]
    stressXX = np.ascontiguousarray(np.array(Stress.flatten()[0:-1:4]))
    stressYY = np.ascontiguousarray(np.array(Stress.flatten()[3::4]))
    stressXY = np.ascontiguousarray(np.array(Stress.flatten()[1:-1:4]))
    pointsToVTK(vtkPath+f'/GraphicMPMSolid{printNum:06d}', pos_x, pos_y, pos_z, data={"vel_x": vel_x, "vel_y": vel_y, "porosity": porosity, "Darcy_k": Darcy_k,
                                                                                          "stressXX": stressXX, "stressXY": stressXY, "stressYY": stressYY, "vol": vol})

    pos_x = np.ascontiguousarray(Fluid.x.to_numpy()[0:Fluid.particleNum[None], 0])
    pos_y = np.ascontiguousarray(Fluid.x.to_numpy()[0:Fluid.particleNum[None], 1])
    pos_z = np.zeros(Fluid.particleNum[None], dtype=np.float32)
    ID = np.ascontiguousarray(Fluid.ID.to_numpy()[0:Fluid.particleNum[None]])
    body = np.ascontiguousarray(Fluid.bodyID.to_numpy()[0:Fluid.particleNum[None]])
    x = np.ascontiguousarray(Fluid.x.to_numpy()[0:Fluid.particleNum[None]])
    vel_x = np.ascontiguousarray(Fluid.v.to_numpy()[0:Fluid.particleNum[None], 0])
    vel_y = np.ascontiguousarray(Fluid.v.to_numpy()[0:Fluid.particleNum[None], 1])
    rad = np.ascontiguousarray(Fluid.rad.to_numpy()[0:Fluid.particleNum[None]])
    epeff = np.ascontiguousarray(Fluid.plasticStrainEff.to_numpy()[0:Fluid.particleNum[None]])
    porosity = np.ascontiguousarray(Fluid.poros.to_numpy()[0:Fluid.particleNum[None]])
    Darcy_k = np.ascontiguousarray(Fluid.kh.to_numpy()[0:Fluid.particleNum[None]])
    vol = np.ascontiguousarray(Fluid.vol.to_numpy()[0:Fluid.particleNum[None]])
    Stress = Fluid.stress.to_numpy()[0:Fluid.particleNum[None]]
    stressXX = np.ascontiguousarray(np.array(Stress.flatten()[0:-1:4]))
    stressYY = np.ascontiguousarray(np.array(Stress.flatten()[3::4]))
    stressXY = np.ascontiguousarray(np.array(Stress.flatten()[1:-1:4]))
    pointsToVTK(vtkPath+f'/GraphicMPMFluid{printNum:06d}', pos_x, pos_y, pos_z, data={"vel_x": vel_x, "vel_y": vel_y, "porosity": porosity, "Darcy_k": Darcy_k,
                                                                                          "stressXX": stressXX, "stressXY": stressXY, "stressYY": stressYY, "vol": vol})

    '''pos_x = np.ascontiguousarray(gridList.x.to_numpy()[0:gridList.gridSum, 0])
    pos_y = np.ascontiguousarray(gridList.x.to_numpy()[0:gridList.gridSum, 1])
    pos_z = np.zeros(gridList.gridSum)
    gridToVTK(vtkPath+f'/GridMPM{printNum:06d}', pos_x, pos_y, pos_z)'''
