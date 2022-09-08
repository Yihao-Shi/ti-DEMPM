import os
from pyevtk.hl import pointsToVTK, gridToVTK
import numpy as np


def WriteFileVTK_MPM(partList, gridList, printNum, vtkPath):
    print('---------------------', 'Writing MPM Graphic files ', printNum, '---------------------')

    if printNum == 0:
        if not os.path.exists(vtkPath):
            os.mkdir(vtkPath)

    pos_x = np.ascontiguousarray(partList.x.to_numpy()[0:partList.particleNum[None], 0])
    pos_y = np.ascontiguousarray(partList.x.to_numpy()[0:partList.particleNum[None], 1])
    pos_z = np.zeros(partList.particleNum[None], dtype=np.float32)
    ID = np.ascontiguousarray(partList.ID.to_numpy()[0:partList.particleNum[None]])
    body = np.ascontiguousarray(partList.bodyID.to_numpy()[0:partList.particleNum[None]])
    x = np.ascontiguousarray(partList.x.to_numpy()[0:partList.particleNum[None]])
    vel_x = np.ascontiguousarray(partList.v.to_numpy()[0:partList.particleNum[None], 0])
    vel_y = np.ascontiguousarray(partList.v.to_numpy()[0:partList.particleNum[None], 1])
    rad = np.ascontiguousarray(partList.rad.to_numpy()[0:partList.particleNum[None]])
    epeff = np.ascontiguousarray(partList.plasticStrainEff.to_numpy()[0:partList.particleNum[None]])
    vol = np.ascontiguousarray(partList.vol.to_numpy()[0:partList.particleNum[None]])
    Stress = partList.stress.to_numpy()[0:partList.particleNum[None]]
    stressXX = np.ascontiguousarray(np.array(Stress.flatten()[0:-1:4]))
    stressYY = np.ascontiguousarray(np.array(Stress.flatten()[3::4]))
    stressXY = np.ascontiguousarray(np.array(Stress.flatten()[1:-1:4]))
    pointsToVTK(vtkPath+f'/GraphicMPM{printNum:06d}', pos_x, pos_y, pos_z, data={"vel_x": vel_x, "vel_y": vel_y, "epeff": epeff, "vol": vol, 
                                                                                 "stressXX": stressXX, "stressXY": stressXY, "stressYY": stressYY})

    '''pos_x = np.ascontiguousarray(gridList.x.to_numpy()[0:gridList.gridSum, 0])
    pos_y = np.ascontiguousarray(gridList.x.to_numpy()[0:gridList.gridSum, 1])
    pos_z = np.zeros(gridList.gridSum)
    gridToVTK(vtkPath+f'/GridMPM{printNum:06d}', pos_x, pos_y, pos_z)'''
