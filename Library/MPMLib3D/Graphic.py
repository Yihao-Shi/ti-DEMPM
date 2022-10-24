import os
from pyevtk.hl import pointsToVTK
import numpy as np


def WriteFileVTK_MPM(partList, gridList, printNum, vtkPath):
    if printNum == 0:
        if not os.path.exists(vtkPath):
            os.mkdir(vtkPath)

    pos_x = np.ascontiguousarray(partList.x.to_numpy()[0:partList.particleNum[None], 0])
    pos_y = np.ascontiguousarray(partList.x.to_numpy()[0:partList.particleNum[None], 1])
    pos_z = np.ascontiguousarray(partList.x.to_numpy()[0:partList.particleNum[None], 2])
    ID = np.ascontiguousarray(partList.ID.to_numpy()[0:partList.particleNum[None]])
    body = np.ascontiguousarray(partList.bodyID.to_numpy()[0:partList.particleNum[None]])
    vel_x = np.ascontiguousarray(partList.v.to_numpy()[0:partList.particleNum[None], 0])
    vel_y = np.ascontiguousarray(partList.v.to_numpy()[0:partList.particleNum[None], 1])
    vel_z = np.ascontiguousarray(partList.v.to_numpy()[0:partList.particleNum[None], 1])
    rad = np.ascontiguousarray(partList.rad.to_numpy()[0:partList.particleNum[None]])
    epeff = np.ascontiguousarray(partList.plasticStrainEff.to_numpy()[0:partList.particleNum[None]])
    vol = np.ascontiguousarray(partList.vol.to_numpy()[0:partList.particleNum[None]])
    Stress = partList.stress.to_numpy()[0:partList.particleNum[None]]
    stressXX = np.ascontiguousarray(np.array(Stress.flatten()[0::9]))
    stressYY = np.ascontiguousarray(np.array(Stress.flatten()[4::9]))
    stressZZ = np.ascontiguousarray(np.array(Stress.flatten()[8::9]))
    pointsToVTK(vtkPath+f'/GraphicMPM{printNum:06d}', pos_x, pos_y, pos_z, data={"vel_x": vel_x, "vel_y": vel_y, "epeff": epeff, "vol": vol, 
                                                                                 "stressXX": stressXX, "stressYY": stressYY, "stressZZ": stressZZ})

