import numpy as np
import os


def MonitorMPM(time, partList, printNum, ascPath):
    if printNum == 0:
        if not os.path.exists(ascPath):
            os.mkdir(ascPath)

    filename = os.path.join(ascPath, 'MonitorMPM%06d.npz' % printNum)
    pos_x = np.ascontiguousarray(partList.x.to_numpy()[0:partList.particleNum[None], 0])
    pos_y = np.ascontiguousarray(partList.x.to_numpy()[0:partList.particleNum[None], 1])
    pos_z = np.ascontiguousarray(partList.x.to_numpy()[0:partList.particleNum[None], 2])
    epeff = np.ascontiguousarray(partList.plasticStrainEff.to_numpy()[0:partList.particleNum[None]])

    np.savez(filename, t_current=time,
                       pos_x=pos_x, pos_y=pos_y, pos_z=pos_z, epeff=epeff)
