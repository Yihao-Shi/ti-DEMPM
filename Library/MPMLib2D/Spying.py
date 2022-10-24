import numpy as np
import os


def MonitorMPM(partList, printNum, ascPath):
    print('---------------------', 'Writing MPM Monitor files ', printNum, '---------------------')
    if printNum == 0:
        if not os.path.exists(ascPath):
            os.mkdir(ascPath)

    filename = os.path.join(ascPath, 'MonitorMPM%06d.npz' % printNum)
    pos_x = np.ascontiguousarray(partList.x.to_numpy()[0:partList.particleNum[None], 0])
    pos_y = np.ascontiguousarray(partList.x.to_numpy()[0:partList.particleNum[None], 1])
    epeff = np.ascontiguousarray(partList.plasticStrainEff.to_numpy()[0:partList.particleNum[None]])

    np.savez(filename, pos_x=pos_x, pos_y=pos_y, epeff=epeff)
