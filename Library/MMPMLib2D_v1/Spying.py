import numpy as np
import os


def MonitorMPM(Solid, Fluid, printNum, ascPath):
    print('---------------------', 'Writing MPM Monitor files ', printNum, '---------------------')
    if printNum == 0:
        if not os.path.exists(ascPath):
            os.mkdir(ascPath)

    filename1 = os.path.join(ascPath, f'MonitorMPMSolid{printNum:06d}.npz')
    pos_x = np.ascontiguousarray(Solid.x.to_numpy()[0:Solid.particleNum[None], 0])
    pos_y = np.ascontiguousarray(Solid.x.to_numpy()[0:Solid.particleNum[None], 1])
    p = np.ascontiguousarray(Solid.P.to_numpy()[0:Solid.particleNum[None]])

    np.savez(filename1, pos_x=pos_x, pos_y=pos_y, p=p)

    filename2 = os.path.join(ascPath, f'MonitorMPMFluid{printNum:06d}.npz')
    pos_x = np.ascontiguousarray(Fluid.x.to_numpy()[0:Fluid.particleNum[None], 0])
    pos_y = np.ascontiguousarray(Fluid.x.to_numpy()[0:Fluid.particleNum[None], 1])
    p = np.ascontiguousarray(Fluid.P.to_numpy()[0:Fluid.particleNum[None]])

    np.savez(filename2, pos_x=pos_x, pos_y=pos_y, p=p)
