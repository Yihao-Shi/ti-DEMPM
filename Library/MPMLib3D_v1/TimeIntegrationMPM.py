import MPMLib2D_v1.Graphic as graphic
import MPMLib2D_v1.Spying as spy
import MPMLib2D_v1.InterpolationUSF as USF
import MPMLib2D_v1.InterpolationUSL as USL
import MPMLib2D_v1.InterpolationMUSL as MUSL
import MPMLib2D_v1.InterpolationGIMP as GIMP
import MPMLib2D_v1.InterpolationMLSMPM as MLSMPM
import time


def Output(t, step, printNum, mpm, start, vtkPath, ascPath):
    print('-------------------------- Time Step = ', step, '---------------------------')
    print('Simulation time = ', t)
    print('Time step = ', mpm.Dt[None])
    print('Physical time = ', time.time() - start)
    graphic.WriteFileVTK_MPM(mpm.lp, mpm.lg, printNum, vtkPath)
    spy.MonitorMPM(mpm.lp, printNum, ascPath)
    print('------------------------------ Running ---------------------------------')


# Solvers
def Solver(mpm, TIME, saveTime, CFL, vtkPath, ascPath, adaptive):
    if mpm.Algorithm == 0:
        SolverUSF(mpm, TIME, saveTime, CFL, vtkPath, ascPath, adaptive)
    elif mpm.Algorithm == 1:
        SolverUSL(mpm, TIME, saveTime, CFL, vtkPath, ascPath, adaptive)
    elif mpm.Algorithm == 2:
        SolverMUSL(mpm, TIME, saveTime, CFL, vtkPath, ascPath, adaptive)
    elif mpm.Algorithm == 3:
        SolverGIMP(mpm, TIME, saveTime, CFL, vtkPath, ascPath, adaptive)
    elif mpm.Algorithm == 4:
        SolverMLSMPM(mpm, TIME, saveTime, CFL, vtkPath, ascPath, adaptive)


#  ------------------------------------------ DMPM --------------------------------------------- #
def SolverUSF(mpm, TIME, saveTime, CFL, vtkPath, ascPath, adaptive):
    t, step, printNum = 0., 0, 0
    start = time.time()
    while t <= TIME:
        if t == 0:
            Output(t, step, printNum, mpm, start, vtkPath, ascPath)
            printNum += 1
            
        USF.GridReset(mpm)
        USF.ParticleToGrid_Momentum(mpm)
        USF.GridVelocity(mpm)
        USF.UpdateStressStrain(mpm)
        USF.ParticleToGrid_Force(mpm)
        USF.GridMomentum(mpm)
        USF.GridToParticle(mpm)

        if t % saveTime < mpm.Dt[None]:
            Output(t, step, printNum, mpm, start, vtkPath, ascPath)
            printNum += 1

        t += mpm.Dt[None]
        step += 1

        for nb in range(mpm.BodyInfo.shape[0]):
            if t % mpm.BodyInfo[nb].DT[1] < mpm.Dt[None] and mpm.BodyInfo[nb].DT[0] <= t <= mpm.BodyInfo[nb].DT[2]:
                mpm.AddParticlesInRun(nb)
        if adaptive:
            mpm.AdaptiveTimeScheme(CFL)


def SolverUSL(mpm, TIME, saveTime, CFL, vtkPath, ascPath, adaptive):
    t, step, printNum = 0., 0, 0
    start = time.time()
    while t <= TIME:
        if t == 0:
            Output(t, step, printNum, mpm, start, vtkPath, ascPath)
            printNum += 1

        USL.GridReset(mpm)
        USL.ParticleToGrid_Momentum(mpm)
        USL.ParticleToGrid_Force(mpm)
        USL.GridMomentum(mpm)
        USL.GridToParticle(mpm)
        USL.UpdateStressStrain(mpm)

        if t % saveTime < mpm.Dt[None]:
            Output(t, step, printNum, mpm, start, vtkPath, ascPath)
            printNum += 1

        t += mpm.Dt[None]
        step += 1

        for nb in range(mpm.BodyInfo.shape[0]):
            if t % mpm.BodyInfo[nb].DT[1] < mpm.Dt[None] and mpm.BodyInfo[nb].DT[0] <= t <= mpm.BodyInfo[nb].DT[2]:
                mpm.AddParticlesInRun(nb)
        if adaptive:
            mpm.AdaptiveTimeScheme(CFL)



def SolverMUSL(mpm, TIME, saveTime, CFL, vtkPath, ascPath, adaptive):
    t, step, printNum = 0., 0, 0
    start = time.time()
    while t <= TIME:
        if t == 0:
            Output(t, step, printNum, mpm, start, vtkPath, ascPath)
            printNum += 1

        MUSL.GridReset(mpm)
        MUSL.ParticleToGrid_Momentum(mpm)
        MUSL.ParticleToGrid_Force(mpm)
        MUSL.GridMomentum(mpm)
        MUSL.GridToParticle(mpm)
        MUSL.NodalMomentumMUSL(mpm)
        MUSL.UpdateStressStrain(mpm)

        if t % saveTime < mpm.Dt[None]:
            Output(t, step, printNum, mpm, start, vtkPath, ascPath)
            printNum += 1

        t += mpm.Dt[None]
        step += 1

        for nb in range(mpm.BodyInfo.shape[0]):
            if t % mpm.BodyInfo[nb].DT[1] < mpm.Dt[None] and mpm.BodyInfo[nb].DT[0] <= t <= mpm.BodyInfo[nb].DT[2]:
                mpm.AddParticlesInRun(nb)
        if adaptive:
            mpm.AdaptiveTimeScheme(CFL)


#  ------------------------------------------ GIMP --------------------------------------------- #
def SolverGIMP(mpm, TIME, saveTime, CFL, vtkPath, ascPath, adaptive):
    t, step, printNum = 0., 0, 0
    start = time.time()
    while t <= TIME:
        if t == 0:
            Output(t, step, printNum, mpm, start, vtkPath, ascPath)
            printNum += 1
        
        GIMP.GridReset(mpm)
        GIMP.ParticleToGrid_Momentum(mpm)
        GIMP.ParticleToGrid_Force(mpm)
        GIMP.GridMomentum(mpm)
        GIMP.GridToParticle(mpm)
        GIMP.UpdateStressStrain(mpm)

        if t % saveTime < mpm.Dt[None]:
            Output(t, step, printNum, mpm, start, vtkPath, ascPath)
            printNum += 1


        t += mpm.Dt[None]
        step += 1

        for nb in range(mpm.BodyInfo.shape[0]):
            if t % mpm.BodyInfo[nb].DT[1] < mpm.Dt[None] and mpm.BodyInfo[nb].DT[0] <= t <= mpm.BodyInfo[nb].DT[2]:
                mpm.AddParticlesInRun(nb)
        if adaptive:
            mpm.AdaptiveTimeScheme(CFL)


#  ----------------------------------------- MLS-MPM -------------------------------------------- #
def SolverMLSMPM(mpm, TIME, saveTime, CFL, vtkPath, ascPath, adaptive):
    t, step, printNum = 0., 0, 0
    start = time.time()
    while t <= TIME:
        if t == 0:
            Output(t, step, printNum, mpm, start, vtkPath, ascPath)
            printNum += 1
        
        MLSMPM.ParticleToGridAPIC(mpm)
        MLSMPM.GridOperationAPIC(mpm)
        MLSMPM.GridToParticleAPIC(mpm)

        if t % saveTime < mpm.Dt[None]:
            Output(t, step, printNum, mpm, start, vtkPath, ascPath)
            printNum += 1

        t += mpm.Dt[None]
        step += 1

        for nb in range(mpm.BodyInfo.shape[0]):
            if t % mpm.BodyInfo[nb].DT[1] < mpm.Dt[None] and mpm.BodyInfo[nb].DT[0] <= t <= mpm.BodyInfo[nb].DT[2]:
                mpm.AddParticlesInRun(nb)
        if adaptive:
            mpm.AdaptiveTimeScheme(CFL)
