import MMPMLib2D_v1.Graphic as graphic
import MMPMLib2D_v1.Spying as spy
import MMPMLib2D_v1.InterpolationMUSL as MUSL
import MMPMLib2D_v1.InterpolationGIMP as GIMP
import MMPMLib2D_v1.InterpolationMLSMPM as MLSMPM
import time


def Output(t, step, printNum, mmpm, start, vtkPath, ascPath):
    print('-------------------------- Time Step = ', step, '---------------------------')
    print('Simulation time = ', t)
    print('Time step = ', mmpm.Dt[None])
    print('Physical time = ', time.time() - start)
    graphic.WriteFileVTK_MPM(mmpm.lps, mmpm.lpf, mmpm.lg, printNum, vtkPath)
    spy.MonitorMPM(mmpm.lps, mmpm.lpf, printNum, ascPath)
    print('------------------------------ Running ---------------------------------')


# Solvers
def Solver(mmpm, TIME, saveTime, CFL, vtkPath, ascPath, adaptive):
    if mmpm.Algorithm == 2:
        SolverMUSL(mmpm, TIME, saveTime, CFL, vtkPath, ascPath, adaptive)
    elif mmpm.Algorithm == 3:
        SolverGIMP(mmpm, TIME, saveTime, CFL, vtkPath, ascPath, adaptive)
    elif mmpm.Algorithm == 4:
        SolverMLSMPM(mmpm, TIME, saveTime, CFL, vtkPath, ascPath, adaptive)


#  ------------------------------------------ DMPM --------------------------------------------- #
def SolverMUSL(mmpm, TIME, saveTime, CFL, vtkPath, ascPath, adaptive):
    t, step, printNum = 0., 0, 0
    start = time.time()
    while t <= TIME:
        if t == 0:
            Output(t, step, printNum, mmpm, start, vtkPath, ascPath)
            printNum += 1
        
        MUSL.GridReset(mmpm)
        MUSL.ParticleToGridSolid(mmpm)
        MUSL.GridInfoUpdateSolid(mmpm)

        MUSL.ParticleToGridHydro(mmpm)
        MUSL.GridInfoUpdateHydro(mmpm)
        MUSL.GridToParticle_Porosity(mmpm)
        MUSL.FluidPointInit(mmpm)

        MUSL.ParticleToGridFuild(mmpm)
        MUSL.GridInfoUpdateFluid(mmpm)
        MUSL.ParticleToGridFluidForce(mmpm)

        MUSL.ParticleToGridMultiplier(mmpm)
        MUSL.GridDragForce(mmpm)

        MUSL.ParticleToGridSolidForce(mmpm)

        MUSL.GridAccerlation(mmpm)
        MUSL.GridMomentumSolid(mmpm)
        MUSL.GridMomentumFluid(mmpm)

        MUSL.GridToParticleSolid(mmpm)
        MUSL.GridToParticleFluid(mmpm)

        MUSL.NodalMomentumMUSLSolid(mmpm)
        MUSL.NodalMomentumMUSLFluid(mmpm)

        MUSL.UpdateStressStrainSolid(mmpm)
        MUSL.UpdateStressStrainFluid(mmpm)

        MUSL.CellAverageTech(mmpm)

        if t % saveTime < mmpm.Dt[None]:
            Output(t, step, printNum, mmpm, start, vtkPath, ascPath)
            printNum += 1

        t += mmpm.Dt[None]
        step += 1


#  ------------------------------------------ GIMP --------------------------------------------- #
def SolverGIMP(mmpm, TIME, saveTime, CFL, vtkPath, ascPath, adaptive):
    t, step, printNum = 0., 0, 0
    start = time.time()
    while t <= TIME:
        if t == 0:
            Output(t, step, printNum, mmpm, start, vtkPath, ascPath)
            printNum += 1
        
        GIMP.GridReset(mmpm)
        GIMP.ParticleToGridSolid(mmpm)
        GIMP.GridInfoUpdateSolid(mmpm)

        GIMP.ParticleToGridHydro(mmpm)
        GIMP.GridInfoUpdateHydro(mmpm)
        GIMP.GridToParticle_Porosity(mmpm)
        GIMP.FluidPointInit(mmpm)

        GIMP.ParticleToGridFluid(mmpm)
        GIMP.GridInfoUpdateFluid(mmpm)
        GIMP.ParticleToGridFluidForce(mmpm)

        GIMP.GridDragForce(mmpm)

        GIMP.ParticleToGridSolidForce(mmpm)

        GIMP.GridAccerlation(mmpm)
        GIMP.GridMomentumSolid(mmpm)
        GIMP.GridMomentumFluid(mmpm)

        GIMP.GridToParticleSolid(mmpm)
        GIMP.GridToParticleFluid(mmpm)

        GIMP.UpdateStressStrainSolid(mmpm)
        GIMP.UpdateStressStrainFluid(mmpm)

        #GIMP.CellAverageTech(mmpm)

        if t % saveTime < mmpm.Dt[None]:
            Output(t, step, printNum, mmpm, start, vtkPath, ascPath)
            printNum += 1

        t += mmpm.Dt[None]
        step += 1


#  ----------------------------------------- MLS-MPM -------------------------------------------- #
def SolverMLSMPM(mmpm, TIME, saveTime, CFL, vtkPath, ascPath, adaptive):
    t, step, printNum = 0., 0, 0
    start = time.time()
    while t <= TIME:
        if t == 0:
            Output(t, step, printNum, mmpm, start, vtkPath, ascPath)
            printNum += 1

        MLSMPM.GridReset(mmpm)
        MLSMPM.ParticleToGrid_Momentum(mmpm)
        MLSMPM.ParticleToGrid_Force(mmpm)
        MLSMPM.GridMomentum(mmpm)
        MLSMPM.GridToParticle(mmpm)
        MLSMPM.UpdateStressStrain(mmpm)
        #print(mmpm.lpf.J[346])

        if t % saveTime < mmpm.Dt[None]:
            Output(t, step, printNum, mmpm, start, vtkPath, ascPath)
            printNum += 1

        t += mmpm.Dt[None]
        step += 1
