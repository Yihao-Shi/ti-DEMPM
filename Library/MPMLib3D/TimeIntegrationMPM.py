import taichi as ti
import MPMLib3D_v1.Graphic as Graphic
import MPMLib3D_v1.Spying as Spy
import time


@ti.data_oriented
class TimeIntegrationMPM:
    def __init__(self, mpm, TIME, saveTime, CFL, vtkPath, ascPath, adaptive):
        self.mpm = mpm
        self.TIME = TIME
        self.saveTime = saveTime
        self.CFL = CFL
        self.vtkPath = vtkPath
        self.ascPath = ascPath
        self.adaptive = adaptive

    def TurnOnSolver(self, t, step, printNum):
        self.t = t
        self.step = step
        self.printNum = printNum

    def FinalizeSolver(self):
        self.t = 0.
        self.step = 0
        self.printNum = 0

    def UpdateSolver(self, TIME, saveTime, CFL, vtkPath, ascPath, adaptive):
        self.TIME = TIME
        self.saveTime = saveTime
        self.CFL = CFL
        self.vtkPath = vtkPath
        self.ascPath = ascPath
        self.adaptive = adaptive

    def Output(self, start):
        print('# Step = ', self.step, '     ', 'Simulation time = ', self.t, '\n')
        Graphic.WriteFileVTK_MPM(self.mpm.partList, self.mpm.gridList, self.printNum, self.vtkPath)
        Spy.MonitorMPM(self.t, self.mpm.partList, self.printNum, self.ascPath)


#  ------------------------------------------ DMPM --------------------------------------------- #
@ti.data_oriented
class SolverUSF(TimeIntegrationMPM):
    def __init__(self, mpm, TIME, saveTime, CFL, vtkPath, ascPath, adaptive):
        super().__init__(mpm, TIME, saveTime, CFL, vtkPath, ascPath, adaptive)


    def Solver(self):
        start = time.time()
        while self.t <= self.TIME:
            if self.t == 0:
                self.Output(start)
                self.printNum += 1
                
            self.mpm.Engine.GridReset()
            self.mpm.Engine.ParticleToGrid_Momentum()
            self.mpm.Engine.GridVelocity()
            self.mpm.Engine.UpdateStressStrain()
            self.mpm.Engine.ParticleToGrid_Force()
            self.mpm.Engine.GridMomentum()
            self.mpm.Engine.GridToParticle()

            if self.saveTime - self.t % self.saveTime < self.mpm.Dt[None]:
                self.Output(start)
                self.printNum += 1

            self.mpm.Engine.Reset()

            self.t += self.mpm.Dt[None]
            self.step += 1

            self.mpm.AddParticlesInRun(self.t)
            if self.adaptive: self.mpm.AdaptiveTimeScheme(self.CFL)

        print('Physical time = ', time.time() - start)

@ti.data_oriented
class SolverUSL(TimeIntegrationMPM):
    def __init__(self, mpm, TIME, saveTime, CFL, vtkPath, ascPath, adaptive):
        super().__init__(mpm, TIME, saveTime, CFL, vtkPath, ascPath, adaptive)


    def Solver(self):
        start = time.time()
        while self.t <= self.TIME:
            if self.t == 0:
                self.Output(start)
                self.printNum += 1

            self.mpm.Engine.GridReset()
            self.mpm.Engine.ParticleToGrid_Momentum()
            self.mpm.Engine.ParticleToGrid_Force()
            self.mpm.Engine.GridMomentum()
            self.mpm.Engine.GridToParticle()
            self.mpm.Engine.UpdateStressStrain()

            if self.saveTime - self.t % self.saveTime < self.mpm.Dt[None]:
                self.Output(start)
                self.printNum += 1

            self.mpm.Engine.Reset()

            self.t += self.mpm.Dt[None]
            self.step += 1

            self.mpm.AddParticlesInRun(self.t)
            if self.adaptive: self.mpm.AdaptiveTimeScheme(self.CFL)

        print('Physical time = ', time.time() - start)


@ti.data_oriented
class SolverMUSL(TimeIntegrationMPM):
    def __init__(self, mpm, TIME, saveTime, CFL, vtkPath, ascPath, adaptive):
        super().__init__(mpm, TIME, saveTime, CFL, vtkPath, ascPath, adaptive)


    def Solver(self):
        start = time.time()
        while self.t <= self.TIME:
            if self.t == 0:
                self.Output(start)
                self.printNum += 1

            self.mpm.Engine.GridReset()
            self.mpm.Engine.ParticleToGrid_Momentum()
            self.mpm.Engine.ParticleToGrid_Force()
            self.mpm.Engine.GridMomentum()
            self.mpm.Engine.GridToParticle()
            self.mpm.Engine.NodalMomentumMUSL()
            self.mpm.Engine.UpdateStressStrain()

            if self.saveTime - self.t % self.saveTime < self.mpm.Dt[None]:
                self.Output(start)
                self.printNum += 1

            self.mpm.Engine.Reset()

            self.t += self.mpm.Dt[None]
            self.step += 1

            self.mpm.AddParticlesInRun(self.t)
            if self.adaptive: self.mpm.AdaptiveTimeScheme(self.CFL)

        print('Physical time = ', time.time() - start)


#  ------------------------------------------ GIMP --------------------------------------------- #
@ti.data_oriented
class SolverGIMP(TimeIntegrationMPM):
    def __init__(self, mpm, TIME, saveTime, CFL, vtkPath, ascPath, adaptive):
        super().__init__(mpm, TIME, saveTime, CFL, vtkPath, ascPath, adaptive)


    def Solver(self):
        start = time.time()
        while self.t <= self.TIME:
            if self.t == 0:
                self.Output(start)
                self.printNum += 1
            
            self.mpm.Engine.GridReset()
            self.mpm.Engine.ParticleToGrid_Momentum()
            self.mpm.Engine.ParticleToGrid_Force()
            self.mpm.Engine.GridMomentum()
            self.mpm.Engine.GridToParticle()
            self.mpm.Engine.UpdateStressStrain()

            if self.saveTime - self.t % self.saveTime < self.mpm.Dt[None]:
                self.Output(start)
                self.printNum += 1

            self.mpm.Engine.Reset()

            self.t += self.mpm.Dt[None]
            self.step += 1

            self.mpm.AddParticlesInRun(self.t)
            if self.adaptive: self.mpm.AdaptiveTimeScheme(self.CFL)

        print('Physical time = ', time.time() - start)


#  ----------------------------------------- MLS-MPM -------------------------------------------- #
@ti.data_oriented
class SolverMLSMPM(TimeIntegrationMPM):
    def __init__(self, mpm, TIME, saveTime, CFL, vtkPath, ascPath, adaptive):
        super().__init__(mpm, TIME, saveTime, CFL, vtkPath, ascPath, adaptive)


    def Solver(self):
        start = time.time()
        while self.t <= self.TIME:
            if self.t == 0:
                self.Output(start)
                self.printNum += 1
            
            self.mpm.Engine.ParticleToGridAPIC()
            self.mpm.Engine.GridOperationAPIC()
            self.mpm.Engine.GridToParticleAPIC()

            if self.saveTime - self.t % self.saveTime < self.mpm.Dt[None]:
                self.Output(start)
                self.printNum += 1

            self.mpm.Engine.Reset()

            self.t += self.mpm.Dt[None]
            self.tep += 1

            self.mpm.AddParticlesInRun(self.t)
            if self.adaptive: self.mpm.AdaptiveTimeScheme(self.CFL)

        print('Physical time = ', time.time() - start)


@ti.data_oriented
class SolverMPDEM:
    def __init__(self, dempm):
        self.dempm = dempm

    def Flow(self):
        pass


@ti.data_oriented
class FlowUSF(SolverMPDEM):
    def __init__(self, dempm):
        super().__init__(dempm)

    def Flow(self):
        self.dempm.MPMEngine.GridReset()
        self.dempm.MPMEngine.ParticleToGrid_Momentum()
        self.dempm.MPMEngine.GridVelocity()
        self.dempm.MPMEngine.UpdateStressStrain()
        self.dempm.MPMEngine.ParticleToGrid_Force()
        self.dempm.MPMEngine.GridMomentum()
        self.dempm.MPMEngine.GridToParticle()


@ti.data_oriented
class FlowUSL(SolverMPDEM):
    def __init__(self, dempm):
        super().__init__(dempm)


    def Flow(self):
        self.dempm.MPMEngine.GridReset()
        self.dempm.MPMEngine.ParticleToGrid_Momentum()
        self.dempm.MPMEngine.ParticleToGrid_Force()
        self.dempm.MPMEngine.GridMomentum()
        self.dempm.MPMEngine.GridToParticle()
        self.dempm.MPMEngine.UpdateStressStrain()


@ti.data_oriented
class FlowMUSL(SolverMPDEM):
    def __init__(self, dempm):
        super().__init__(dempm)


    def Flow(self):
        self.dempm.MPMEngine.GridReset()
        self.dempm.MPMEngine.ParticleToGrid_Momentum()
        self.dempm.MPMEngine.ParticleToGrid_Force()
        self.dempm.MPMEngine.GridMomentum()
        self.dempm.MPMEngine.GridToParticle()
        self.dempm.MPMEngine.NodalMomentumMUSL()
        self.dempm.MPMEngine.UpdateStressStrain()


@ti.data_oriented
class FlowGIMP(SolverMPDEM):
    def __init__(self, dempm):
        super().__init__(dempm)


    def Flow(self):
        self.dempm.MPMEngine.GridReset()
        self.dempm.MPMEngine.ParticleToGrid_Momentum()
        self.dempm.MPMEngine.ParticleToGrid_Force()
        self.dempm.MPMEngine.GridMomentum()
        self.dempm.MPMEngine.GridToParticle()
        self.dempm.MPMEngine.UpdateStressStrain()


@ti.data_oriented
class FlowMLSMPM(SolverMPDEM):
    def __init__(self, dempm):
        super().__init__(dempm)


    def Flow(self):
        self.dempm.MPMEngine.ParticleToGridAPIC()
        self.dempm.MPMEngine.GridOperationAPIC()
        self.dempm.MPMEngine.GridToParticleAPIC()

