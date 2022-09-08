import taichi as ti
import time
import DEMLib3D_v1.Graphic as graphic
import DEMLib3D_v1.Spying as spy


@ti.data_oriented
class TimeIntegrationMPM:
    def __init__(self, dem, TIME, saveTime, CFL, vtkPath, ascPath, adaptive):
        self.dem = dem
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
        print('------------------------ Time Step = ', self.step, '------------------------')
        print('Simulation time = ', self.t)
        print('Time step = ', self.dem.Dt[None])
        print('Physical time = ', time.time() - start)
        graphic.WriteFileVTK_DEM(self.dem.partList, self.printNum, self.vtkPath)
        spy.MonitorDEM(self.t, self.dem.partList, self.printNum, self.ascPath)
        print('------------------------------- Running --------------------------------')


#  --------------------------------------- Euler Algorithm ----------------------------------------- #
@ti.data_oriented
class SolverEuler(TimeIntegrationMPM):
    def __init__(self, dem, TIME, saveTime, CFL, vtkPath, ascPath, adaptive):
        super().__init__(dem, TIME, saveTime, CFL, vtkPath, ascPath, adaptive)


    def Solver(self):
        start = time.time()
        while self.t <= self.TIME:
            if self.t == 0:
                self.Output(start)
                self.printNum += 1
            
            self.dem.Engine.UpdatePosition()
            self.dem.neighborList.InitWallList()
            self.dem.neighborList.SumParticles()
            self.dem.neighborList.BoardNeighborList()
            self.dem.neighborList.BoardSearchP2W()
            self.dem.neighborList.BoardSearchP2P()
            self.dem.neighborList.FineSearchP2W()
            self.dem.neighborList.FineSearchP2P()
            self.dem.contPair.Reset()
            self.dem.Engine.UpdateVelocity()
            self.dem.Engine.UpdateAngularVelocity(self.t)
            
            if self.t % self.saveTime < self.dem.Dt[None]:
                self.Output(start)
                self.printNum += 1

            self.t += self.dem.Dt[None]
            self.step += 1


#  --------------------------------------- Verlet Algorithm ----------------------------------------- #
@ti.data_oriented
class SolverVerlet(TimeIntegrationMPM):
    def __init__(self, dem, TIME, saveTime, CFL, vtkPath, ascPath, adaptive):
        super().__init__(dem, TIME, saveTime, CFL, vtkPath, ascPath, adaptive)


    def Solver(self):
        pass


@ti.data_oriented
class SolverDEMPM:
    def __init__(self, dem):
        self.dem = dem

    def Flow(self):
        pass


@ti.data_oriented
class FlowEuler(SolverDEMPM):
    def __init__(self, dem):
        super().__init__(dem)


    def Flow(self, t):
        self.dem.Engine.UpdatePosition()
        self.dem.neighborList.InitWallList()
        self.dem.neighborList.SumParticles()
        self.dem.neighborList.BoardNeighborList()
        self.dem.neighborList.BoardSearchP2W()
        self.dem.neighborList.BoardSearchP2P()
        self.dem.neighborList.FineSearchP2W()
        self.dem.neighborList.FineSearchP2P()
        self.dem.contPair.Reset()
        self.dem.Engine.UpdateVelocity()
        self.dem.Engine.UpdateAngularVelocity(t)


@ti.data_oriented
class FlowVerlet(SolverDEMPM):
    def __init__(self, dem):
        super().__init__(dem)


    def Flow(self, t):
        pass