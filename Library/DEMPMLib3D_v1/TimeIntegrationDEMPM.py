import taichi as ti
import MPMLib3D_v1.Graphic as GraphicMPM
import MPMLib3D_v1.Spying as SpyMPM
import DEMLib3D_v1.Graphic as GraphicDEM
import DEMLib3D_v1.Spying as SpyDEM
import time

@ti.data_oriented
class TimeIntegrationDEMPM:
    def __init__(self, dempm, mpm, dem, TIME, saveTime, CFL, vtkPath, ascPath, adaptive):
        self.dempm = dempm
        self.mpm = mpm
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
        print('------------------ Time Step = ', self.step, '------------------')
        print('Simulation time = ', self.t)
        print('Time step = ', self.dempm.Dt)
        print('Physical time = ', time.time() - start)
        GraphicMPM.WriteFileVTK_MPM(self.mpm.partList, self.mpm.gridList, self.printNum, self.vtkPath)
        GraphicDEM.WriteFileVTK_DEM(self.dem.partList, self.printNum, self.vtkPath)
        SpyMPM.MonitorMPM(self.t, self.mpm.partList, self.printNum, self.ascPath)
        SpyDEM.MonitorDEM(self.t, self.dem.partList, self.printNum, self.ascPath)
        print('------------------------- Running --------------------------')

    def Solver(self):
        start = time.time()
        while self.t <= self.TIME:
            if self.t == 0:
                self.Output(start)
                self.printNum += 1
            
            self.dempm.DEMPMneighborList.SumMPMParticles()
            self.dempm.DEMPMneighborList.BoardMPMNeighborList()
            self.dempm.DEMPMneighborList.BoardSearchP2M()
            self.dempm.DEMPMneighborList.FineSearchP2M()
            self.dempm.DEMPMcontPair.Reset()
            self.dempm.MPMLoop.Flow()
            self.dempm.DEMLoop.Flow(self.t)

            if self.t % self.saveTime < self.dempm.Dt:
                self.Output(start)
                self.printNum += 1

            self.t += self.dempm.Dt
            self.step += 1

