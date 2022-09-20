from DEMPMLib3D_v1.Function import *
import DEMPMLib3D_v1.PenaltyMethod as PenaltyMethod
import DEMPMLib3D_v1.BarrierMethod as BarrierMethod
import DEMPMLib3D_v1.DEMPMContactPairs as DEMPMContactPair
import DEMPMLib3D_v1.NeighborSearchMultiLinkedCell as NeighborSearchMultiLinkedCell
import DEMLib3D_v1.DEMEngine as DEMEngine
import MPMLib3D_v1.MPMEngine as MPMEngine
import DEMLib3D_v1.TimeIntegrationDEM as TimeIntegrationDEM
import MPMLib3D_v1.TimeIntegrationMPM as TimeIntegrationMPM
import DEMPMLib3D_v1.TimeIntegrationDEMPM as TimeIntegrationDEMPM


@ti.data_oriented
class DEMPM:
    def __init__(self, CMtype, dem, mpm):
        self.CMtype = CMtype                                                      # /DE-MPM contact types/ 0 for linear contact model; 1 for the barrier method
        self.dem = dem
        self.mpm = mpm
        
        self.MainLoop = None

        def CheckDEMPM():
            if self.dem.SearchAlgorithm != 1:
                print("ERROR: DEM Neighbour Search Alogrithm must choose Linked Cell Method")
                assert 0

            if not all(self.dem.Domain == self.mpm.Domain):
                print("ERROR: DEM domain must be as same as MPM domain")
                assert 0

            if self.dem.Dt[None] == self.mpm.Dt[None]:
                self.Dt = self.mpm.Dt[None]
            else:
                print("ERROR: DEM timestep must be as same as MPM timestep")
                assert 0

        CheckDEMPM()
        self.MPMContactInfo = ti.Struct.field({                                   # List of contact parameters
            "Kn": float,                                                          # Contact normal stiffness
            "Ks": float,                                                          # Contact tangential stiffness
            "Kapa": float,                                                        # Barrier Scalar Parameters
            "limitDisp": float,                                                   # Maximum Microslip Displacement
            "Mu": float,                                                          # Friction coefficient
            "NormalViscousDamping": float,                                        # Viscous Damping
            "TangViscousDamping": float,                                          # Viscous Damping
        }, shape=(self.mpm.matNum,))

        self.DEMContactInfo = ti.Struct.field({                                   # List of contact parameters
            "Kapa": float,                                                        # Barrier Scalar Parameters
            "limitDisp": float,                                                   # Maximum Microslip Displacement
        }, shape=(self.dem.matNum,))

    def AddContactParams(self):
        print('------------------------ DEMPM ContactModels Initialization ------------------------')
        if ti.static(self.CMtype == 0):
            self.DEMPMcontModel = PenaltyMethod.PenaltyMethod(self.dem.contModel, self.mpm.matList, self.Dt)
            for nmpm in range(self.MPMContactInfo.shape[0]):
                self.mpm.matList.DEMPMPenaltyMethod()
                self.DEMPMcontModel.InitMPM(nmpm, self.MPMContactInfo)
        elif ti.static(self.CMtype == 1):
            self.DEMPMcontModel = BarrierMethod.BarrierMethod(self.dem.contModel, self.mpm.matList, self.Dt)
            for nmpm in range(self.MPMContactInfo.shape[0]):
                self.mpm.matList.DEMPMPenaltyMethod()
                self.DEMPMcontModel.InitMPM(nmpm, self.MPMContactInfo)
            for ndem in range(self.DEMContactInfo.shape[0]):
                self.dem.contModel.DEMPMBarrierMethod()
                self.DEMPMcontModel.InitDEM(ndem, self.DEMContactInfo)

    def AddContactPair(self, max_dempm_contact_num):
        print('------------------------ DEMPM Contact Pair Initialization ------------------------')
        self.DEMPMcontPair = DEMPMContactPair.DEMPMContactPair(max_dempm_contact_num, self.dem.partList, self.mpm.partList, self.DEMPMcontModel)
        
    def AddNeighborList(self, multiplier, scaler, max_potential_dempm_particle_pairs):
        print('------------------------ DEMPM Neighbor Initialization ------------------------')
        self.multiplier = multiplier
        self.scaler = scaler
        self.MPMGridSize = self.multiplier * (self.mpm.Dx[0] + self.mpm.Dx[1] + self.mpm.Dx[2]) / 3. / self.mpm.Npic 
        self.SmoothRange = self.scaler * (self.mpm.Dx[0] + self.mpm.Dx[1] + self.mpm.Dx[2]) / 3. / self.mpm.Npic 
        self.DEMPMneighborList = NeighborSearchMultiLinkedCell.NeighborSearchMultiLinkedCell(self.mpm.Domain, self.MPMGridSize, 
                                                                                             self.dem.DEMGridSize, self.SmoothRange, max_potential_dempm_particle_pairs, 
                                                                                             self.mpm.partList, self.dem.partList, self.dem.neighborList, self.DEMPMcontPair)

    # ===================================== Solve ==================================================== #
    def Solver(self, TIME, saveTime, CFL, vtkPath, ascPath, adaptive):
        print('------------------------ DEMPM Solver ------------------------')
        if self.MainLoop:
            self.MainLoop.UpdateSolver(TIME, saveTime, CFL, vtkPath, ascPath, adaptive)
            self.MainLoop.Solver()
        else:
            if ti.static(self.mpm.Algorithm == 0):
                print("MPM Integration Scheme: USF\n")
                self.MPMEngine = MPMEngine.USF(self.mpm.Gravity, self.mpm.threshold, self.mpm.Damp, self.mpm.alphaPIC, self.mpm.Dt[None], self.mpm.partList, self.mpm.gridList, self.mpm.matList)
                self.MPMLoop = TimeIntegrationMPM.FlowUSF(self)
            elif ti.static(self.mpm.Algorithm == 1):
                print("MPM Integration Scheme: USL\n")
                self.MPMEngine = MPMEngine.USL(self.mpm.Gravity, self.mpm.threshold, self.mpm.Damp, self.mpm.alphaPIC, self.mpm.Dt[None], self.mpm.partList, self.mpm.gridList, self.mpm.matList)
                self.MPMLoop = TimeIntegrationMPM.FlowUSL(self)
            elif ti.static(self.mpm.Algorithm == 2):
                print("MPM Integration Scheme: MUSL\n")
                self.MPMEngine = MPMEngine.MUSL(self.mpm.Gravity, self.mpm.threshold, self.mpm.Damp, self.mpm.alphaPIC, self.mpm.Dt[None], self.mpm.partList, self.mpm.gridList, self.mpm.matList)
                self.MPMLoop = TimeIntegrationMPM.FlowMUSL(self)
            elif ti.static(self.mpm.Algorithm == 3):
                print("MPM Integration Scheme: undeformed GIMP\n")
                if ti.static(self.mpm.ShapeFunction != 1): 
                    print("!!ERROR: Shape Function must equal to 1")
                    assert 0
                self.MPMEngine = MPMEngine.GIMP(self.mpm.Gravity, self.mpm.threshold, self.mpm.Damp, self.mpm.alphaPIC, self.mpm.Dt[None], self.mpm.partList, self.mpm.gridList, self.mpm.matList)
                self.MPMLoop = TimeIntegrationMPM.FlowGIMP(self)
            elif ti.static(self.mpm.Algorithm == 4):
                print("MPM Integration Scheme: MLSMPM\n")
                self.MPMEngine = MPMEngine.MLSMPM(self.mpm.Gravity, self.mpm.threshold, self.mpm.Damp, self.mpm.alphaPIC, self.mpm.Dt[None], self.mpm.partList, self.mpm.gridList, self.mpm.matList)
                self.MPMLoop = TimeIntegrationMPM.FlowMLSMPM(self)

            if ti.static(self.dem.Algorithm == 0):
                print("DEM Integration Scheme: Euler\n")
                self.DEMEngine = DEMEngine.Euler(self.dem.Gravity, self.dem.Dt[None], self.dem.partList, self.dem.contModel)
                self.DEMLoop = TimeIntegrationDEM.FlowEuler(self, self.dem)
            elif ti.static(self.dem.Algorithm == 1):
                print("DEM Integration Scheme: Verlet\n")
                self.DEMEngine = DEMEngine.Verlet(self.dem.Gravity, self.dem.Dt[None], self.dem.partList, self.dem.contModel)
                self.DEMLoop = TimeIntegrationDEM.FlowVerlet(self, self.dem)

            self.MainLoop = TimeIntegrationDEMPM.TimeIntegrationDEMPM(self, self.mpm, self.dem, TIME, saveTime, CFL, vtkPath, ascPath, adaptive)
            self.MainLoop.TurnOnSolver(t=0., step=0, printNum=0)
            self.MainLoop.Solver()



