import taichi as ti
import MPMLib3D_v1.MPMParticles as Particle
import MPMLib3D_v1.MPMMaterials as Material
import MPMLib3D_v1.MPMGrids as Grid
import MPMLib3D_v1.MPMCells as Cell
import MPMLib3D_v1.MPMEngine as Engine
import MPMLib3D_v1.TimeIntegrationMPM as TimeIntegrationMPM


@ti.data_oriented
class MPM:
    def __init__(self, domain, dx, npic, boundary, damping,                                            # MPM domain
                 algorithm, shape_func, gravity, timeStep, Stablization,                               # MPM settings
                 max_particle_num, bodyNum, matNum, alphaPIC, Dynamic_Allocate, Contact_Detection):    # Input initialization
        self.Domain = domain
        self.Dx = dx
        self.inv_Dx = 1. / dx
        self.Npic = npic
        self.Boundary = boundary
        self.Damp = damping
        self.Algorithm = algorithm
        self.ShapeFunction = shape_func
        self.Gravity = gravity
        self.alphaPIC = alphaPIC
        self.Dynamic_Allocate = Dynamic_Allocate
        self.Contact_Detection = Contact_Detection
        self.Dt = ti.field(float, shape=())
        self.Dt[None] = timeStep
        
        self.bodyNum = int(bodyNum)
        self.matNum = int(matNum)
        self.max_particle_num = int(max_particle_num)
        self.Stablization = Stablization
        self.threshold = 1e-8

        self.MainLoop = None
            
        
        self.BasisType = 0
        if ti.static(self.ShapeFunction) == 0: 
            print("Shape Function: Linear")
            self.BasisType = 8
        elif ti.static(self.ShapeFunction) == 1: 
            print("Shape Function: GIMP")
            self.BasisType = 27
        elif ti.static(self.ShapeFunction) == 2: 
            print("Shape Function: Quadratic B-spline")
            self.BasisType = 27
        elif ti.static(self.ShapeFunction) == 3: 
            print("Shape Function: Cubic B-spline")
            self.BasisType = 64
        elif ti.static(self.ShapeFunction) == 4: 
            print("Shape Function: NURBS")

        if ti.static(self.Stablization) == 0: 
            print("Stablization Technique: NULL")
        elif ti.static(self.Stablization) == 1: 
            print("Stablization Technique: Reduce Integration")
        elif ti.static(self.Stablization) == 2: 
            print("Stablization Technique: B-Bar Method")
        elif ti.static(self.Stablization) == 3: 
            print("Stablization Technique: F-Bar Method")
        elif ti.static(self.Stablization) == 4: 
            print("Stablization Technique: Cell Average")
        

        self.BodyInfo = ti.Struct.field({                                         # List of body parameters
            "ID": int,                                                            # Body ID
            "Mat": int,                                                           # Material name
            "Type": int,                                                          # /Body shape/ 0 for rectangle; 1 for triangle; 2 for sphere
            "MacroRho": float,                                                    # Macro density
            "pos0": ti.types.vector(3, float),                                    # Initial position or center of sphere of Body
            "len": ti.types.vector(3, float),                                     # Size of Body
            "rad": float,                                                         # Radius
            "v0": ti.types.vector(3, float),                                      # Initial velocity
            "fixedV": ti.types.vector(3, int),                                    # Fixed velocity
            "DT": ti.types.vector(3, float)                                       # DT
        }, shape=(self.bodyNum,))
        
        self.MatInfo = ti.Struct.field({                                          # List of Material parameters
            "Type": int,                                                          # /Constitutive model/ 0 for Hyper-elastic; 1 for Mohr-Coulomb; 2 for Drucker-Parger; 3 for Newtonian
            "Modulus": float,                                                     # Young's Modulus for soil; Bulk modulus for fluid
            "mu": float,                                                          # Possion ratio
            "h": float,                                                           # Hardening
            "Cohesion": float,                                                    # Cohesion coefficient
            "InternalFric": float,                                                # Angle of internal friction
            "Dilation": float,                                                    # Angle of dilatation
            "lodeT": float,                                                       # Transition angle
            "dpType": int,                                                        # Drucker-Prager types
            "Viscosity": float,                                                   # Viscosity
            "SoundSpeed": float,                                                  # Sound Speed
            "miu_s": float,                                                       # Rheology parameters
            "miu_2": float,                                                       # Rheology parameters   
            "I0": float,                                                          # Rheology parameters
            "ParticleRho": float,                                                 # Particle density = (Initial volume fraction * macro density) ! Not Macro density
            "ParticleDiameter": float                                             # Particle diameter
        }, shape=(self.matNum,))

        
    def AddGrid(self):
        print('------------------------ Grid Initialization ------------------------')
        self.gridList = Grid.GridList(self.Domain, self.threshold, self.ShapeFunction, self.Dx, self.Dt[None], self.Contact_Detection)                    # List of self grids
        print('Grid Number = ', self.gridList.gnum)
        self.gridList.GridInit()
        return self.gridList

    def AddMaterial(self):
        print('------------------------ Material Initialization ------------------------')
        self.matList = Material.MaterialList(self.matNum)                                                                     # List of self materials
        for nm in range(self.MatInfo.shape[0]):
            print('Particle Material ID = ', nm)
            if self.MatInfo[nm].Type == 0:
                self.matList.matType[nm] = self.MatInfo[nm].Type                                       # Hyper-elastic Model
                self.matList.ElasticModel(nm, self.MatInfo)
            elif self.MatInfo[nm].Type == 1:                                                      # Mohr-Coulomb Model
                self.matList.matType[nm] = self.MatInfo[nm].Type
                self.matList.MohrCoulombModel(nm, self.MatInfo)
            elif self.MatInfo[nm].Type == 2:                                                      # Drucker-Prager Model
                self.matList.matType[nm] = self.MatInfo[nm].Type
                self.matList.DruckerPragerModel(nm, self.MatInfo)
            elif self.MatInfo[nm].Type == 3:                                                      # Newtonian Model
                self.matList.matType[nm] = self.MatInfo[nm].Type
                self.matList.NewtonianModel(nm, self.MatInfo)
            elif self.MatInfo[nm].Type == 4:
                self.matList.matType[nm] = self.MatInfo[nm].Type
                self.matList.ViscoplasticModel(nm, self.MatInfo)
            elif self.MatInfo[nm].Type == 5:
                self.matList.matType[nm] = self.MatInfo[nm].Type
                self.matList.ElasticViscoplasticModel(nm, self.MatInfo)
        return self.matList

    def AddCell(self):
        print('------------------------ Cell Initialization ------------------------')
        self.cellList = Cell.CellList(self.Domain, self.Dx)                                                                  # List of self cells
        print('Cell Number = ', self.gridList.gnum, '\n')
        self.cellList.CellInit()
        return self.cellList

    def AddParticle(self):
        print('------------------------ Body Initialization ------------------------')
        self.partList = Particle.ParticleList(self.Dx, self.Domain, self.threshold, self.max_particle_num, self.ShapeFunction, self.Stablization, self.Dt[None], 
                                              self.gridList, self.matList, self.cellList)                     # List of self particles
        self.partList.StoreShapeFuncs(self.BasisType)
        for nb in range(self.BodyInfo.shape[0]):
            if self.BodyInfo[nb].Type == 0:
                self.partList.AddRectangle(nb, self.BodyInfo, self.Npic)
            elif self.BodyInfo[nb].Type == 1:
                self.partList.AddTriangle(nb, self.BodyInfo, self.Npic)
            elif self.BodyInfo[nb].Type == 2:
                self.partList.AddCircle(nb, self.BodyInfo, self.Npic)
        if self.Algorithm == 3:
            self.partList.StoreParticleLen()
            self.partList.GIMPInit(self.Npic)
        self.partList.MPMCell()    
        return self.partList

    def AddParticlesInRun(self, t):
        for nb in range(self.BodyInfo.shape[0]):
            if t % self.BodyInfo[nb].DT[1] < self.Dt[None] and self.BodyInfo[nb].DT[0] <= t <= self.BodyInfo[nb].DT[2]:
                print('------------------------ Add Body  ------------------------')
                if self.BodyInfo[nb].Type == 0:
                    self.partList.AddRectangle(nb, self.BodyInfo, self.Npic, self.Dx)
                elif self.BodyInfo[nb].Type == 1:
                    self.partList.AddTriangle(nb, self.BodyInfo, self.Npic)
                elif self.BodyInfo[nb].Type == 2:
                    self.partList.AddCircle(nb, self.BodyInfo, self.Npic)
                if self.Algorithm == 3:
                    self.partList.GIMPInit(self.Dx, self.Npic)
    
    @ti.kernel
    def CriticalTimeStep(self, CFL: float):
        matList = self.matList
        SoundSpeed = 0.
        for nb in range(self.BodyInfo.shape[0]):
            matID = self.BodyInfo[nb].Mat
            young = self.MatInfo[matID].Modulus
            mu = self.MatInfo[matID].mu
            c_2 = young * (1 - mu) / (1 + mu) / (1 - 2 * mu) / self.BodyInfo[nb].MacroRho
            ti.atomic_max(SoundSpeed, ti.sqrt(c_2))
        dt_c = ti.min(self.Dx[0], self.Dx[1], self.Dx[2]) / SoundSpeed
        self.Dt[None] = ti.min(CFL * dt_c, self.Dt[None])



    # ============================================= Time Solver ======================================================= #
    def Solver(self, TIME, saveTime, CFL, vtkPath, ascPath, adaptive):
        print('------------------------ MPM Solver ------------------------')
        if self.MainLoop:
            self.MainLoop.UpdateSolver(TIME, saveTime, CFL, vtkPath, ascPath, adaptive)
            self.MainLoop.Solver()
        else:
            if ti.static(self.Algorithm == 0):
                print("Integration Scheme: USF\n")
                self.Engine = Engine.USF(self.Gravity, self.threshold, self.Damp, self.alphaPIC, self.Dt[None], self.partList, self.gridList, self.matList)
                self.MainLoop = TimeIntegrationMPM.SolverUSF(self, TIME, saveTime, CFL, vtkPath, ascPath, adaptive)
                self.MainLoop.TurnOnSolver(t=0., step=0, printNum=0)
                self.MainLoop.Solver()
            elif ti.static(self.Algorithm == 1):
                print("Integration Scheme: USL\n")
                self.Engine = Engine.USL(self.Gravity, self.threshold, self.Damp, self.alphaPIC, self.Dt[None], self.partList, self.gridList, self.matList)
                self.MainLoop = TimeIntegrationMPM.SolverUSL(self, TIME, saveTime, CFL, vtkPath, ascPath, adaptive)
                self.MainLoop.TurnOnSolver(t=0., step=0, printNum=0)
                self.MainLoop.Solver()
            elif ti.static(self.Algorithm == 2):
                print("Integration Scheme: MUSL\n")
                self.Engine = Engine.MUSL(self.Gravity, self.threshold, self.Damp, self.alphaPIC, self.Dt[None], self.partList, self.gridList, self.matList)
                self.MainLoop = TimeIntegrationMPM.SolverMUSL(self, TIME, saveTime, CFL, vtkPath, ascPath, adaptive)
                self.MainLoop.TurnOnSolver(t=0., step=0, printNum=0)
                self.MainLoop.Solver()
            elif ti.static(self.Algorithm == 3):
                print("Integration Scheme: undeformed GIMP\n")
                if ti.static(self.ShapeFunction != 1): 
                    print("!!ERROR: Shape Function must equal to 1")
                    assert 0
                self.Engine = Engine.GIMP(self.Gravity, self.threshold, self.Damp, self.alphaPIC, self.Dt[None], self.partList, self.gridList, self.matList)
                self.MainLoop = TimeIntegrationMPM.SolverGIMP(self, TIME, saveTime, CFL, vtkPath, ascPath, adaptive)
                self.MainLoop.TurnOnSolver(t=0., step=0, printNum=0)
                self.MainLoop.Solver()
            elif ti.static(self.Algorithm == 4):
                print("Integration Scheme: MLSMPM\n")
                self.Engine = Engine.MLSMPM(self.Gravity, self.threshold, self.Damp, self.alphaPIC, self.Dt[None], self.partList, self.gridList, self.matList)
                self.MainLoop = TimeIntegrationMPM.SolverMLSMPM(self, TIME, saveTime, CFL, vtkPath, ascPath, adaptive)
                self.MainLoop.TurnOnSolver(t=0., step=0, printNum=0)
                self.MainLoop.Solver()
