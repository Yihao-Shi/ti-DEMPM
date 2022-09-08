import taichi as ti
import MPMLib2D_v1.MPMParticles as particle
import MPMLib2D_v1.MPMMaterials as material
import MPMLib2D_v1.MPMGrids as grid
import MPMLib2D_v1.MPMCells as cell
import MPMLib2D_v1.TimeIntegrationMPM as Solver


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
        self.Stablization = Stablization                                                                 # /Anti-locking Technique/ 0 for NULL; 1 for b-bar; 2 for f-bar; 3 for cell-average
        self.threshold = 1e-8

        if ti.static(self.Algorithm) == 0: 
            print("Integration Scheme: USF")
        elif ti.static(self.Algorithm) == 1: 
            print("Integration Scheme: USL")
        elif ti.static(self.Algorithm) == 2: 
            print("Integration Scheme: MUSL")
        elif ti.static(self.Algorithm) == 3: 
            print("Integration Scheme: undeformed GIMP")
            if ti.static(self.ShapeFunction != 1): 
                print("!!ERROR: Shape Function must equal to 1")
                assert 0
        elif ti.static(self.Algorithm) == 4: 
            print("Integration Scheme: MLSMPM")
        
        self.BasisType = 0
        if ti.static(self.ShapeFunction) == 0: 
            print("Shape Function: Linear")
            self.BasisType = 4
        elif ti.static(self.ShapeFunction) == 1: 
            print("Shape Function: GIMP")
            self.BasisType = 9
        elif ti.static(self.ShapeFunction) == 2: 
            print("Shape Function: Quadratic B-spline")
            self.BasisType = 9
        elif ti.static(self.ShapeFunction) == 3: 
            print("Shape Function: Cubic B-spline")
            self.BasisType = 16
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
            "pos0": ti.types.vector(2, float),                                    # Initial position or center of sphere of Body
            "len": ti.types.vector(2, float),                                     # Size of Body
            "rad": float,                                                         # Radius
            "v0": ti.types.vector(2, float),                                      # Initial velocity
            "fixedV": ti.types.vector(2, int),                                    # Fixed velocity
            "DT": ti.types.vector(3, float)                                       # DT
        }, shape=(self.bodyNum,))
        
        self.MatInfo = ti.Struct.field({                                          # List of material parameters
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
        print('---------------------------------------- Grid Initialization ------------------------------------------')
        self.lg = grid.GridList(self.Domain, self.Dx, self.Contact_Detection)                    # List of mpm grids
        print('Grid Number = ', self.lg.gnum)
        self.lg.GridInit(self.Dx)
        return self.lg

    def AddMaterial(self):
        print('------------------------------------- Material Initialization -----------------------------------------')
        self.lm = material.MaterialList(self.matNum)                                                                     # List of mpm materials
        for nm in range(self.MatInfo.shape[0]):
            print('Particle material ID = ', nm)
            if self.MatInfo[nm].Type == 0:
                self.lm.matType[nm] = self.MatInfo[nm].Type                                       # Hyper-elastic Model
                self.lm.ElasticModel(nm, self.MatInfo)
            elif self.MatInfo[nm].Type == 1:                                                      # Mohr-Coulomb Model
                self.lm.matType[nm] = self.MatInfo[nm].Type
                self.lm.MohrCoulombModel(nm, self.MatInfo)
            elif self.MatInfo[nm].Type == 2:                                                      # Drucker-Prager Model
                self.lm.matType[nm] = self.MatInfo[nm].Type
                self.lm.DruckerPragerModel(nm, self.MatInfo)
            elif self.MatInfo[nm].Type == 3:                                                      # Newtonian Model
                self.lm.matType[nm] = self.MatInfo[nm].Type
                self.lm.NewtonianModel(nm, self.MatInfo)
            elif self.MatInfo[nm].Type == 4:
                self.lm.matType[nm] = self.MatInfo[nm].Type
                self.lm.ViscoplasticModel(nm, self.MatInfo)
            elif self.MatInfo[nm].Type == 5:
                self.lm.matType[nm] = self.MatInfo[nm].Type
                self.lm.ElasticViscoplasticModel(nm, self.MatInfo)
        return self.lm

    

    def AddParticle(self):
        print('---------------------------------------- Body Initialization ------------------------------------------')
        self.lp = particle.ParticleList(self.Dx, self.Domain, self.max_particle_num)                                                            # List of mpm particles
        self.lp.StoreShapeFuncs(self.max_particle_num, self.BasisType)
        for nb in range(self.BodyInfo.shape[0]):
            if self.BodyInfo[nb].Type == 0:
                self.lp.AddRectangle(nb, self.BodyInfo, self.lm, self.Npic)
            elif self.BodyInfo[nb].Type == 1:
                self.lp.AddTriangle(nb, self.BodyInfo, self.lm, self.Npic)
            elif self.BodyInfo[nb].Type == 2:
                self.lp.AddCircle(nb, self.BodyInfo, self.lm, self.Npic)
        if self.Algorithm == 3:
            self.lp.StoreParticleLen(self.max_particle_num)
            self.lp.GIMPInit(self.Npic)
        return self.lp

    def AddCell(self):
	    print('---------------------------------------- Cell Initialization ------------------------------------------')
        self.cell = cell.CellList(self.Domain, self.Dx)                                                                  # List of mpm cells
        print('Cell Number = ', self.lg.gnum)
        self.cell.CellInit()
        self.lp.MPMCell(self.cell)
        return self.cell
        
    def AddParticlesInRun(self, nb):
        print('---------------------------------------- Adding Body  -------------------------------------------------')
        if self.BodyInfo[nb].Type == 0:
            self.lp.AddRectangle(nb, self.BodyInfo, self.Npic)
        elif self.BodyInfo[nb].Type == 1:
            self.lp.AddTriangle(nb, self.BodyInfo, self.Npic)
        elif self.BodyInfo[nb].Type == 2:
            self.lp.AddCircle(nb, self.BodyInfo, self.Npic)
        if self.Algorithm == 3:
            self.lp.GIMPInit(self.Npic)

    @ti.kernel
    def CriticalTimeStep(self, CFL: float):
        matList = self.lm
        SoundSpeed = 0.
        for nb in range(self.BodyInfo.shape[0]):
            matID = self.BodyInfo[nb].Mat
            young = self.MatInfo[matID].Modulus
            mu = self.MatInfo[matID].mu
            c_2 = young * (1 - mu) / (1 + mu) / (1 - 2 * mu) / self.BodyInfo[nb].MacroRho
            ti.atomic_max(SoundSpeed, ti.sqrt(c_2))
        dt_c = ti.min(self.Dx[0], self.Dx[1]) / SoundSpeed
        self.Dt[None] = ti.min(CFL * dt_c, self.Dt[None])

    @ti.kernel
    def AdaptiveTimeScheme(self, CFL: float):
        maxVel = ti.Vector([0., 0.])
        for np in range(self.lp.particleNum[None]):
            if self.lp.v[np][0] + self.lp.cs[np] > maxVel[0]:
                maxVel[0] = self.lp.v[np][0] + self.lp.cs[np]
            if self.lp.v[np][1] + self.lp.cs[np] > maxVel[1]:
                maxVel[1] = self.lp.v[np][1] + self.lp.cs[np]
        critT = CFL * self.Dx[0] / maxVel.norm()
        if self.UserDefinedDt >  critT:
            self.Dt[None] = critT
        else:
            self.Dt[None] = self.UserDefinedDt
