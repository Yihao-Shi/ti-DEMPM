from __init__ import *
import MMPMLib2D_v1.MMPM_particles as particle
import MMPMLib2D_v1.MMPM_materials as material
import MMPMLib2D_v1.MMPM_grids as grid
import MMPMLib2D_v1.MMPM_cells as cell
import MMPMLib2D_v1.TimeIntegrationMMPM as Solver


@ti.data_oriented
class MMPM:
    def __init__(self, domain, dx, npic, boundary, damping,                                            # MPM domain
                 algorithm, shape_func, gravity, timeStep,                                 # MPM settings
                 max_particle_num_solid, max_particle_num_fluid, bodyNum_solid, bodyNum_fluid, matNum, 
                 alphaPIC, Dynamic_Allocate, Contact_Detection):    # Input initialization
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

        self.bodyNum_solid = int(bodyNum_solid)
        self.bodyNum_fluid = int(bodyNum_fluid)
        self.matNum = int(matNum)
        self.max_particle_num_solid = int(max_particle_num_solid)
        self.max_particle_num_fluid = int(max_particle_num_fluid)
        self.threshold = 1e-12

        if self.Algorithm == 0: 
            print("Integration Scheme: USF")
        elif self.Algorithm == 1: 
            print("Integration Scheme: USL")
        elif self.Algorithm == 2: 
            print("Integration Scheme: MUSL")
        elif self.Algorithm == 3: 
            print("Integration Scheme: undeformed GIMP")
            if self.ShapeFunction != 1: print("!!ERROR: Shape Function must equal to 1")
        elif self.Algorithm == 4: 
            print("Integration Scheme: MLSMPM")
        
        self.BasisType = 0
        if self.ShapeFunction == 0: 
            print("Shape Function: Linear")
            self.BasisType = 4
        elif self.ShapeFunction == 1: 
            print("Shape Function: GIMP")
            self.BasisType = 9
        elif self.ShapeFunction == 2: 
            print("Shape Function: Quadratic B-spline")
            self.BasisType = 9
        elif self.ShapeFunction == 3: 
            print("Shape Function: Cubic B-spline")
            self.BasisType = 16
        elif self.ShapeFunction == 4: 
            print("Shape Function: NURBS")

        self.BodyInfoSolid = ti.Struct.field({                                    # List of body parameters
            "ID": int,                                                            # Body ID
            "Mat": int,                                                           # Material name
            "Type": int,                                                          # /Body shape/ 0 for rectangle; 1 for triangle; 2 for sphere
            "MacroRho": float,                                                    # Macro density
            "pos0": ti.types.vector(2, float),                                    # Initial position or center of sphere of Body
            "len": ti.types.vector(2, float),                                     # Size of Body
            "rad": float,                                                         # Radius
            "v0": ti.types.vector(2, float),                                      # Initial velocity
            "fixedV": ti.types.vector(2, int),                                    # Fixed velocity
            "permeability": float,                                                # Permenability
            "porosity": float,                                                    # Void ratio
            "DT": ti.types.vector(3, float)                                       # DT
        }, shape=(self.bodyNum_solid,))

        self.BodyInfoFluid = ti.Struct.field({                                    # List of body parameters
            "ID": int,                                                            # Body ID
            "Mat": int,                                                           # Material name
            "Type": int,                                                          # /Body shape/ 0 for rectangle; 1 for triangle; 2 for sphere
            "MacroRho": float,                                                    # Macro density
            "pos0": ti.types.vector(2, float),                                    # Initial position or center of sphere of Body
            "len": ti.types.vector(2, float),                                     # Size of Body
            "rad": float,                                                         # Radius
            "v0": ti.types.vector(2, float),                                      # Initial velocity
            "fixedV": ti.types.vector(2, int),                                    # Fixed velocity
            "saturation": float,                                                  # Saturation
            "DT": ti.types.vector(3, float)                                       # DT
        }, shape=(self.bodyNum_fluid,))

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
    
    def AddCell(self):
        print('---------------------------------------- Cell Initialization ------------------------------------------')
        self.cell = cell.CellList(self.Domain, self.Dx)                                                                  # List of mpm cells
        print('Cell Number = ', self.cell.cnum)
        self.cell.CellInit()
        self.lps.MPMCell(self.cell)
        self.lpf.MPMCell(self.cell)
        return self.cell

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

    def AddParticles(self):
        print('---------------------------------------- Body Initialization ------------------------------------------')
        self.lps = particle.ParticleList(self.Dx, self.Domain, self.max_particle_num_solid)                                                            # List of mpm particles
        self.lps.StoreShapeFuncs(self.max_particle_num_solid, self.BasisType)
        for nb in range(self.BodyInfoSolid.shape[0]):
            if self.BodyInfoSolid[nb].Type == 0:
                self.lps.AddRectangle(nb, self.BodyInfoSolid, self.lm, self.Npic)
            elif self.BodyInfoSolid[nb].Type == 1:
                self.lps.AddTriangle(nb, self.BodyInfoSolid, self.lm, self.Npic)
            elif self.BodyInfoSolid[nb].Type == 2:
                self.lps.AddCircle(nb, self.BodyInfoSolid, self.lm, self.Npic)
        if self.Algorithm == 3:
            self.lps.StoreParticleLen(self.max_particle_num_solid)
            self.lps.GIMPInit(self.Npic)

        self.lpf = particle.ParticleList(self.Dx, self.Domain, self.max_particle_num_fluid)                                                            # List of mpm particles
        self.lpf.StoreShapeFuncs(self.max_particle_num_fluid, self.BasisType)
        for nb in range(self.BodyInfoFluid.shape[0]):
            if self.BodyInfoFluid[nb].Type == 0:
                self.lpf.AddRectangle(nb, self.BodyInfoFluid, self.lm, self.Npic)
            elif self.BodyInfoFluid[nb].Type == 1:
                self.lpf.AddTriangle(nb, self.BodyInfoFluid, self.lm, self.Npic)
            elif self.BodyInfoFluid[nb].Type == 2:
                self.lpf.AddCircle(nb, self.BodyInfoFluid, self.lm, self.Npic)
        if self.Algorithm == 3:
            self.lpf.StoreParticleLen(self.max_particle_num_fluid)
            self.lpf.GIMPInit(self.Npic)
        
        self.lps.StoreInitialPoros(self.max_particle_num_solid)
        self.lpf.StoreHydroProperties(self.max_particle_num_fluid)
        self.lps.SolidPhaseInit(self.BodyInfoSolid)
        self.lpf.FluidPhaseInit(self.BodyInfoFluid)
        return self.lps, self.lpf

    @ti.kernel
    def CriticalTimeStep(self, CFL: float):
        matList = self.lm
        SoundSpeed = 0.
        for nb in range(self.BodyInfoSolid.shape[0]):
            matID = self.BodyInfoSolid[nb].Mat
            bulk = matList.k[matID]
            shear = matList.g[matID]
            poros = self.BodyInfoSolid[nb].porosity
            mu = self.MatInfo[matID].mu
            c_2 = young * (1 - mu) / (1 + mu) / (1 - 2 * mu) / self.BodyInfo[nb].MacroRho
            ti.atomic_max(SoundSpeed, ti.sqrt(c_2))
        dt_c = ti.min(self.Dx[0], self.Dx[1]) / SoundSpeed
        self.Dt[None] = ti.min(CFL * dt_c, self.Dt[None])
    
    @ti.kernel
    def AdaptiveTimeScheme(self, CFL: float):
        pass
