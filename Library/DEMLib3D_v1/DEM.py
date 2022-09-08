import taichi as ti
import DEMLib3D_v1.DEMWalls as Wall
import DEMLib3D_v1.ContactModels as ContactModels
import DEMLib3D_v1.DEMParticles as Particle
import DEMLib3D_v1.NeighborSearchAabb as NeighborSearchAabb
import DEMLib3D_v1.NeighborSearchLinkedCell as NeighborSearchLinkedCell
import DEMLib3D_v1.ContactPairs as ContactPairs
import DEMLib3D_v1.DEMEngine as Engine
import DEMLib3D_v1.TimeIntegrationDEM as TimeIntegrationDEM

@ti.data_oriented
class DEM:
    def __init__(self, cmType, domain, boundary, algorithm, 
                 gravity, timeStep, bodyNum, wallNum, matNum, searchAlgorithm):
        self.CmType = cmType
        self.Domain = domain
        self.Boundary = boundary
        self.Algorithm = algorithm
        self.Gravity = gravity
        self.Dt = ti.field(float, shape=())
        self.Dt[None] = timeStep
        self.SearchAlgorithm = searchAlgorithm

        self.bodyNum = int(bodyNum)
        self.wallNum = int(wallNum)
        self.matNum = int(matNum)

        self.MainLoop = None

        self.BodyInfo = ti.Struct.field({                                         # List of body parameters
            "ID": int,                                                            # Body ID
            "Mat": int,                                                           # Material name
            "shapeType": int,                                                     # /ShapeType of DEM_PARTICLE/ 0 for sphere; 1 for Cuboid; 2 for Tetrahedron 
            "GenerateType": int,                                                  # /Generate type/ 0 for create; 1 for generate; 2 for distribute; 3 for fill in box;

            "pos0": ti.types.vector(3, float),                                    # Initial position 
            "len": ti.types.vector(3, float),                                     # Size of box
            "rlo": float,                                                         # the minimum Radius
            "rhi": float,                                                         # the maximum Radius
            "pnum": int,                                                          # the number of particle in the group
            "VoidRatio": float,                                                   # The macro void ratio of granular assembly

            "fex":ti.types.vector(3, float),                                      # The externel force
            "tex":ti.types.vector(3, float),                                      # The externel torque
            "v0": ti.types.vector(3, float),                                      # Initial velocity
            "orientation": ti.types.vector(3, float),                             # Initial orientation
            "w0": ti.types.vector(3, float),                                      # Initial angular velocity
            "fixedV": ti.types.vector(3, int),                                    # Fixed velocity
            "fixedW": ti.types.vector(3, int),                                    # Fixed angular velocity
            "DT": ti.types.vector(3, float)                                       # DT
        }, shape=(self.bodyNum,))

        self.WallInfo = ti.Struct.field({                                         # List of wall parameters
            "ID": int,                                                            # Body ID
            "Mat": int,                                                           # Material name
            "point1": ti.types.vector(3, float),                                  # The vertex of the wall
            "point2": ti.types.vector(3, float),                                  # The vertex of the wall
            "point3": ti.types.vector(3, float),                                  # The vertex of the wall
            "point4": ti.types.vector(3, float),                                  # The vertex of the wall
            "norm": ti.types.vector(3, float),                                    # The wall normal (actived side)
            "fex":ti.types.vector(3, float),                                      # The externel force
            "tex":ti.types.vector(3, float),                                      # The externel torque
            "v0": ti.types.vector(3, float),                                      # Initial velocity
            "w0": ti.types.vector(3, float),                                      # Initial angular velocity
            "fixedV": ti.types.vector(3, int),                                    # Fixed velocity
            "fixedW": ti.types.vector(3, int),                                    # Fixed angular velocity
            "DT": ti.types.vector(3, float)                                       # DT
            }, shape=(self.wallNum,))
        
        self.MatInfo = ti.Struct.field({                                          # List of material parameters
            "ParticleRho": float,                                                 # Particle density 
            "Modulus": float,                                                     # Shear Modulus
            "possion": float,                                                     # Possion ratio
            "Kn": float,                                                          # Hardening
            "Ks": float,                                                          # Cohesion coefficient
            "Mu": float,                                                          # Angle of internal friction
            "Rmu": float,                                                         # Angle of dilatation
            "Tmu": float,                                                         # Angle of dilatation
            "cohesion": float,                                                    # Bond cohesion
            "ForceLocalDamping": float,                                           # Local Damping
            "TorqueLocalDamping": float,                                          # Local Damping
            "NormalViscousDamping": float,                                        # Viscous Damping
            "TangViscousDamping": float,                                          # Viscous Damping
            "Restitution": float                                                  # Restitution coefficient
        }, shape=(self.matNum,))

    
    def AddContactModel(self):
        print('------------------------ ContactModels Initialization ------------------------')
        for nm in range(self.MatInfo.shape[0]):
            if ti.static(self.CmType == 0):
                self.contModel = ContactModels.LinearContactModel(self.matNum, self.Dt[None])
            elif ti.static(self.CmType == 1):
                self.contModel = ContactModels.HertzMindlinContactModel(self.matNum, self.Dt[None])
            elif ti.static(self.CmType == 2):
                self.contModel = ContactModels.LinearRollingResistanceContactModel(self.matNum, self.Dt[None])
            elif ti.static(self.CmType == 3):
                self.contModel = ContactModels.LinearBondContactModel(self.matNum, self.Dt[None])
            elif ti.static(self.CmType == 4):
                self.contModel = ContactModels.LinearParallelBondContactModel(self.matNum, self.Dt[None])
            self.contModel.ParticleMaterialInit(nm, self.MatInfo)

    def AddBodies(self, max_particle_num):
        print('------------------------ Body Initialization ------------------------')
        self.partList = Particle.DEMParticle(max_particle_num, self.contModel, self.Gravity)
        for nb in range(self.BodyInfo.shape[0]):
            if ti.static(self.BodyInfo[nb].GenerateType == 0):
                self.partList.CreateSphere(nb, self.BodyInfo)
            elif ti.static(self.BodyInfo[nb].GenerateType == 1):
                self.partList.GenerateSphere(nb, self.BodyInfo)
            elif ti.static(self.BodyInfo[nb].GenerateType == 2):
                self.partList.DistributeSphere(nb, self.BodyInfo)
            elif ti.static(self.BodyInfo[nb].GenerateType == 3):
                self.partList.FillBallInBox(nb, self.BodyInfo)
        self.Rad_max = self.partList.FindMaxRadius()

    def AddWall(self):
        print('------------------------ Wall Initialization ------------------------')
        self.wallList = Wall.DEMWall(self.wallNum)
        for nw in range(self.WallInfo.shape[0]):
            self.wallList.CreatePlane(nw, self.WallInfo)
    
    def AddContactPair(self, max_contact_num):
        print('------------------------ Contact Pair Initialization ------------------------')
        if ti.static(self.CmType == 0):
            self.contPair = ContactPairs.Linear(max_contact_num, self.partList, self.wallList, self.contModel)
        elif ti.static(self.CmType == 1):
            self.contPair = ContactPairs.HertzMindlin(max_contact_num, self.partList, self.wallList, self.contModel)
        elif ti.static(self.CmType == 2):
            self.contPair = ContactPairs.LinearRollingResistance(max_contact_num, self.partList, self.wallList, self.contModel)
        elif ti.static(self.CmType == 3):
            self.contPair = ContactPairs.LinearContactBond(max_contact_num, self.partList, self.wallList, self.contModel)
        elif ti.static(self.CmType == 4):
            self.contPair = ContactPairs.LinearParallelBond(max_contact_num, self.partList, self.wallList, self.contModel)

    def AddNeighborList(self, multiplier, max_potential_particle_pairs, max_potential_wall_pairs):
        print('------------------------ Neighbor Initialization ------------------------')
        self.multiplier = multiplier
        self.DEMGridSize = self.multiplier * self.Rad_max
        if ti.static(self.SearchAlgorithm == 0):
            print("Neighbor Searching Algorithm: Axis-sligned Bounding Box\n")
            self.neighborList = NeighborSearchAabb.NeighborSearchAabb(self.Domain, max_potential_particle_pairs, max_potential_wall_pairs, self.partList, self.wallList, self.contPair)
        if ti.static(self.SearchAlgorithm == 1):
            print("Neighbor Searching Algorithm: Linked Cell\n")
            self.neighborList = NeighborSearchLinkedCell.NeighborSearchLinkedCell2(self.Domain, self.DEMGridSize, self.Rad_max, self.wallNum, max_potential_particle_pairs, max_potential_wall_pairs, self.partList, self.wallList, self.contPair)
            self.neighborList.CellInit()

        self.particle_safe_disp = (self.multiplier - 2) / 2.


    # ============================================== Solver ================================================= #
    def Solver(self, TIME, saveTime, CFL, vtkPath, ascPath, adaptive):
        print('------------------------ DEM Solver ------------------------')
        if self.MainLoop:
            self.MainLoop.UpdateSolver(TIME, saveTime, CFL, vtkPath, ascPath, adaptive)
            self.MainLoop.Solver()
        else:
            if ti.static(self.Algorithm == 0):
                print("Integration Scheme: Euler\n")
                self.Engine = Engine.Euler(self.Gravity, self.Dt[None], self.partList, self.contModel)
                self.MainLoop = TimeIntegrationDEM.SolverEuler(self, TIME, saveTime, CFL, vtkPath, ascPath, adaptive)
                self.MainLoop.TurnOnSolver(t=0., step=0, printNum=0)
                self.MainLoop.Solver()
            elif ti.static(self.Algorithm == 1):
                print("Integration Scheme: Verlet\n")
                self.Engine = Engine.Verlet(self.Gravity, self.Dt[None], self.partList, self.contModel)
                self.MainLoop = TimeIntegrationDEM.SolverVerlet(self, TIME, saveTime, CFL, vtkPath, ascPath, adaptive)
                self.MainLoop.TurnOnSolver(t=0., step=0, printNum=0)
                self.MainLoop.Solver()
