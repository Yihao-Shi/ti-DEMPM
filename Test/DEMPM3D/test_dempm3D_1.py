from __init__ import *
import math
import MPMLib3D_v1.MPM as MPM
import DEMLib3D_v1.DEM as DEM
import DEMPMLib3D_v1.DEMPM as DEMPM
ti.init(arch=ti.cpu, default_fp=ti.f32, debug=True)


@ti.kernel
def InitialStress(mpm: ti.template()):
    for np in range(mpm.partList.particleNum[None]):
        mpm.partList.stress[np][2, 2] = -(mpm.partList.x[np][2] - (mpm.BodyInfo[0].pos0[2] + mpm.BodyInfo[0].len[2])) * mpm.Gravity[2] * mpm.BodyInfo[0].MacroRho
        mpm.partList.stress[np][0, 0] = mpm.partList.stress[np][1, 1] = mpm.partList.stress[np][2, 2] * mpm.MatInfo[0].mu / (1 - mpm.MatInfo[0].mu)


# Test for Linear contact models
if __name__ == "__main__":
    # DEM domain
    dem = DEM.DEM(cmType=0,                                                 # /Contact model type/ 0 for linear model; 1 for hertz model; 2 for linear rolling model; 3 for linear bond model
                  domain=ti.Vector([10, 0.5, 1]),                              # domain size
                  boundary=ti.Vector([0, 0, 0]),                            # periodic boundary condition
                  algorithm=0,                                              # /Integration scheme/ 0 for Euler; 1 for Verlet; 
                  gravity=ti.Vector([0., 0., -9.8]),                        # Gravity (body force)
                  timeStep=1.e-4,                                           # Time step
                  bodyNum=1,                                                # Taichi field of body parameters *Number of Body*
                  matNum=1,                                                 # Taichi field of material parameters *Number of Material*
                  wallNum=1,                                                # Taichi field of material parameters *Number of Wall*
                  searchAlgorithm=1,                                        # /Search Algorithm/ 0 for sorted based; 1 for linked cell;  
                  )
    
    # Physical parameters of particles
    dem.MatInfo[0].Kn = 1e6                                                 # Contact normal stiffness
    dem.MatInfo[0].Ks = 1e6                                                 # Contact tangential stiffness
    dem.MatInfo[0].Mu = 0.5                                                 # Friction coefficient
    dem.MatInfo[0].localDamping = [0., 0.]                                  # /Local damping/ Translation & Rolling
    dem.MatInfo[0].visDamping = [0., 0.]                                    # /Viscous damping/ Normal & Tangential
    dem.MatInfo[0].ParticleRho = 100                                        # Particle density

    # DEM body domain
    dem.BodyInfo[0].ID = 0                                                  # Body ID
    dem.BodyInfo[0].Mat = 0                                                 # Material Name of Body
    dem.BodyInfo[0].shapeType = 0                                           # /ShapeType of DEM_PARTICLE/ 0 for sphere; 1 for SDEM; 
    dem.BodyInfo[0].GenerateType = 0                                        # /Generate type/ 0 for create; 1 for generate; 2 for fill in box; 3 for wall
    dem.BodyInfo[0].pos0 = ti.Vector([0.5, 0.25, 0.7])                         # Initial position or center of sphere of Body
    dem.BodyInfo[0].rhi = 0.2                                              # Particle radius
    dem.BodyInfo[0].rlo = 0.2                                              # Particle radius
    dem.BodyInfo[0].pnum = 1                                                # The number of particle in the group
    dem.BodyInfo[0].v0 = ti.Vector([0, 0, 0])                               # Initial velocity
    dem.BodyInfo[0].w0 = ti.Vector([0, 0, 0])                               # Initial angular velocity
    dem.BodyInfo[0].fixedV = ti.Vector([0, 0, 0])                           # Fixed velocity
    dem.BodyInfo[0].fixedW = ti.Vector([1, 1, 1])                           # Fixed angular velocity
    dem.BodyInfo[0].orientation = ti.Vector([0, 0, 1])                      # Initial orientation
    dem.BodyInfo[0].DT = ti.Vector([0, 1e6, 5])                             # DT
    
    dem.WallInfo[0].ID = 0                                                  # Body ID
    dem.WallInfo[0].Mat = 0                                                 # Material Name of Body
    dem.WallInfo[0].point1 = ti.Vector([0, 0., 0])                          # The vertex of the wall
    dem.WallInfo[0].point2 = ti.Vector([5, 0, 0])                           # The vertex of the wall
    dem.WallInfo[0].point3 = ti.Vector([5, 5, 0])                           # The vertex of the wall
    dem.WallInfo[0].point4 = ti.Vector([0, 5, 0])                           # The vertex of the wall
    dem.WallInfo[0].norm = ti.Vector([0, 0, 1])                             # The norm of the wall

    # Create MPM domain
    dem.AddContactModel()
    dem.AddBodies(max_particle_num=1)
    dem.AddWall()
    dem.AddContactPair(max_contact_num=1)
    dem.AddNeighborList(multiplier=2, max_potential_particle_pairs=1, max_potential_wall_pairs=1)



    # MPM domain
    mpm = MPM.MPM(domain=ti.Vector([10, 0.5, 1]),                           # domain size
                  dx=ti.Vector([0.05, 0.05, 0.05]),                         # size of one grid
                  npic=2,                                                   # Number of MPM particles per subdomain
                  boundary=ti.Vector([0, 0, 0]),                            # periodic boundary condition
                  damping=0.,                                               # Damping ratio
                  algorithm=3,                                              # /Transfer algorithm/ 0 for USF; 1 for USL; 2 for MUSL; 3 for GIMP; 4 for MLS; 
                  shape_func=1,                                             # /Shape Functions/ 0 for Linear; 1 for GIMP; 2 for Quadratic BSpline; 3 for Cubic BSpline; 4 for NURBS
                  gravity=ti.Vector([0, 0, -9.8]),                          # Gravity (body force)
                  timeStep=1e-4,                                            # Time step
                  Stablization=0,                                           # /Stablization Technique/ 0 for NULL; 1 for Reduced Integration (deactivate); 2 for b-bar; 3 for f-bar (deactivate); 3 for cell-average
                  max_particle_num=3.2e5,                                     # The max number of particles
                  bodyNum=1,                                                # Taichi field of body parameters *Number of Body*
                  matNum=1,                                                 # Taichi field of material parameters *Number of Material*
                  alphaPIC=0.005,                                           # The weight of PIC velocity
                  Dynamic_Allocate=False,                                   # Sparse data structures
                  Contact_Detection=False                                   # Multibody contact detection
                  )

    # Physical parameters of particles
    mpm.MatInfo[0].Type = 0                                                 # /Constitutive model/ 0 for Hyper-elastic; 1 for Mohr-Coulomb; 2 for Drucker-Parger; 3 for Newtonian
    mpm.MatInfo[0].Modulus = 1e8                                            # Young's Modulus for soil; Bulk modulus for fluid
    mpm.MatInfo[0].mu = 0.3                                                 # Possion ratio

    # MPM body domain
    mpm.BodyInfo[0].ID = 0                                                  # Body ID
    mpm.BodyInfo[0].Mat = 0                                                 # Material Name of Body
    mpm.BodyInfo[0].Type = 0                                                # /Body shape/ 0 for rectangle; 1 for triangle; 2 for sphere
    mpm.BodyInfo[0].MacroRho = 2650                                         # Macro density
    mpm.BodyInfo[0].pos0 = ti.Vector([0., 0., 0.])                          # Initial position or center of sphere of Body
    mpm.BodyInfo[0].len = ti.Vector([10., 0.5, 0.5])                        # Size of Body
    mpm.BodyInfo[0].v0 = ti.Vector([0, 0, 0])                               # Initial velocity
    mpm.BodyInfo[0].fixedV = ti.Vector([0, 0, 0])                           # Fixed velocity
    mpm.BodyInfo[0].DT = ti.Vector([0, 1e6, 0])                             # DT

    # Create MPM domain
    Grid = mpm.AddGrid()
    Material = mpm.AddMaterial()
    Cell = mpm.AddCell()
    Particle = mpm.AddParticle()

    # Define boundary conditions
    for i, j, k in ti.ndrange((0, 201), (0, 11), (0, 2)):
        Grid.SetNonSlippingBC(i, j, k)
    for i, j, k in ti.ndrange((0, 201), (0, 2), (0, 21)):
        Grid.SetNonSlippingBC(i, j, k)
    for i, j, k in ti.ndrange((0, 201), (9, 11), (0, 21)):
        Grid.SetNonSlippingBC(i, j, k)
    for i, j, k in ti.ndrange((0, 2), (0, 11), (0, 21)):
        Grid.SetNonSlippingBC(i, j, k)
    for i, j, k in ti.ndrange((199, 201), (0, 11), (0, 21)):
        Grid.SetNonSlippingBC(i, j, k)

    # Define initial stress field
    #InitialStress(mpm)



    # DEMPM domain
    dempm = DEMPM.DEMPM(CMtype=0,                                           # /Contact model type/ 0 for penalty method; 1 for barrier method; 
                  dem=dem,                                                  # DEM pointer
                  mpm=mpm,                                                  # MPM pointer
                  )

    dempm.MPMContactInfo[0].Kn = 1e6                                        # Contact normal stiffness between MPM and DEM
    dempm.MPMContactInfo[0].Ks = 1e6                                        # Contact tangential stiffness between MPM and DEM
    dempm.MPMContactInfo[0].Mu = 0.5                                        # Friction coefficient between MPM and DEM
    dempm.MPMContactInfo[0].visDamping = [0., 0.]                           # /Viscous damping between MPM and DEM/ Normal & Tangential

    dempm.AddContactParams()
    dempm.AddContactPair(max_dempm_contact_num=5000)
    dempm.AddNeighborList(multiplier=2, scaler=0, max_potential_dempm_particle_pairs=20000)

    # Solver
    TIME: float = 1.5                                                       # Total simulation time
    saveTime: float = 0.01                                                  # save per time step
    CFL = 0.5                                                               # Courant-Friedrichs-Lewy condition
    vtkPath = './vtkDataTest1'                                              # VTK output path
    ascPath = './vtkDataTest1/postProcessing'                               # Monitoring data path

    dempm.Solver(TIME, saveTime, CFL, vtkPath, ascPath, adaptive=False)



