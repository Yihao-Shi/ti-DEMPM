from __init__ import *
import math
import MPMLib3D.MPM as MPM
import DEMLib3D.DEM as DEM
import DEMPMLib3D.DEMPM as DEMPM
ti.init(arch=ti.cpu, default_fp=ti.f32, debug=False)


@ti.kernel
def InitialStress(mpm: ti.template()):
    for np in range(mpm.partList.particleNum[None]):
        mpm.partList.stress[np][2, 2] = -(mpm.partList.x[np][2] - (mpm.BodyInfo[0].pos0[2] + mpm.BodyInfo[0].len[2])) * mpm.Gravity[2] * mpm.BodyInfo[0].MacroRho
        mpm.partList.stress[np][0, 0] = mpm.partList.stress[np][1, 1] = mpm.partList.stress[np][2, 2] * mpm.MatInfo[0].mu / (1 - mpm.MatInfo[0].mu)


# Test for Linear contact models
if __name__ == "__main__":
    # DEM domain
    dem = DEM.DEM(cmType=0,                                                 # /Contact model type/ 0 for linear model; 1 for hertz model; 2 for linear rolling model; 3 for linear bond model
                  domain=ti.Vector([1.02, 0.1, 0.5]),                       # domain size
                  boundary=ti.Vector([0, 0, 0]),                            # periodic boundary condition
                  algorithm=0,                                              # /Integration scheme/ 0 for Euler; 1 for Verlet; 
                  gravity=ti.Vector([0., 0., -9.8]),                        # Gravity (body force)
                  timeStep=1.e-5,                                           # Time step
                  bodyNum=1,                                                # Taichi field of body parameters *Number of Body*
                  matNum=1,                                                 # Taichi field of material parameters *Number of Material*
                  wallNum=0,                                                # Taichi field of material parameters *Number of Wall*
                  searchAlgorithm=1,                                        # /Search Algorithm/ 0 for sorted based; 1 for linked cell;  
                  )
    
    # Physical parameters of particles
    dem.MatInfo[0].Kn = 2e7                                                 # Contact normal stiffness
    dem.MatInfo[0].Ks = 1e4                                                 # Contact tangential stiffness
    dem.MatInfo[0].Mu = 0.2                                                 # Friction coefficient
    dem.MatInfo[0].ForceLocalDamping = 0.                                   # /Local damping/ 
    dem.MatInfo[0].TorqueLocalDamping = 0.                                  # /Local damping/ 
    dem.MatInfo[0].NormalViscousDamping = 0.                                # /Viscous damping/ 
    dem.MatInfo[0].TangViscousDamping = 0.                                  # /Viscous damping/
    dem.MatInfo[0].ParticleRho = 2061                                     # Particle density

    # DEM body domain
    dem.BodyInfo[0].ID = 0                                                  # Body ID
    dem.BodyInfo[0].Mat = 0                                                 # Material Name of Body
    dem.BodyInfo[0].shapeType = 0                                           # /ShapeType of DEM_PARTICLE/ 0 for sphere; 1 for SDEM; 
    dem.BodyInfo[0].GenerateType = 0                                        # /Generate type/ 0 for create; 1 for generate; 2 for fill in box; 3 for wall
    dem.BodyInfo[0].pos0 = ti.Vector([0.51, 0.05, 0.3123])                  # Initial position or center of sphere of Body
    dem.BodyInfo[0].rhi = 0.0223                                            # Particle radius
    dem.BodyInfo[0].rlo = 0.0223                                            # Particle radius
    dem.BodyInfo[0].pnum = 1                                                # The number of particle in the group
    dem.BodyInfo[0].v0 = ti.Vector([0, 0, -3.03])                           # Initial velocity
    dem.BodyInfo[0].w0 = ti.Vector([0, 0, 0])                               # Initial angular velocity
    dem.BodyInfo[0].fixedV = ti.Vector([0, 0, 0])                           # Fixed velocity
    dem.BodyInfo[0].fixedW = ti.Vector([0, 0, 0])                           # Fixed angular velocity
    dem.BodyInfo[0].orientation = ti.Vector([0, 0, 1])                      # Initial orientation
    dem.BodyInfo[0].DT = ti.Vector([0, 1e6, 5])                             # DT

    # Create MPM domain
    dem.AddContactModel()
    dem.AddBodies(max_particle_num=1)
    dem.AddWall(max_facet_num=1)
    dem.AddContactPair(max_contact_num=1)
    dem.AddNeighborList(multiplier=2, max_potential_particle_pairs=1, max_potential_wall_pairs=1)



    # MPM domain
    mpm = MPM.MPM(domain=ti.Vector([1.02, 0.1, 0.5]),                       # domain size
                  dx=ti.Vector([0.01, 0.01, 0.01]),                         # size of one grid
                  npic=2,                                                   # Number of MPM particles per subdomain
                  boundary=ti.Vector([0, 0, 0]),                            # periodic boundary condition
                  damping=0.,                                             # Damping ratio
                  algorithm=3,                                              # /Transfer algorithm/ 0 for USF; 1 for USL; 2 for MUSL; 3 for GIMP; 4 for MLS; 
                  shape_func=1,                                             # /Shape Functions/ 0 for Linear; 1 for GIMP; 2 for Quadratic BSpline; 3 for Cubic BSpline; 4 for NURBS
                  gravity=ti.Vector([0, 0, -9.8]),                          # Gravity (body force)
                  timeStep=1e-5,                                            # Time step
                  Stablization=0,                                           # /Stablization Technique/ 0 for NULL; 1 for Reduced Integration (deactivate); 2 for b-bar; 3 for f-bar (deactivate); 3 for cell-average
                  max_particle_num=2.4e5,                                   # The max number of particles
                  bodyNum=1,                                                # Taichi field of body parameters *Number of Body*
                  matNum=1,                                                 # Taichi field of material parameters *Number of Material*
                  alphaPIC=0.005,                                           # The weight of PIC velocity
                  Dynamic_Allocate=False,                                   # Sparse data structures
                  Contact_Detection=False                                   # Multibody contact detection
                  )

    # Physical parameters of particles
    mpm.MatInfo[0].Type = 2                                                 # /Constitutive model/ 0 for Hyper-elastic; 1 for Mohr-Coulomb; 2 for Drucker-Parger; 3 for Newtonian
    mpm.MatInfo[0].Modulus = 7.5e4                                          # Young's Modulus for soil; Bulk modulus for fluid
    mpm.MatInfo[0].mu = 0.2                                                 # Possion ratio
    mpm.MatInfo[0].InternalFric = 12 / 180 * math.pi                        # Internal friction angle
    mpm.MatInfo[0].Dilation = 0 / 180 * math.pi                            # Angle of dilatation
    mpm.MatInfo[0].Cohesion = 0.                                            # Cohesion coefficient
    mpm.MatInfo[0].Tensile = 0.                                             # Tensile cut-off
    mpm.MatInfo[0].dpType = 2                                               # Drucker-Prager model

    # MPM body domain
    mpm.BodyInfo[0].ID = 0                                                  # Body ID
    mpm.BodyInfo[0].Mat = 0                                                 # Material Name of Body
    mpm.BodyInfo[0].Type = 0                                                # /Body shape/ 0 for rectangle; 1 for triangle; 2 for sphere
    mpm.BodyInfo[0].MacroRho = 200                                          # Macro density
    mpm.BodyInfo[0].pos0 = ti.Vector([0., 0., 0.])                          # Initial position or center of sphere of Body
    mpm.BodyInfo[0].len = ti.Vector([1.02, 0.1, 0.29])                      # Size of Body
    mpm.BodyInfo[0].v0 = ti.Vector([0, 0, 0])                               # Initial velocity
    mpm.BodyInfo[0].fixedV = ti.Vector([0, 0, 0])                           # Fixed velocity
    mpm.BodyInfo[0].DT = ti.Vector([0, 1e6, 0])                             # DT

    # Create MPM domain
    Grid = mpm.AddGrid()
    Material = mpm.AddMaterial()
    Cell = mpm.AddCell()
    Particle = mpm.AddParticle()

    # Define boundary conditions
    for i, j, k in ti.ndrange((0, 102), (0, 11), (0, 2)):
        Grid.SetSlippingBC(i, j, k, ti.Vector([0, 0, -1]))
    for i, j, k in ti.ndrange((0, 102), (0, 11), (49, 51)):
        Grid.SetSlippingBC(i, j, k, ti.Vector([0, 0, 1]))
    for i, j, k in ti.ndrange((0, 102), (0, 2), (0, 51)):
        Grid.SetSlippingBC(i, j, k, ti.Vector([0, -1, 0]))
    for i, j, k in ti.ndrange((0, 102), (9, 11), (0, 51)):
        Grid.SetSlippingBC(i, j, k, ti.Vector([0, 1, 0]))
    for i, j, k in ti.ndrange((0, 2), (0, 11), (0, 51)):
        Grid.SetSlippingBC(i, j, k, ti.Vector([-1, 0, 0]))
    for i, j, k in ti.ndrange((101, 103), (0, 11), (0, 51)):
        Grid.SetSlippingBC(i, j, k, ti.Vector([1, 0, 0]))

    # Define initial stress field
    InitialStress(mpm)



    # DEMPM domain
    dempm = DEMPM.DEMPM(CMtype=0,                                           # /Contact model type/ 0 for penalty method; 1 for barrier method; 
                  dem=dem,                                                  # DEM pointer
                  mpm=mpm,                                                  # MPM pointer
                  )

    dempm.MPMContactInfo[0].Kn = 1e6                                        # Contact normal stiffness between MPM and DEM
    dempm.MPMContactInfo[0].Ks = 1e4                                        # Contact tangential stiffness between MPM and DEM
    dempm.MPMContactInfo[0].Mu = 0.2                                        # Friction coefficient between MPM and DEM
    dempm.MPMContactInfo[0].NormalViscousDamping = 0.                       # /Viscous damping/ 
    dempm.MPMContactInfo[0].TangViscousDamping = 0.                         # /Viscous damping/

    dempm.AddContactParams()
    dempm.AddContactPair(max_dempm_contact_num=500)
    dempm.AddNeighborList(multiplier=2, scaler=0, max_potential_dempm_particle_pairs=80000)

    # Solver
    TIME: float = 0.2                                                       # Total simulation time
    saveTime: float = 0.002                                                  # save per time step
    CFL = 0.5                                                               # Courant-Friedrichs-Lewy condition
    vtkPath = './vtkDataTest3'                                              # VTK output path
    ascPath = './vtkDataTest3/postProcessing/vel_330'                       # Monitoring data path

    dempm.Solver(TIME, saveTime, CFL, vtkPath, ascPath, adaptive=False)



