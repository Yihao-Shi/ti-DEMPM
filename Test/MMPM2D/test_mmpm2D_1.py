from __init__ import *
import DEMPMLib2D_v1.MPM as MPM
import DEMPMLib2D_v1.TimeIntegrationMMPM as TimeIntegrationMMPM
import math
ti.init(arch=ti.cpu, default_fp=ti.f64, debug=True)


@ti.kernel
def InitialStressFluid(mpm: ti.template()):
    for np in range(mpm.lp.particleNum[None]):
        mpm.lp.P[np] = -10000
        mpm.lp.stress[np][0, 0] = mpm.lp.stress[np][1, 1] = mpm.lp.P[np]


@ti.kernel
def InitialStressSolid(mpm: ti.template()):
    for np in range(mpm.lp.particleNum[None]):
        mpm.lp.stress[np][1, 1] = -(mpm.lp.x[np][1] - (mpm.BodyInfo[0].pos0[1] + mpm.BodyInfo[0].len[1])) * mpm.Gravity[1] * mpm.BodyInfo[0].MacroRho
        mpm.lp.stress[np][0, 0] = mpm.lp.stress[np][1, 1] * mpm.MatInfo[0].mu / (1 - mpm.MatInfo[0].mu)


# Test for Newtonian models using GIMP
if __name__ == "__main__":
    # MPM domain
    mpm1 = MPM.MPM(domain=ti.Vector([0.06, 1.5]),                           # domain size
                  dx=ti.Vector([0.01, 0.01]),                               # size of one grid
                  npic=2,                                                   # Number of MPM particles per subdomain
                  boundary=ti.Vector([0, 0]),                               # periodic boundary condition
                  damping=0.002,                                            # Damping ratio
                  algorithm=3,                                              # /Transfer algorithm/ 0 for USF; 1 for USL; 2 for MUSL; 3 for GIMP; 4 for APIC; 5 for polyPIC; 6 for CPIC
                  gravity=ti.Vector([0, -0.01]),                             # Gravity (body force)
                  timeStep=1e-7,                                            # Time step
                  max_particle_num=2.4e3,                                     # The max number of particles
                  bodyNum=1,                                                # Taichi field of body parameters *Number of Body*
                  matNum=1,                                                 # Taichi field of material parameters *Number of Material*
                  alphaPIC=0.005,                                           # The weight of PIC velocity
                  Dynamic_Allocate=False,                                   # Sparse data structures
                  Contact_Detection=False                                   # Multibody contact detection
                  )

    # Physical parameters of particles
    mpm1.MatInfo[0].Type = 3                                                 # /Constitutive model/ 0 for Hyper-elastic; 1 for Mohr-Coulomb; 2 for Drucker-Parger; 3 for Newtonian
    mpm1.MatInfo[0].Modulus = 3.6e5                                          # Young's Modulus for soil; Bulk modulus for fluid
    mpm1.MatInfo[0].Viscosity = 1.e-3                                        # Angle of dilatation
    mpm1.MatInfo[0].SoundSpeed = 100                                         # Cohesion coefficient

    # MPM body domain
    mpm1.BodyInfo[0].ID = 0                                                  # Body ID
    mpm1.BodyInfo[0].Mat = 0                                                 # Material Name of Body
    mpm1.BodyInfo[0].Type = 0                                                # /Body shape/ 0 for rectangle; 1 for triangle; 2 for sphere
    mpm1.BodyInfo[0].MacroRho = 1000                                         # Macro density
    mpm1.BodyInfo[0].pos0 = ti.Vector([0, 0])                          # Initial position or center of sphere of Body
    mpm1.BodyInfo[0].len = ti.Vector([0.06, 1.0])                             # Size of Body
    mpm1.BodyInfo[0].v0 = ti.Vector([0, 0])                                  # Initial velocity
    mpm1.BodyInfo[0].fixedV = ti.Vector([0, 0])                              # Fixed velocity
    mpm1.BodyInfo[0].Darcy_k = 1e12                                         # Soild permenability
    mpm1.BodyInfo[0].porosity = 1                                         # Solid void ratio
    mpm1.BodyInfo[0].DT = ti.Vector([0, 1e6, 0])                             # DT

    # Create MPM domain
    mpm1.AddGrid()
    mpm1.AddParticles()
    mpm1.AddMaterial()

    # Define boundary conditions
    for i, j in ti.ndrange((0, 7), (0, 1)):
        mpm1.SetSlippingBC(i, j, ti.Vector([0, -1]))
    for i, j in ti.ndrange((0, 7), (100, 101)):
        mpm1.SetSlippingBC(i, j, ti.Vector([0, 1]))
    for i, j in ti.ndrange((0, 1), (0, 151)):
        mpm1.SetSlippingBC(i, j, ti.Vector([-1, 0]))
    for i, j in ti.ndrange((6, 7), (0, 151)):
        mpm1.SetSlippingBC(i, j, ti.Vector([1, 0]))

    # Define initial stress field
    # InitialStressFluid(mpm1)

    # MPM domain
    mpm2 = MPM.MPM(domain=ti.Vector([0.06, 1.5]),                               # domain size
                  dx=ti.Vector([0.01, 0.01]),                                     # size of one grid
                  npic=2,                                                   # Number of MPM particles per subdomain
                  periodic=ti.Vector([0, 0]),                               # periodic boundary condition
                  damping=0.2,                                               # Damping ratio
                  algorithm=3,                                              # /Transfer algorithm/ 0 for USF; 1 for USL; 2 for MUSL; 3 for GIMP; 4 for APIC; 5 for polyPIC; 6 for CPIC
                  shape_func=2,                                             # /Shape Functions/ 0 for Linear; 1 for GIMP; 2 for Quadratic BSpline; 3 for Cubic BSpline; 4 for NURBS
                  gravity=ti.Vector([0, -0.01]),                             # Gravity (body force)
                  timeStep=1.e-7,                                           # Time step
                  b_bar_method=0,                                           # B Bar Method
                  max_particle_num=2.4e3,                                     # The max number of particles
                  bodyNum=1,                                                # Taichi field of body parameters *Number of Body*
                  matNum=1,                                                 # Taichi field of material parameters *Number of Material*
                  alphaPIC=0.005,                                           # The weight of PIC velocity
                  Dynamic_Allocate=False,                                   # Sparse data structures /Default: False/
                  Contact_Detection=False                                   # Multibody contact detection
                  )

    # Physical parameters of particles
    mpm2.MatInfo[0].Type = 0                                                 # /Constitutive model/ 0 for Hyper-elastic; 1 for Mohr-Coulomb; 2 for Drucker-Parger; 3 for Newtonian
    mpm2.MatInfo[0].Modulus = 1e6                                          # Young's Modulus for soil; Bulk modulus for fluid
    mpm2.MatInfo[0].mu = 0.3                                                 # Possion ratio
    mpm2.MatInfo[0].InternalFric = 19.2 / 180 * math.pi                      # Internal friction angle
    mpm2.MatInfo[0].Dilation = 0.                                            # Angle of dilatation
    mpm2.MatInfo[0].Cohesion = 0.                                           # Cohesion coefficient
    mpm2.MatInfo[0].Tensile = 0.                                             # Tensile cut-off
    mpm2.MatInfo[0].dpType = 2                                               # Drucker-Prager model

    # MPM body domain
    mpm2.BodyInfo[0].ID = 0                                                  # Body ID
    mpm2.BodyInfo[0].Mat = 0                                                 # Material Name of Body
    mpm2.BodyInfo[0].Type = 0                                                # /Body shape/ 0 for rectangle; 1 for triangle; 2 for sphere; 3 for flow
    mpm2.BodyInfo[0].MacroRho = 2650                                            # Macro density
    mpm2.BodyInfo[0].pos0 = ti.Vector([0, 0])                              # Initial position or center of sphere of Body
    mpm2.BodyInfo[0].len = ti.Vector([0.06, 1])                                # Size of Body
    mpm2.BodyInfo[0].v0 = ti.Vector([0, 0])                                  # Initial velocity
    mpm2.BodyInfo[0].Darcy_k = 2e-4                                         # Soild permenability
    mpm2.BodyInfo[0].porosity = 0.3                                         # Solid void ratio
    mpm2.BodyInfo[0].fixedV = ti.Vector([0, 0])                              # Fixed velocity
    mpm2.BodyInfo[0].DT = ti.Vector([0, 1e6, 5])                             # DT

    # Solver
    TIME: float = 1                                                        # Total simulation time
    saveTime: float = 1e-3                                                   # save per time step
    CFL = 0.5                                                               # Courant-Friedrichs-Lewy condition
    vtkPath = './vtkDataTest1'                                              # VTK output path
    ascPath = './vtkDataTest1/postProcessing'                               # Monitoring data path

    # Create MPM domain
    mpm2.AddGrid()
    mpm2.AddMaterial()
    mpm2.AddParticles()

    # Define Dirichlet boundary conditions
    for i, j in ti.ndrange((0, 7), (0, 1)):
        mpm2.SetSlippingBC(i, j, ti.Vector([0, -1]))
    for i, j in ti.ndrange((0, 7), (150, 151)):
        mpm2.SetSlippingBC(i, j, ti.Vector([0, 1]))
    for i, j in ti.ndrange((0, 1), (0, 151)):
        mpm2.SetSlippingBC(i, j, ti.Vector([-1, 0]))
    for i, j in ti.ndrange((6, 7), (0, 151)):
        mpm2.SetSlippingBC(i, j, ti.Vector([1, 0]))

    # Define Newman boundary conditions
    SoilTractionBD = ti.Vector.field(2, int, 6)
    FluidFreeBD = ti.Vector.field(2, int, 6)
    mpm1.FindTractionBoundary(0, 6, 99, 99, FluidFreeBD, 0)
    mpm2.FindTractionBoundary(0, 6, 99, 99, SoilTractionBD, 0)
    f = ti.Vector([0, -500])
    mpm2.SetSolidNewmanBC(SoilTractionBD, f)


    # Define initial stress field
    InitialStressSolid(mpm2)

    # Solve
    TimeIntegrationMMPM.Solver(mpm1, mpm2, TIME, saveTime, CFL, vtkPath, ascPath, adaptive=False)

    '''for i, j in ti.ndrange((0, 7), (100, 101)):
        mpm1.ReSetDirichletBC(i, j)
    for i, j in ti.ndrange((0, 7), (150, 151)):
        mpm1.SetSlippingBC(i, j, ti.Vector([0, 1]))
    mpm2.Damp = 0.002
    # Define Free Surface
    mpm1.SetFreeSurfaceBC(FluidFreeBD)
    TIME: float = 1                                                        # Total simulation time
    saveTime: float = 1e-3                                                   # save per time step
    TimeIntegrationMMPM.Solver(mpm1, mpm2, TIME, saveTime, CFL, vtkPath, ascPath, adaptive=False)'''
