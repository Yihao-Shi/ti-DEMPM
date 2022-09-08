from __init__ import *
import MPMLib2D_v1.MPM as MPM
import MPMLib2D_v1.TimeIntegrationMPM as TimeIntegrationMPM
import math
ti.init(arch=ti.cpu, default_fp=ti.f32, debug=True)


@ti.kernel
def InitialStress(mpm: ti.template()):
    for np in range(mpm.lp.particleNum[None]):
        mpm.lp.stress[np][1, 1] = -(mpm.lp.x[np][1] - (mpm.BodyInfo[0].pos0[1] + mpm.BodyInfo[0].len[1])) * mpm.Gravity[1] * mpm.BodyInfo[0].MacroRho
        mpm.lp.stress[np][0, 0] = mpm.lp.stress[np][1, 1] * mpm.MatInfo[0].mu / (1 - mpm.MatInfo[0].mu)


# Test for Drucker-Prager models using GIMP
if __name__ == "__main__":
    # MPM domain
    mpm = MPM.MPM(domain=ti.Vector([0.8, 0.2]),                             # domain size
                  dx=ti.Vector([0.0025, 0.0025]),                           # size of one grid
                  npic=4,                                                   # Number of MPM particles per subdomain
                  boundary=ti.Vector([0, 0]),                               # periodic boundary condition
                  damping=0.002,                                            # Damping ratio
                  algorithm=3,                                              # /Transfer algorithm/ 0 for USF; 1 for USL; 2 for MUSL; 3 for GIMP; 4 for MLS; 
                  shape_func=1,                                             # /Shape Functions/ 0 for Linear; 1 for GIMP; 2 for Quadratic BSpline; 3 for Cubic BSpline; 4 for NURBS
                  gravity=ti.Vector([0, -9.8]),                             # Gravity (body force)
                  timeStep=1e-5,                                            # Time step
                  Stablization=0,                                           # /Stablization Technique/ 0 for NULL; 1 for Reduced Integration (deactivate); 2 for b-bar; 3 for f-bar (deactivate); 3 for cell-average
                  max_particle_num=6e4,                                     # The max number of particles
                  bodyNum=1,                                                # Taichi field of body parameters *Number of Body*
                  matNum=1,                                                 # Taichi field of material parameters *Number of Material*
                  alphaPIC=0.005,                                           # The weight of PIC velocity
                  Dynamic_Allocate=False,                                   # Sparse data structures
                  Contact_Detection=False                                   # Multibody contact detection
                  )

    # Physical parameters of particles
    mpm.MatInfo[0].Type = 2                                                 # /Constitutive model/ 0 for Hyper-elastic; 1 for Mohr-Coulomb; 2 for Drucker-Parger; 3 for Newtonian
    mpm.MatInfo[0].Modulus = 8.4e5                                          # Young's Modulus for soil; Bulk modulus for fluid
    mpm.MatInfo[0].mu = 0.3                                                 # Possion ratio
    mpm.MatInfo[0].InternalFric = 19.2 / 180 * math.pi                      # Internal friction angle
    mpm.MatInfo[0].Dilation = 0.                                            # Angle of dilatation
    mpm.MatInfo[0].Cohesion = 0.                                           # Cohesion coefficient
    mpm.MatInfo[0].Tensile = 0.                                             # Tensile cut-off
    mpm.MatInfo[0].dpType = 2                                               # Drucker-Prager model

    # MPM body domain
    mpm.BodyInfo[0].ID = 0                                                  # Body ID
    mpm.BodyInfo[0].Mat = 0                                                 # Material Name of Body
    mpm.BodyInfo[0].Type = 0                                                # /Body shape/ 0 for rectangle; 1 for triangle; 2 for sphere; 3 for flow
    mpm.BodyInfo[0].MacroRho = 2650                                         # Macro density
    mpm.BodyInfo[0].pos0 = ti.Vector([0.01, 0.01])                        # Initial position or center of sphere of Body
    mpm.BodyInfo[0].len = ti.Vector([0.2, 0.1])                             # Size of Body
    mpm.BodyInfo[0].v0 = ti.Vector([0, 0])                                  # Initial velocity
    mpm.BodyInfo[0].fixedV = ti.Vector([0, 0])                              # Fixed velocity
    mpm.BodyInfo[0].DT = ti.Vector([0, 1e6, 0])                             # DT

    # Solver
    TIME: float = 0.6                                                        # Total simulation time
    saveTime: float = 0.01                                                  # save per time step
    CFL = 0.5                                                               # Courant-Friedrichs-Lewy condition
    vtkPath = './vtkDataTest4'                                              # VTK output path
    ascPath = './vtkDataTest4/postProcessing'                               # Monitoring data path

    # Create MPM domain
    Grid = mpm.AddGrid()
    Material = mpm.AddMaterial()
    Particle = mpm.AddParticle()
    Cell = mpm.AddCell()

    # Define boundary conditions
    for i, j in ti.ndrange((0, 320), (3, 5)):
        Grid.SetFrictionBC(i, j, 0.5, ti.Vector([0, -1]))
    for i, j in ti.ndrange((3, 5), (0, 80)):
        Grid.SetNonSlippingBC(i, j)

    # Define initial stress field
    InitialStress(mpm)

    # Solver
    TimeIntegrationMPM.Solver(mpm, TIME, saveTime, CFL, vtkPath, ascPath, adaptive=False)

