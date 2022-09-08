from __init__ import *
import MPMLib2D_v1.MPM as MPM
import MPMLib2D_v1.TimeIntegrationMPM as TimeIntegrationMPM
import math
ti.init(arch=ti.cpu, default_fp=ti.f32, debug=True)


@ti.kernel
def InitialStress(mpm: ti.template()):
    for np in range(mpm.lp.particleNum[None]):
        if 10 < mpm.lp.x[np][0] < 40:
            mpm.lp.stress[np][1, 1] = -(mpm.lp.x[np][1] - 45) * mpm.Gravity[1] * mpm.BodyInfo[0].MacroRho
            mpm.lp.stress[np][0, 0] = mpm.lp.stress[np][1, 1] * mpm.MatInfo[0].mu / (1 - mpm.MatInfo[0].mu)
        elif 40 < mpm.lp.x[np][0] < 60:
            mpm.lp.stress[np][1, 1] = -(mpm.lp.x[np][1] - (-mpm.lp.x[np][1] + 105)) * mpm.Gravity[1] * mpm.BodyInfo[0].MacroRho
            mpm.lp.stress[np][0, 0] = mpm.lp.stress[np][1, 1] * mpm.MatInfo[0].mu / (1 - mpm.MatInfo[0].mu)
        elif mpm.lp.x[np][0] > 60:
            mpm.lp.stress[np][1, 1] = -(mpm.lp.x[np][1] - 25) * mpm.Gravity[1] * mpm.BodyInfo[0].MacroRho
            mpm.lp.stress[np][0, 0] = mpm.lp.stress[np][1, 1] * mpm.MatInfo[0].mu / (1 - mpm.MatInfo[0].mu)


# Test for Drucker-Prager models using GIMP
if __name__ == "__main__":
    # MPM domain
    mpm = MPM.MPM(domain=ti.Vector([130, 50]),                              # domain size
                  dx=ti.Vector([0.5, 0.5]),                                 # size of one grid
                  npic=2,                                                   # Number of MPM particles per subdomain
                  boundary=ti.Vector([0, 0]),                               # periodic boundary condition
                  damping=0.1,                                              # Damping ratio
                  algorithm=3,                                              # /Transfer algorithm/ 0 for USF; 1 for USL; 2 for MUSL; 3 for GIMP; 4 for APIC; 5 for polyPIC; 6 for CPIC
                  shape_func=2,                                             # /Shape Functions/ 0 for Linear; 1 for GIMP; 2 for Quadratic BSpline; 3 for Cubic BSpline; 4 for NURBS
                  gravity=ti.Vector([0, -9.8]),                             # Gravity (body force)
                  timeStep=1e-5,                                            # Time step
                  Stablization=0,                                           # /Stablization Technique/ 0 for NULL; 1 for Reduced Integration (deactivate); 2 for b-bar; 3 for f-bar (deactivate); 3 for cell-average
                  max_particle_num=6e5,                                     # The max number of particles
                  bodyNum=3,                                                # Taichi field of body parameters *Number of Body*
                  matNum=1,                                                 # Taichi field of material parameters *Number of Material*
                  alphaPIC=0.005,                                           # The weight of PIC velocity
                  Dynamic_Allocate=False,                                   # Sparse data structures
                  Contact_Detection=False                                   # Multibody contact detection
                  )

    # Physical parameters of particles
    mpm.MatInfo[0].Type = 2                                                 # /Constitutive model/ 0 for Hyper-elastic; 1 for Mohr-Coulomb; 2 for Drucker-Parger; 3 for Newtonian
    mpm.MatInfo[0].Modulus = 7e7                                            # Young's Modulus for soil; Bulk modulus for fluid
    mpm.MatInfo[0].mu = 0.3                                                 # Possion ratio
    mpm.MatInfo[0].InternalFric = 19.2 / 180 * math.pi                      # Internal friction angle
    mpm.MatInfo[0].Dilation = 0.                                            # Angle of dilatation
    mpm.MatInfo[0].Cohesion = 10e3                                          # Cohesion coefficient
    mpm.MatInfo[0].Tensile = 27.48e3                                        # Tensile cut-off
    mpm.MatInfo[0].dpType = 2                                               # Drucker-Prager model

    # MPM body domain
    mpm.BodyInfo[0].ID = 0                                                  # Body ID
    mpm.BodyInfo[0].Mat = 0                                                 # Material Name of Body
    mpm.BodyInfo[0].Type = 0                                                # /Body shape/ 0 for rectangle; 1 for triangle; 2 for sphere; 3 for flow
    mpm.BodyInfo[0].MacroRho = 2650                                         # Macro density
    mpm.BodyInfo[0].pos0 = ti.Vector([10, 10])                              # Initial position or center of sphere of Body
    mpm.BodyInfo[0].len = ti.Vector([110, 15])                              # Size of Body
    mpm.BodyInfo[0].v0 = ti.Vector([0, 0])                                  # Initial velocity
    mpm.BodyInfo[0].fixedV = ti.Vector([0, 0])                              # Fixed velocity
    mpm.BodyInfo[0].DT = ti.Vector([0, 1e6, 0])                             # DT

    mpm.BodyInfo[1].ID = 0                                                  # Body ID
    mpm.BodyInfo[1].Mat = 0                                                 # Material Name of Body
    mpm.BodyInfo[1].Type = 1                                                # /Body shape/ 0 for rectangle; 1 for triangle; 2 for sphere; 3 for flow
    mpm.BodyInfo[1].MacroRho = 2650                                         # Macro density
    mpm.BodyInfo[1].pos0 = ti.Vector([40, 25])                              # Initial position or center of sphere of Body
    mpm.BodyInfo[1].len = ti.Vector([20, 20])                               # Size of Body
    mpm.BodyInfo[1].v0 = ti.Vector([0, 0])                                  # Initial velocity
    mpm.BodyInfo[1].fixedV = ti.Vector([0, 0])                              # Fixed velocity
    mpm.BodyInfo[1].DT = ti.Vector([0, 1e6, 0])                             # DT

    mpm.BodyInfo[2].ID = 0                                                  # Body ID
    mpm.BodyInfo[2].Mat = 0                                                 # Material Name of Body
    mpm.BodyInfo[2].Type = 0                                                # /Body shape/ 0 for rectangle; 1 for triangle; 2 for sphere; 3 for flow
    mpm.BodyInfo[2].MacroRho = 2650                                         # Macro density
    mpm.BodyInfo[2].pos0 = ti.Vector([10, 25])                              # Initial position or center of sphere of Body
    mpm.BodyInfo[2].len = ti.Vector([30, 20])                               # Size of Body
    mpm.BodyInfo[2].v0 = ti.Vector([0, 0])                                  # Initial velocity
    mpm.BodyInfo[2].fixedV = ti.Vector([0, 0])                              # Fixed velocity
    mpm.BodyInfo[2].DT = ti.Vector([0, 1e6, 0])                             # DT

    # Solver
    TIME: float = 10                                                        # Total simulation time
    saveTime: float = 0.1                                                  # save per time step
    CFL = 0.5                                                               # Courant-Friedrichs-Lewy condition
    vtkPath = './vtkDataTest2'                                              # VTK output path
    ascPath = './vtkDataTest2/postProcessing'                               # Monitoring data path

    # Create MPM domain
    Grid = mpm.AddGrid()
    Material = mpm.AddMaterial()
    Particle = mpm.AddParticle()
    Cell = mpm.AddCell()

    # Define boundary conditions
    for i, j in ti.ndrange((0, 261), (19, 21)):
        Grid.SetNonSlippingBC(i, j)
    for i, j in ti.ndrange((19, 21), (0, 101)):
        Grid.SetSlippingBC(i, j, ti.Vector([-1, 0]))
    for i, j in ti.ndrange((239, 241), (0, 101)):
        Grid.SetSlippingBC(i, j, ti.Vector([1, 0]))

    # Define initial stress field
    InitialStress(mpm)

    # Solver
    TimeIntegrationMPM.Solver(mpm, TIME, saveTime, CFL, vtkPath, ascPath, adaptive=False)

