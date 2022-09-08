from __init__ import *
import MPMLib3D_v1.MPM as MPM
import math
ti.init(arch=ti.cpu, default_fp=ti.f32, debug=True)


@ti.kernel
def InitialStress(mpm: ti.template()):
    for np in range(mpm.partList.particleNum[None]):
        mpm.partList.stress[np][2, 2] = -(mpm.partList.x[np][2] - (mpm.BodyInfo[0].pos0[2] + mpm.BodyInfo[0].len[2])) * mpm.Gravity[2] * mpm.BodyInfo[0].MacroRho
        mpm.partList.stress[np][0, 0] = mpm.partList.stress[np][1, 1] = mpm.partList.stress[np][2, 2] * mpm.MatInfo[0].mu / (1 - mpm.MatInfo[0].mu)


# Test for Drucker-Prager models using GIMP
if __name__ == "__main__":
    # MPM domain
    mpm = MPM.MPM(domain=ti.Vector([0.8, 0.2, 0.2]),                             # domain size
                  dx=ti.Vector([0.0025, 0.0025, 0.0025]),                           # size of one grid
                  npic=1,                                                   # Number of MPM particles per subdomain
                  boundary=ti.Vector([0, 0, 0]),                               # periodic boundary condition
                  damping=0.002,                                            # Damping ratio
                  algorithm=3,                                              # /Transfer algorithm/ 0 for USF; 1 for USL; 2 for MUSL; 3 for GIMP; 4 for MLS; 
                  shape_func=1,                                             # /Shape Functions/ 0 for Linear; 1 for GIMP; 2 for Quadratic BSpline; 3 for Cubic BSpline; 4 for NURBS
                  gravity=ti.Vector([0, 0, -9.8]),                             # Gravity (body force)
                  timeStep=1e-5,                                            # Time step
                  Stablization=2,                                           # /Stablization Technique/ 0 for NULL; 1 for Reduced Integration (deactivate); 2 for b-bar; 3 for f-bar (deactivate); 3 for cell-average
                  max_particle_num=1.28e5,                                     # The max number of particles
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
    mpm.BodyInfo[0].pos0 = ti.Vector([0.005, 0.005, 0.005])                        # Initial position or center of sphere of Body
    mpm.BodyInfo[0].len = ti.Vector([0.2, 0.05, 0.1])                             # Size of Body
    mpm.BodyInfo[0].v0 = ti.Vector([0, 0, 0])                                  # Initial velocity
    mpm.BodyInfo[0].fixedV = ti.Vector([0, 0, 0])                              # Fixed velocity
    mpm.BodyInfo[0].DT = ti.Vector([0, 1e6, 0])                             # DT

    # Solver
    TIME: float = 1.5                                                        # Total simulation time
    saveTime: float = 0.01                                                  # save per time step
    CFL = 0.5                                                               # Courant-Friedrichs-Lewy condition
    vtkPath = './vtkDataTest6'                                              # VTK output path
    ascPath = './vtkDataTest6/postProcessing'                               # Monitoring data path

    # Create MPM domain
    Grid = mpm.AddGrid()
    Material = mpm.AddMaterial()
    Cell = mpm.AddCell()
    Particle = mpm.AddParticle()

    # Define boundary conditions
    for i, j, k in ti.ndrange((0, 321), (0, 81), (1, 3)):
        Grid.SetFrictionBC(i, j, k, 0.8, ti.Vector([0, 0, -1]))
    for i, j, k in ti.ndrange((1, 3), (0, 81), (0, 81)):
        Grid.SetNonSlippingBC(i, j, k)
    for i, j, k in ti.ndrange((0, 321), (1, 3), (0, 81)):
        Grid.SetSlippingBC(i, j, k, ti.Vector([0, -1, 0]))
    for i, j, k in ti.ndrange((0, 321), (22, 24), (0, 81)):
        Grid.SetSlippingBC(i, j, k, ti.Vector([0, 1, 0]))

    # Define initial stress field
    InitialStress(mpm)

    # Solver
    mpm.Solver(TIME, saveTime, CFL, vtkPath, ascPath, adaptive=False)

