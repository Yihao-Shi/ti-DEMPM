from __init__ import *
import MPMLib2D_v1.MPM as MPM
import MPMLib2D_v1.TimeIntegrationMPM as TimeIntegrationMPM
ti.init(arch=ti.cpu, default_fp=ti.f32, debug=True)


# Test for Hyper-Elastic models using GIMP
if __name__ == "__main__":
    # MPM domain
    mpm = MPM.MPM(domain=ti.Vector([50, 40]),                               # domain size
                  dx=ti.Vector([1, 1]),                                     # size of one grid
                  npic=4,                                                   # Number of MPM particles per subdomain
                  boundary=ti.Vector([0, 0]),                               # periodic boundary condition
                  damping=0.,                                               # Damping ratio
                  algorithm=2,                                              # /Transfer algorithm/ 0 for USF; 1 for USL; 2 for MUSL; 3 for GIMP; 4 for APIC; 5 for polyPIC; 6 for CPIC
                  shape_func=2,                                             # /Shape Functions/ 0 for Linear; 1 for GIMP; 2 for Quadratic BSpline; 3 for Cubic BSpline; 4 for NURBS
                  gravity=ti.Vector([0, -9.8]),                             # Gravity (body force)
                  timeStep=1.e-4,                                           # Time step
                  Stablization=0,                                           # /Stablization Technique/ 0 for NULL; 1 for Reduced Integration (deactivate); 2 for b-bar; 3 for f-bar (deactivate); 3 for cell-averages
                  max_particle_num=4e3,                                     # The max number of particles
                  bodyNum=1,                                                # Taichi field of body parameters *Number of Body*
                  matNum=1,                                                 # Taichi field of material parameters *Number of Material*
                  alphaPIC=0.005,                                           # The weight of PIC velocity
                  Dynamic_Allocate=False,                                   # Sparse data structures /Default: False/
                  Contact_Detection=False                                   # Multibody contact detection
                  )

    # Physical parameters of particles
    mpm.MatInfo[0].Type = 0                                                 # /Constitutive model/ 0 for Hyper-elastic; 1 for Mohr-Coulomb; 2 for Drucker-Parger; 3 for Newtonian
    mpm.MatInfo[0].Modulus = 2e4                                            # Young's Modulus
    mpm.MatInfo[0].mu = 0.3                                                 # Possion ratio

    # MPM body domain
    mpm.BodyInfo[0].ID = 0                                                  # Body ID
    mpm.BodyInfo[0].Mat = 0                                                 # Material Name of Body
    mpm.BodyInfo[0].Type = 0                                                # /Body shape/ 0 for rectangle; 1 for triangle; 2 for sphere; 3 for flow
    mpm.BodyInfo[0].MacroRho = 5                                            # Macro density
    mpm.BodyInfo[0].pos0 = ti.Vector([20, 20])                              # Initial position or center of sphere of Body
    mpm.BodyInfo[0].len = ti.Vector([15, 7])                                # Size of Body
    mpm.BodyInfo[0].v0 = ti.Vector([0, 0])                                  # Initial velocity
    mpm.BodyInfo[0].fixedV = ti.Vector([0, 0])                              # Fixed velocity
    mpm.BodyInfo[0].DT = ti.Vector([0, 1e6, 5])                             # DT

    # Solver
    TIME: float = 5                                                        # Total simulation time
    saveTime: float = 0.01                                                   # save per time step
    CFL = 0.5                                                               # Courant-Friedrichs-Lewy condition
    vtkPath = './vtkDataTest6'                                              # VTK output path
    ascPath = './vtkDataTest6/postProcessing'                               # Monitoring data path

    # Create MPM domain
    Grid = mpm.AddGrid()
    Material = mpm.AddMaterial()
    Particle = mpm.AddParticle()
    Cell = mpm.AddCell()

    # Define boundary conditions
    for i, j in ti.ndrange((19, 21), (0, 40)):
        Grid.SetNonSlippingBC(i, j)

    # Solve
    TimeIntegrationMPM.Solver(mpm, TIME, saveTime, CFL, vtkPath, ascPath, adaptive=False)


