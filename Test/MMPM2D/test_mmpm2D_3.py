from __init__ import *
import MMPMLib2D_v1.MMPM as MMPM
import MMPMLib2D_v1.TimeIntegrationMMPM as TimeIntegrationMMPM
import math
ti.init(arch=ti.cpu, default_fp=ti.f32, debug=True)


@ti.kernel
def InitialStressFluid(partList: ti.template(), BodyInfo: ti.template(), Gravity: ti.template()):
    for np in range(partList.particleNum[None]):
        bodyID = partList.bodyID[np]
        partList.P[np] = (partList.x[np][1] - (BodyInfo[bodyID].pos0[1] + BodyInfo[bodyID].len[1])) * Gravity[1] * BodyInfo[bodyID].MacroRho
        partList.stress[np][0, 0] = partList.stress[np][1, 1] = partList.P[np]


@ti.kernel
def InitialStressSolid(partList: ti.template(), BodyInfoSolid: ti.template(), BodyInfoFluid: ti.template(), MatInfo: ti.template(), Gravity: ti.template()):
    for np in range(partList.particleNum[None]):
        matID = partList.materialID[np]
        bodyID = partList.bodyID[np]
        partList.stress[np][1, 1] = -(partList.x[np][1] - (BodyInfoSolid[bodyID].pos0[1] + BodyInfoSolid[bodyID].len[1])) * Gravity[1] * (BodyInfoSolid[bodyID].MacroRho - BodyInfoFluid[0].MacroRho)
        partList.stress[np][0, 0] = partList.stress[np][1, 1] * MatInfo[matID].mu / (1 - MatInfo[matID].mu)


# Test for Newtonian models using GIMP
if __name__ == "__main__":
    # MMPM domain
    mmpm = MMPM.MMPM(domain=ti.Vector([0.35, 0.22]),                          # domain size
                  dx=ti.Vector([0.002, 0.002]),                             # size of one grid
                  npic=4,                                                   # Number of MPM particles per subdomain
                  boundary=ti.Vector([0, 0]),                               # periodic boundary condition
                  damping=0.002,                                             # Damping ratio
                  algorithm=3,                                              # /Transfer algorithm/ 0 for USF; 1 for USL; 2 for MUSL; 3 for GIMP; 4 for APIC; 5 for polyPIC; 6 for CPIC
                  shape_func=1,                                             # /Shape Functions/ 0 for Linear; 1 for GIMP; 2 for Quadratic BSpline; 3 for Cubic BSpline; 4 for NURBS
                  gravity=ti.Vector([0, -9.8]),                             # Gravity (body force)
                  timeStep=5e-6,                                            # Time step
                  max_particle_num_solid=9.6e3,                             # The max number of particles (Solid Phase)
                  max_particle_num_fluid=9.6e3,                             # The max number of particles (Fluid Phase)
                  bodyNum_solid=1,                                          # Taichi field of body parameters *Number of Body*
                  bodyNum_fluid=1,                                          # Taichi field of body parameters *Number of Body*
                  matNum=2,                                                 # Taichi field of material parameters *Number of Material*
                  alphaPIC=0.005,                                            # The weight of PIC velocity
                  Dynamic_Allocate=False,                                   # Sparse data structures
                  Contact_Detection=False                                   # Multibody contact detection
                  )

    # Physical parameters of particles
    mmpm.MatInfo[0].Type = 2                                                # /Constitutive model/ 0 for Hyper-elastic; 1 for Mohr-Coulomb; 2 for Drucker-Parger; 3 for Newtonian
    mmpm.MatInfo[0].Modulus = 1e7                                           # Young's Modulus for soil; Bulk modulus for fluid
    mmpm.MatInfo[0].mu = 0.3                                                # Possion ratio
    mmpm.MatInfo[0].InternalFric = 35 / 180 * math.pi                       # Internal friction angle
    mmpm.MatInfo[0].Dilation = 0.                                           # Angle of dilatation
    mmpm.MatInfo[0].Cohesion = 0.                                           # Cohesion coefficient
    mmpm.MatInfo[0].Tensile = 0.                                            # Tensile cut-off
    mmpm.MatInfo[0].dpType = 2                                              # Drucker-Prager model

    mmpm.MatInfo[1].Type = 3                                                # /Constitutive model/ 0 for Hyper-elastic; 1 for Mohr-Coulomb; 2 for Drucker-Parger; 3 for Newtonian
    mmpm.MatInfo[1].Modulus = 1e5                                           # Young's Modulus for soil; Bulk modulus for fluid
    mmpm.MatInfo[1].Viscosity = 1.e-3                                       # Angle of dilatation
    mmpm.MatInfo[1].SoundSpeed = 100                                        # Cohesion coefficient

    # MMPM body domain
    mmpm.BodyInfoSolid[0].ID = 0                                            # Body ID
    mmpm.BodyInfoSolid[0].Mat = 0                                           # Material Name of Body
    mmpm.BodyInfoSolid[0].Type = 0                                          # /Body shape/ 0 for rectangle; 1 for triangle; 2 for sphere; 3 for flow
    mmpm.BodyInfoSolid[0].MacroRho = 2600                                   # Macro density
    mmpm.BodyInfoSolid[0].pos0 = ti.Vector([0.01, 0.01])                    # Initial position or center of sphere of Body
    mmpm.BodyInfoSolid[0].len = ti.Vector([0.04, 0.06])                     # Size of Body
    mmpm.BodyInfoSolid[0].v0 = ti.Vector([0, 0])                            # Initial velocity
    mmpm.BodyInfoSolid[0].permeability = 1e-3                                    # Soild permenability
    mmpm.BodyInfoSolid[0].porosity = 0.3                                    # Solid void ratio
    mmpm.BodyInfoSolid[0].fixedV = ti.Vector([0, 0])                        # Fixed velocity
    mmpm.BodyInfoSolid[0].DT = ti.Vector([0, 1e6, 5])                       # DT

    mmpm.BodyInfoFluid[0].ID = 0                                            # Body ID
    mmpm.BodyInfoFluid[0].Mat = 1                                           # Material Name of Body
    mmpm.BodyInfoFluid[0].Type = 0                                          # /Body shape/ 0 for rectangle; 1 for triangle; 2 for sphere
    mmpm.BodyInfoFluid[0].MacroRho = 1000                                   # Macro density
    mmpm.BodyInfoFluid[0].pos0 = ti.Vector([0.01, 0.01])                    # Initial position or center of sphere of Body
    mmpm.BodyInfoFluid[0].len = ti.Vector([0.04, 0.06])                     # Size of Body
    mmpm.BodyInfoFluid[0].v0 = ti.Vector([0, 0])                            # Initial velocity
    mmpm.BodyInfoFluid[0].fixedV = ti.Vector([0, 0])                        # Fixed velocity
    mmpm.BodyInfoFluid[0].DT = ti.Vector([0, 1e6, 0])                       # DT

    # Create MPM domain
    Grid = mmpm.AddGrid()
    Material = mmpm.AddMaterial()
    Solid, Fluid = mmpm.AddParticles()
    Cell = mmpm.AddCell()
    
    # Define maximum porosity
    poros_max=0.5
    Solid.SetMaxPoros(poros_max)
    Fluid.SetMaxPoros(poros_max)

    # Define boundary conditions
    for i, j in ti.ndrange((0, 101), (4, 6)):
        Grid.SetFrictionBC(i, j, 0.5, ti.Vector([0, -1]))
    for i, j in ti.ndrange((0, 101), (40, 41)):
        Grid.SetSlippingBC(i, j, ti.Vector([0, 1]))
    for i, j in ti.ndrange((4, 6), (0, 41)):
        Grid.SetSlippingBC(i, j, ti.Vector([-1, 0]))
    for i, j in ti.ndrange((95, 97), (0, 41)):
        Grid.SetSlippingBC(i, j, ti.Vector([1, 0]))

    # Define Dirichlet boundary conditions
    
    # Solver
    TIME: float = 1                                                         # Total simulation time
    saveTime: float = 1e-2                                                  # save per time step
    CFL = 0.5                                                               # Courant-Friedrichs-Lewy condition
    vtkPath = './vtkDataTest3'                                              # VTK output path
    ascPath = './vtkDataTest3/postProcessing'                               # Monitoring data path


    # Define initial stress field
    InitialStressSolid(Solid, mmpm.BodyInfoSolid, mmpm.BodyInfoFluid, mmpm.MatInfo, mmpm.Gravity)
    InitialStressFluid(Fluid, mmpm.BodyInfoFluid, mmpm.Gravity)

    # Solve
    TimeIntegrationMMPM.Solver(mmpm, TIME, saveTime, CFL, vtkPath, ascPath, adaptive=False)f


# Test for Newtonian models using GIMP
if __name__ == "__main__":
    # MPM domain
    mpm1 = MPM.MPM(domain=ti.Vector([0.35, 0.22]),                           # domain size
                  dx=ti.Vector([0.005, 0.005]),                               # size of one grid
                  npic=4,                                                   # Number of MPM particles per subdomain
                  periodic=ti.Vector([0, 0]),                               # periodic boundary condition
                  damping=0.05,                                            # Damping ratio
                  algorithm=3,                                              # /Transfer algorithm/ 0 for USF; 1 for USL; 2 for MUSL; 3 for GIMP; 4 for APIC; 5 for polyPIC; 6 for CPIC
                  shape_func=2,                                             # /Shape Functions/ 0 for Linear; 1 for GIMP; 2 for Quadratic BSpline; 3 for Cubic BSpline; 4 for NURBS
                  gravity=ti.Vector([0, -9.8]),                             # Gravity (body force)
                  timeStep=1e-5,                                            # Time step
                  max_particle_num=1.6e4,                                     # The max number of particles
                  bodyNum=1,                                                # Taichi field of body parameters *Number of Body*
                  matNum=1,                                                 # Taichi field of material parameters *Number of Material*
                  alphaPIC=0.005,                                           # The weight of PIC velocity
                  Dynamic_Allocate=False,                                   # Sparse data structures
                  Contact_Detection=False                                   # Multibody contact detection
                  )

    # Physical parameters of particles
    mpm1.MatInfo[0].Type = 3                                                 # /Constitutive model/ 0 for Hyper-elastic; 1 for Mohr-Coulomb; 2 for Drucker-Parger; 3 for Newtonian
    mpm1.MatInfo[0].Modulus = 2e6                                            # Young's Modulus for soil; Bulk modulus for fluid
    mpm1.MatInfo[0].Viscosity = 1.e-3                                        # Angle of dilatation
    mpm1.MatInfo[0].SoundSpeed = 100                                         # Cohesion coefficient

    # MPM body domain
    mpm1.BodyInfo[0].ID = 0                                                  # Body ID
    mpm1.BodyInfo[0].Mat = 0                                                 # Material Name of Body
    mpm1.BodyInfo[0].Type = 0                                                # /Body shape/ 0 for rectangle; 1 for triangle; 2 for sphere
    mpm1.BodyInfo[0].MacroRho = 1000                                         # Macro density
    mpm1.BodyInfo[0].pos0 = ti.Vector([0.05, 0.05])                          # Initial position or center of sphere of Body
    mpm1.BodyInfo[0].len = ti.Vector([0.25, 0.1])                             # Size of Body
    mpm1.BodyInfo[0].v0 = ti.Vector([0, 0])                                  # Initial velocity
    mpm1.BodyInfo[0].fixedV = ti.Vector([0, 0])                              # Fixed velocity
    mpm1.BodyInfo[0].Darcy_k = 5e-3                                          # Soild permenability
    mpm1.BodyInfo[0].porosity = 1                                            # Solid void ratio
    mpm1.BodyInfo[0].DT = ti.Vector([0, 1e6, 0])                             # DT

    # Create MPM domain
    mpm1.AddGrid()
    mpm1.AddParticles()
    mpm1.AddMaterial()

    # Define boundary conditions
    for i, j in ti.ndrange((0, 71), (9, 11)):
        mpm1.SetSlippingBC(i, j, ti.Vector([0, -1]))
    for i, j in ti.ndrange((0, 71), (34, 36)):
        mpm1.SetSlippingBC(i, j, ti.Vector([0, 1]))
    for i, j in ti.ndrange((9, 11), (0, 41)):
        mpm1.SetSlippingBC(i, j, ti.Vector([-1, 0]))
    for i, j in ti.ndrange((60, 62), (0, 41)):
        mpm1.SetSlippingBC(i, j, ti.Vector([1, 0]))

    # Define initial stress field
    InitialStressFluid(mpm1)

    # MPM domain
    mpm2 = MPM.MPM(domain=ti.Vector([0.35, 0.22]),                           # domain size
                  dx=ti.Vector([0.005, 0.005]),                               # size of one grid
                  npic=4,                                                   # Number of MPM particles per subdomain
                  periodic=ti.Vector([0, 0]),                               # periodic boundary condition
                  damping=0.05,                                            # Damping ratio
                  algorithm=3,                                              # /Transfer algorithm/ 0 for USF; 1 for USL; 2 for MUSL; 3 for GIMP; 4 for APIC; 5 for polyPIC; 6 for CPIC
                  gravity=ti.Vector([0, -9.8]),                             # Gravity (body force)
                  timeStep=1e-5,                                            # Time step
                  max_particle_num=3.1e3,                                     # The max number of particles
                  bodyNum=1,                                                # Taichi field of body parameters *Number of Body*
                  matNum=1,                                                 # Taichi field of material parameters *Number of Material*
                  alphaPIC=0.005,                                           # The weight of PIC velocity
                  Dynamic_Allocate=False,                                   # Sparse data structures
                  Contact_Detection=False                                   # Multibody contact detection
                  )

    # Physical parameters of particles
    mpm2.MatInfo[0].Type = 2                                                 # /Constitutive model/ 0 for Hyper-elastic; 1 for Mohr-Coulomb; 2 for Drucker-Parger; 3 for Newtonian
    mpm2.MatInfo[0].Modulus = 9e6                                          # Young's Modulus for soil; Bulk modulus for fluid
    mpm2.MatInfo[0].mu = 0.3                                                 # Possion ratio
    mpm2.MatInfo[0].InternalFric = 25 / 180 * math.pi                      # Internal friction angle
    mpm2.MatInfo[0].Dilation = 0.                                            # Angle of dilatation
    mpm2.MatInfo[0].Cohesion = 0.                                           # Cohesion coefficient
    mpm2.MatInfo[0].Tensile = 0.                                             # Tensile cut-off
    mpm2.MatInfo[0].dpType = 2                                               # Drucker-Prager model

    # MPM body domain
    mpm2.BodyInfo[0].ID = 0                                                  # Body ID
    mpm2.BodyInfo[0].Mat = 0                                                 # Material Name of Body
    mpm2.BodyInfo[0].Type = 0                                                # /Body shape/ 0 for rectangle; 1 for triangle; 2 for sphere; 3 for flow
    mpm2.BodyInfo[0].MacroRho = 2500                                            # Macro density
    mpm2.BodyInfo[0].pos0 = ti.Vector([0.05, 0.05])                              # Initial position or center of sphere of Body
    mpm2.BodyInfo[0].len = ti.Vector([0.06, 0.08])                                # Size of Body
    mpm2.BodyInfo[0].v0 = ti.Vector([0, 0])                                  # Initial velocity
    mpm2.BodyInfo[0].Darcy_k = 5e-3                                         # Soild permenability
    mpm2.BodyInfo[0].porosity = 0.45                                         # Solid void ratio
    mpm2.BodyInfo[0].fixedV = ti.Vector([0, 0])                              # Fixed velocity
    mpm2.BodyInfo[0].DT = ti.Vector([0, 1e6, 5])                             # DT

    # Solver
    TIME: float = 1500e-3                                                        # Total simulation time
    saveTime: float = 1e-5                                                   # save per time step
    CFL = 0.5                                                               # Courant-Friedrichs-Lewy condition
    vtkPath = './vtkDataTest3'                                              # VTK output path
    ascPath = './vtkDataTest3/postProcessing'                               # Monitoring data path

    # Create MPM domain
    mpm2.AddGrid()
    mpm2.AddMaterial()
    mpm2.AddParticles()

    # Define Dirichlet boundary conditions
    for i, j in ti.ndrange((0, 71), (9, 11)):
        mpm2.SetFrictionBC(i, j, 0.3, ti.Vector([0, -1]))
    for i, j in ti.ndrange((0, 71), (34, 36)):
        mpm2.SetSlippingBC(i, j, ti.Vector([0, 1]))
    for i, j in ti.ndrange((9, 11), (0, 41)):
        mpm2.SetSlippingBC(i, j, ti.Vector([-1, 0]))
    for i, j in ti.ndrange((60, 62), (0, 41)):
        mpm2.SetSlippingBC(i, j, ti.Vector([1, 0]))

    # Define initial stress field
    InitialStressSolid(mpm2)

    # Solve
    TimeIntegrationMMPM.Solver(mpm1, mpm2, TIME, saveTime, CFL, vtkPath, ascPath, adaptive=False)

