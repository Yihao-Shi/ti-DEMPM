from __init__ import *
import DEMLib3D.DEM as DEM
import math
ti.init(arch=ti.cpu, default_fp=ti.f32, debug=False)


# Test for Linear contact models // Friction
if __name__ == "__main__":
    # DEM domain
    dem = DEM.DEM(cmType=0,                                                 # /Contact model type/ 0 for linear model; 1 for hertz model; 2 for linear rolling model; 3 for linear bond model
                  domain=ti.Vector([10, 5, 5]),                             # domain size
                  boundary=ti.Vector([0, 0, 0]),                            # periodic boundary condition
                  algorithm=1,                                              # /Integration scheme/ 0 for Euler; 1 for Verlet; 2 for sympletic
                  gravity=ti.Vector([0., 0., 0.]),                        # Gravity (body force)
                  timeStep=1.e-3,                                           # Time step
                  bodyNum=2,                                               # Taichi field of body parameters *Number of Body*
                  matNum=1,                                                 # Taichi field of material parameters *Number of Material*
                  wallNum=0,                                                # Taichi field of material parameters *Number of Wall*
                  searchAlgorithm=1,                                        # /Search Algorithm/ 0 for sorted based; 1 for linked cell;  
                  )
    
    # Physical parameters of particles
    dem.MatInfo[0].Kn = 1e4                                                 # Contact normal stiffness
    dem.MatInfo[0].Ks = 1e4                                                 # Contact tangential stiffness
    dem.MatInfo[0].Mu = 0.5                                                 # Friction coefficient
    dem.MatInfo[0].ForceLocalDamping = 0.                                   # /Local damping/ 
    dem.MatInfo[0].TorqueLocalDamping = 0.                                  # /Local damping/ 
    dem.MatInfo[0].NormalViscousDamping = 0.2                               # /Viscous damping/ 
    dem.MatInfo[0].TangViscousDamping = 0.                                 # /Viscous damping/ 
    dem.MatInfo[0].ParticleRho = 2650                                       # Particle density

    # DEM body domain
    dem.BodyInfo[0].ID = 0                                                  # Body ID
    dem.BodyInfo[0].Mat = 0                                                 # Material Name of Body
    dem.BodyInfo[0].shapeType = 0                                           # /ShapeType of DEM_PARTICLE/ 0 for sphere; 1 for SDEM; 
    dem.BodyInfo[0].GenerateType = 0                                        # /Generate type/ 0 for create; 1 for generate; 2 for fill in box; 3 for wall
    dem.BodyInfo[0].pos0 = pos = ti.Vector([5, 2.2, 2.025])                                              # Initial position or center of sphere of Body
    dem.BodyInfo[0].rhi = 0.025                                                 # Particle radius
    dem.BodyInfo[0].rlo = 0.025                                                 # Particle radius
    dem.BodyInfo[0].pnum = 1                                                # The number of particle in the group
    dem.BodyInfo[0].v0 = ti.Vector([0, 1, 0])                              # Initial velocity
    dem.BodyInfo[0].w0 = ti.Vector([0, 0, 0])                               # Initial angular velocity
    dem.BodyInfo[0].fixedV = ti.Vector([0, 0, 0])                           # Fixed velocity
    dem.BodyInfo[0].fixedW = ti.Vector([1, 1, 1])                           # Fixed angular velocity
    dem.BodyInfo[0].orientation = ti.Vector([0, 0, 1])                               # Initial orientation
    dem.BodyInfo[0].DT = ti.Vector([0, 1e6, 5])                             # DT
    
    dem.BodyInfo[1].ID = 1                                                  # Body ID
    dem.BodyInfo[1].Mat = 0                                                 # Material Name of Body
    dem.BodyInfo[1].shapeType = 0                                           # /ShapeType of DEM_PARTICLE/ 0 for sphere; 1 for SDEM; 
    dem.BodyInfo[1].GenerateType = 0                                        # /Generate type/ 0 for create; 1 for generate; 2 for fill in box; 3 for wall
    dem.BodyInfo[1].pos0 = pos = ti.Vector([5, 2.3, 2.025])                                              # Initial position or center of sphere of Body
    dem.BodyInfo[1].rhi = 0.025                                                 # Particle radius
    dem.BodyInfo[1].rlo = 0.025                                                 # Particle radius
    dem.BodyInfo[1].pnum = 1                                                # The number of particle in the group
    dem.BodyInfo[1].v0 = ti.Vector([0, 0, 0])                              # Initial velocity
    dem.BodyInfo[1].w0 = ti.Vector([0, 0, 0])                               # Initial angular velocity
    dem.BodyInfo[1].fixedV = ti.Vector([1, 1, 1])                           # Fixed velocity
    dem.BodyInfo[1].fixedW = ti.Vector([1, 1, 1])                           # Fixed angular velocity
    dem.BodyInfo[1].orientation = ti.Vector([0, 0, 1])                               # Initial orientation
    dem.BodyInfo[1].DT = ti.Vector([0, 1e6, 5])                             # DT

    # Create MPM domain
    dem.AddContactModel()
    dem.AddBodies(max_particle_num=2)
    dem.AddWall(max_facet_num=1)
    dem.AddContactPair(max_contact_num=1)
    dem.AddNeighborList(multiplier=4, max_potential_particle_pairs=1, max_potential_wall_pairs=1)

    # Step1
    TIME: float = 0.2                                                         # Total simulation time
    saveTime: float = 0.002                                                   # save per time step
    CFL = 0.5                                                               # Courant-Friedrichs-Lewy condition
    vtkPath = './vtkDataTest0'                                              # VTK output path
    ascPath = './vtkDataTest0/postProcessing'                               # Monitoring data path

    dem.Solver(TIME, saveTime, CFL, vtkPath, ascPath, adaptive=False)

