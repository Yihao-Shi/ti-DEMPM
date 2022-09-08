import taichi as ti
from MPMLib2D_v1.Function import *
import MPMLib2D_v1.ConsitutiveModel as cm


@ti.kernel
def GridReset(mmpm: ti.template()):
    gridList = mmpm.lg
    for ng in gridList.id:
        gridList.GridReset(ng)


@ti.func
def CalNDN(np, partList, gridList, ShapeFunction, dx, stablization):
    
    # Reset shape function (N) and gradient of shape function (DN)
    partList.ResetShapeFuncs(np)
    
    # Find min position of nodes which are influenced by this particle
    minLocalNodeId, maxLocalNodeId = gridList.FindNode(ShapeFunction, partList.x[np], dx)

    # Find nodes within the influence range
    activeID = 0
    for i in range(minLocalNodeId[0], maxLocalNodeId[0] + 1):
        for j in range(minLocalNodeId[1], maxLocalNodeId[1] + 1):
            nodeID = gridList.GetNodeID(i, j)
            partList.UpdateShapeFuncs(ShapeFunction, np, activeID, nodeID, gridList.x[nodeID], stablization)
            activeID += 1


@ti.kernel
def ParticleToGrid_Momentum(mmpm: ti.template()):
    gridList, partList = mmpm.lg, mmpm.lpf
    for np in range(partList.particleNum[None]):
        CalNDN(np, partList, gridList, mmpm.ShapeFunction, mmpm.Dx, stablization=0)
        for ln in range(partList.LnID.shape[1]):
            if partList.LnID[np, ln] >= 0:
                nodeID = partList.LnID[np, ln]
                nm = partList.Projection(np, ln, partList.m[np])
                gridList.UpdateFluidMass(nodeID, nm)
                gridList.UpdateFluidMomentumPIC(nodeID, nm * partList.v[np])


@ti.kernel
def ParticleToGrid_Force(mmpm: ti.template()):
    gridList, partList = mmpm.lg, mmpm.lpf
    for np in range(partList.particleNum[None]):
        fInt = -partList.vol[np] * partList.stress[np]
        fex = partList.m[np] * mmpm.Gravity + partList.fc[np]
        for ln in range(mmpm.lpf.LnID.shape[1]):
            if partList.LnID[np, ln] >= 0:
                nodeID = partList.LnID[np, ln]
                df = partList.Projection(np, ln, fex) + partList.ComputeInternalForce(np, ln, fInt)
                gridList.UpdateFluidForce(nodeID, df)


@ti.kernel
def GridMomentum(mmpm: ti.template()):
    gridList = mmpm.lg
    for ng in gridList.id:
        if gridList.mw[ng] > mmpm.threshold:
            gridList.ApplyGlobalDampingFluid(ng, mmpm.Damp)
            gridList.ComputeFluidMomentum(ng, mmpm.Dt[None])
            gridList.ApplyBoundaryConditionFluid(ng, mmpm.Dt[None])
            gridList.ComputeFluidVelocity(ng)


@ti.kernel
def GridToParticle(mmpm: ti.template()):
    gridList, partList = mmpm.lg, mmpm.lpf
    for np in range(partList.particleNum[None]):
        partList.LinearPICFLIPFluid(np, gridList, mmpm.alphaPIC, mmpm.Dt[None])


@ti.kernel
def UpdateStressStrain(mmpm: ti.template()):
    gridList, partList, matList, cellList = mmpm.lg, mmpm.lpf, mmpm.lm, mmpm.cell
    for np in range(partList.particleNum[None]):
        partList.CalLocalDvFluid(gridList, np)
        partList.UpdateDeformationGrad(np, mmpm.Dt[None], matList, cellList, stablization=0, mode=0)
        partList.UpdatePartPropertiesFluid(np)
        partList.UpdateStrainFluid(np, gridList)
        cm.CalStress(np, ti.Matrix.identity(float, 2), ti.Matrix.identity(float, 2), partList, matList)
        partList.ResetParticleForce(np)

