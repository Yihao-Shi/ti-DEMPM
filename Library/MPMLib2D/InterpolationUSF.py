import taichi as ti
from Common.Function import *
import MPMLib2D.ConsitutiveModel as cm


@ti.kernel
def GridReset(mpm: ti.template()):
    gridList = mpm.lg
    for ng in gridList.id:
        gridList.GridReset(ng)


@ti.func
def CalNDN(mpm, np):
    gridList, partList = mpm.lg, mpm.lp
    
    # Reset shape function (N) and gradient of shape function (DN)
    partList.ResetShapeFuncs(np)
    
    # Find min position of nodes which are influenced by this particle
    minLocalNodeId, maxLocalNodeId = gridList.FindNode(mpm.ShapeFunction, partList.x[np], mpm.Dx)

    # Find nodes within the influence range
    activeID = 0
    for i in range(minLocalNodeId[0], maxLocalNodeId[0] + 1):
        for j in range(minLocalNodeId[1], maxLocalNodeId[1] + 1):
            nodeID = gridList.GetNodeID(i, j)
            partList.UpdateShapeFuncs(mpm.ShapeFunction, np, activeID, nodeID, gridList.x[nodeID], mpm.Stablization)
            activeID += 1


@ti.kernel
def ParticleToGrid_Momentum(mpm: ti.template()):
    gridList, partList = mpm.lg, mpm.lp
    for np in range(partList.particleNum[None]):
        CalNDN(mpm, np)
        for ln in range(partList.LnID.shape[1]):
            if partList.LnID[np, ln] >= 0:
                nodeID = partList.LnID[np, ln]
                nm = partList.Projection(np, ln, partList.m[np])
                ve = partList.ComputeParticleVel(np, gridList.x[nodeID])
                gridList.UpdateNodalMass(nodeID, nm)
                gridList.UpdateNodalMomentumPIC(nodeID, nm * ve)


@ti.kernel
def GridVelocity(mpm: ti.template()):
    gridList = mpm.lg
    for ng in gridList.id:
        if gridList.m[ng] > mpm.threshold:
            gridList.ApplyBoundaryCondition(ng, mpm.Dt[None])
            gridList.ComputeNodalVelocity(ng)


@ti.kernel
def UpdateStressStrain(mpm: ti.template()):
    gridList, partList, matList, cellList = mpm.lg, mpm.lp, mpm.lm, mpm.cell
    for nc in range(cellList.cellSum):
        cellList.CellReset(nc)

    for np in range(partList.particleNum[None]):
        partList.CalLocalDv(gridList, np)
        partList.UpdateDeformationGrad(np, mpm.Dt[None], matList, cellList, mpm.Stablization, mode=0)

    for np in range(partList.particleNum[None]):
        partList.UpdateDeformationGrad(np, mpm.Dt[None], matList, cellList, mpm.Stablization, mode=1)
    
    for nc in range(cellList.cellSum):
        cellList.ComputeCellJacobian(nc)

    for np in range(partList.particleNum[None]):
        partList.UpdateDeformationGrad(np, mpm.Dt[None], matList, cellList, mpm.Stablization, mode=2)

    for np in range(partList.particleNum[None]):
        partList.UpdatePartProperties(np)
        de, dw = partList.UpdateStrain(np, mpm.Dt[None], gridList)
        cm.CalStress(np, de, dw, partList, matList)


@ti.kernel
def ParticleToGrid_Force(mpm: ti.template()):
    gridList, partList = mpm.lg, mpm.lp
    for np in range(partList.particleNum[None]):
        # Particle Internal force and external force
        fInt = -partList.vol[np] * partList.stress[np]
        fex = partList.m[np] * mpm.Gravity + partList.fc[np]
        for ln in range(partList.LnID.shape[1]):
            if partList.LnID[np, ln] >= 0:
                nodeID = partList.LnID[np, ln]
                df = partList.Projection(np, ln, fex) + partList.ComputeInternalForce(np, ln, fInt)
                gridList.UpdateNodalForce(nodeID, df)


@ti.kernel
def GridMomentum(mpm: ti.template()):
    gridList = mpm.lg
    for ng in gridList.id:
        if gridList.m[ng] > mpm.threshold:
            gridList.ApplyGlobalDamping(ng, mpm.Damp)
            gridList.ComputeNodalMomentum(ng, mpm.Dt[None])
            gridList.ApplyBoundaryCondition(ng, mpm.Dt[None])
            gridList.ComputeNodalVelocity(ng)


@ti.kernel
def GridToParticle(mpm: ti.template()):
    gridList, partList = mpm.lg, mpm.lp
    for np in range(partList.particleNum[None]):
        partList.LinearPICFLIP(np, gridList, mpm.alphaPIC, mpm.Dt[None])
        partList.ResetParticleForce(np)
