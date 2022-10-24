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
        partList.ResetParticleForce(np)


@ti.kernel
def ParticleToGrid_Momentum(mpm: ti.template()):
    gridList, partList = mpm.lg, mpm.lp
    for np in range(partList.particleNum[None]):
        CalNDN(mpm, np)
        for ln in range(mpm.lp.LnID.shape[1]):
            if partList.LnID[np, ln] >= 0:
                nodeID = partList.LnID[np, ln]
                nm = partList.Projection(np, ln, partList.m[np])
                gridList.UpdateNodalMass(nodeID, nm)
                gridList.UpdateNodalMomentumAPIC(nodeID, nm * partList.v[np])


@ti.kernel
def ParticleToGridAPIC(mpm: ti.template()):
    gridList, partList = mpm.lg, mpm.lp
    for np in range(partList.particleNum[None]):
        fInt = -partList.vol[np] * partList.stress[np] * 4 * mpm.inv_Dx[0] * mpm.inv_Dx[1]
        fex = partList.m[np] * mpm.Gravity + partList.fc[np] + partList.fd[np]
        for ln in range(partList.LnID.shape[1]):
            if partList.LnID[np, ln] >= 0:
                nodeID = partList.LnID[np, ln]
                SF = partList.LnShape[np, ln]
                dpos = gridList.x[nodeID] - partList.x[np]
                gridList.mv[nodeID] += SF * partList.m[np] * (partList.v[np] + partList.gradv[np] @ dpos)
                gridList.m[nodeID] += SF * partList.m[np]
                gridList.f[nodeID] += SF * (fex + fInt @ dpos)


@ti.kernel
def GridOperationAPIC(mpm: ti.template()):
    gridList = mpm.lg
    for ng in mpm.lg.m:
        if gridList.m[ng] > mpm.threshold:  
            gridList.ApplyGlobalDamping(ng, mpm.Damp)
            gridList.ComputeNodalMomentum(ng, mpm.Dt[None])
            gridList.ApplyBoundaryCondition(ng, mpm.Dt[None])
            gridList.ComputeNodalVelocity(ng)


@ti.kernel
def GridToParticleAPIC(mpm: ti.template()):
    gridList, partList = mpm.lg, mpm.lp
    inv_dx = 2. / mpm.Dx.sum()
    for np in range(partList.particleNum[None]):  
        vAPIC, vFLIP = ti.Matrix.zero(float, 2), partList.v[np]
        pos = partList.x[np]
        new_C = ti.Matrix.zero(float, 2, 2)
        for ln in range(partList.LnID.shape[1]):
            if partList.LnID[np, ln] >= 0:  
                nodeID = partList.LnID[np, ln]
                SF = partList.LnShape[np, ln]
                dpos = (gridList.x[nodeID] - partList.x[np]) * mpm.inv_Dx
                vAPIC += SF * gridList.v[nodeID] 
                vFLIP += SF * gridList.f[nodeID] / gridList.m[nodeID] * mpm.Dt[None]
                new_C += 4 * inv_dx * SF * gridList.v[nodeID].outer_product(dpos)
                pos += SF * gridList.v[nodeID] * mpm.Dt[None]
        new_v = mpm.alphaPIC * vAPIC + (1 - mpm.alphaPIC) * vFLIP
        partList.v[np] = new_v * Zero2OneVector(partList.fixVel[np]) + partList.v0[np] * partList.fixVel[np]
        partList.x[np] = pos * Zero2OneVector(partList.fixVel[np]) + partList.v0[np] * mpm.Dt[None] * partList.fixVel[np]
        partList.gradv[np] = new_C
        partList.fc[np], partList.fd[np] = ti.Matrix.zero(float, 2), ti.Matrix.zero(float, 2)

