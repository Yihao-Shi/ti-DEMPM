import taichi as ti
from Common.Function import *
import MPMLib2D.ConsitutiveModel as cm


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
def ParticleToGridSolid(mmpm: ti.template()):
    gridList, SolidPhase, LiquidPhase = mmpm.lg, mmpm.lps, mmpm.lpf
    for np in range(SolidPhase.particleNum[None]):
        CalNDN(np, SolidPhase, gridList, mmpm.Dx, stablization=2)
        for ln in range(SolidPhase.LnID.shape[1]):
            if SolidPhase.LnID[np, ln] >= 0:
                nodeID = SolidPhase.LnID[np, ln]
                nm = SolidPhase.Projection(np, ln, SolidPhase.m[np])
                gridList.UpdateNodalMass(nodeID, nm)
                gridList.UpdateNodalMomentumPIC(nodeID, nm * SolidPhase.v[np])


@ti.kernel
def GridInfoUpdateSolid(mmpm: ti.template()):
    gridList = mmpm.lg
    for ng in gridList.id:
        if gridList.m[ng] > mmpm.threshold:
            gridList.ApplyBoundaryCondition(ng, mmpm.Dt[None])
            gridList.ComputeNodalVelocity(ng)
                

@ti.kernel
def ParticleToGridHydro(mmpm: ti.template()):
    gridList, SolidPhase, LiquidPhase = mmpm.lg, mmpm.lps, mmpm.lpf
    for np in range(SolidPhase.particleNum[None]):
        for ln in range(SolidPhase.LnID.shape[1]):
            if SolidPhase.LnID[np, ln] >= 0:
                nodeID = SolidPhase.LnID[np, ln]
                nm = SolidPhase.Projection(np, ln, SolidPhase.m[np])
                gridList.UpdateNodalPorosity(nodeID, nm * SolidPhase.poros[np])
                gridList.UpdateNodalPermeability(nodeID, nm * SolidPhase.kh[np])


@ti.kernel
def GridInfoUpdateHydro(mmpm: ti.template()):
    gridList = mmpm.lg
    for ng in gridList.id:
        if gridList.m[ng] > mmpm.threshold:
            gridList.ComputeNodalPorosity(ng)
            gridList.ComputeNodalPermeability(ng)


@ti.kernel
def GridToParticle_Porosity(mmpm: ti.template()):
    gridList, LiquidPhase = mmpm.lg, mmpm.lpf
    for np in range(LiquidPhase.particleNum[None]):
        CalNDN(np, LiquidPhase, gridList, mmpm.Dx, stablization=0)
        LiquidPhase.PorosityTransfer(np, gridList)


@ti.kernel
def FluidPointInit(mmpm: ti.template()):
    LiquidPhase = mmpm.lpf
    for np in range(LiquidPhase.particleNum[None]):
        LiquidPhase.UpdatePartPropertiesFluid(np)


@ti.kernel
def ParticleToGridFuild(mmpm: ti.template()):
    gridList, LiquidPhase = mmpm.lg, mmpm.lpf
    for np in range(LiquidPhase.particleNum[None]):
        for ln in range(LiquidPhase.LnID.shape[1]):
            if LiquidPhase.LnID[np, ln] >= 0:
                nodeID = LiquidPhase.LnID[np, ln]
                nm = LiquidPhase.Projection(np, ln, LiquidPhase.m[np])
                gridList.UpdateFluidMass(nodeID, nm)
                gridList.UpdateFluidMomentumPIC(nodeID, nm * LiquidPhase.v[np])


@ti.kernel
def GridInfoUpdateFluid(mmpm: ti.template()):
    gridList = mmpm.lg
    for ng in gridList.id:
        if gridList.mw[ng] > mmpm.threshold:
            gridList.ApplyBoundaryCondition(ng, mmpm.Dt[None])
            gridList.ComputeFluidVelocity(ng)


@ti.kernel
def ParticleToGridFluidForce(mmpm: ti.template()):
    gridList, LiquidPhase = mmpm.lg, mmpm.lpf
    for np in range(LiquidPhase.particleNum[None]):
        fInt = LiquidPhase.vol[np] * LiquidPhase.stress[np]
        fex = LiquidPhase.m[np] * mmpm.Gravity + LiquidPhase.fc[np]
        porosity = LiquidPhase.poros[np]
        for ln in range(LiquidPhase.LnID.shape[1]):
            if LiquidPhase.LnID[np, ln] >= 0:
                nodeID = LiquidPhase.LnID[np, ln]
                df = LiquidPhase.Projection(np, ln, fex) + LiquidPhase.ComputeInternalForce(np, ln, fInt * porosity)
                dfInt = LiquidPhase.ComputeInternalForce(np, ln, fInt)
                gridList.StorePorePressure(nodeID, dfInt)
                gridList.UpdateFluidForce(nodeID, df)


@ti.kernel
def ParticleToGridMultiplier(mmpm: ti.template()):
    gridList, LiquidPhase = mmpm.lg, mmpm.lpf
    for np in range(LiquidPhase.particleNum[None]):
        for ln in range(LiquidPhase.LnID.shape[1]):
            if LiquidPhase.LnID[np, ln] >= 0:
                nodeID = LiquidPhase.LnID[np, ln]
                perm = LiquidPhase.ComputePermeability(np, mmpm.Gravity)
                multiplier = LiquidPhase.Projection(np, ln, perm)
                gridList.UpdateMultiplier(nodeID, perm)


@ti.kernel
def GridDragForce(mmpm: ti.template()):
    gridList = mmpm.lg
    for ng in gridList.id:
        if gridList.kh[ng] > mmpm.threshold:
            gridList.ComputeDragForce(ng)


@ti.kernel
def ParticleToGridSolidForce(mmpm: ti.template()):
    gridList, SolidPhase = mmpm.lg, mmpm.lps
    for np in range(SolidPhase.particleNum[None]):
        fInt = -SolidPhase.vol[np] * SolidPhase.stress[np]
        fex = SolidPhase.m[np] * mmpm.Gravity + SolidPhase.fc[np]
        for ln in range(SolidPhase.LnID.shape[1]):
            if SolidPhase.LnID[np, ln] >= 0:
                nodeID = SolidPhase.LnID[np, ln]
                df = SolidPhase.Projection(np, ln, fex) + SolidPhase.ComputeInternalForce(np, ln, fInt)
                gridList.UpdateNodalForce(nodeID, df)


@ti.kernel
def GridAccerlation(mmpm: ti.template()):
    gridList = mmpm.lg
    for ng in gridList.id:
        if gridList.mw[ng] > mmpm.threshold:
            gridList.FluidForceAssemble(ng)

    for ng in gridList.id:
        if gridList.m[ng] > mmpm.threshold:
            gridList.SolidForceAssemble(ng, mmpm.Gravity)
       

@ti.kernel
def GridMomentumSolid(mmpm: ti.template()):
    gridList = mmpm.lg
    for ng in gridList.id:
        if gridList.m[ng] > mmpm.threshold:
            gridList.ApplyGlobalDamping(ng, mmpm.Damp)
            gridList.ComputeNodalMomentum(ng, mmpm.Dt[None])
            gridList.ApplyBoundaryCondition(ng, mmpm.Dt[None])
            gridList.ComputeNodalVelocity(ng) 


@ti.kernel
def GridMomentumFluid(mmpm: ti.template()):
    gridList = mmpm.lg 
    for ng in gridList.id:
        if gridList.mw[ng] > mmpm.threshold:   
            gridList.ApplyGlobalDampingFluid(ng, mmpm.Damp)
            gridList.ComputeFluidMomentum(ng, mmpm.Dt[None])
            gridList.ApplyBoundaryConditionFluid(ng, mmpm.Dt[None])      
            gridList.ComputeFluidVelocity(ng)


@ti.kernel
def GridToParticleSolid(mmpm: ti.template()):
    gridList, SolidPhase = mmpm.lg, mmpm.lps
    for np in range(SolidPhase.particleNum[None]):
        SolidPhase.LinearPICFLIP(np, gridList, mmpm.alphaPIC, mmpm.Dt[None])
        SolidPhase.ResetParticleForce(np)


@ti.kernel
def GridToParticleFluid(mmpm: ti.template()):
    gridList, LiquidPhase = mmpm.lg, mmpm.lpf
    for np in range(LiquidPhase.particleNum[None]):
        LiquidPhase.LinearPICFLIPFluid(np, gridList, mmpm.alphaPIC, mmpm.Dt[None])
        LiquidPhase.ResetParticleForce(np)


@ti.kernel
def UpdateStressStrainSolid(mmpm: ti.template()):
    gridList, SolidPhase, matList, cellList = mmpm.lg, mmpm.lps, mmpm.lm, mmpm.cell
    for np in range(SolidPhase.particleNum[None]):
        SolidPhase.CalLocalDv(gridList, np)
        SolidPhase.UpdateDeformationGrad(np, mmpm.Dt[None], matList, cellList, stablization=0, mode=0)
        SolidPhase.UpdatePartProperties(np)
        
        if SolidPhase.poros[np] < SolidPhase.poros_max:
            de, dw = SolidPhase.UpdateStrain(np, mmpm.Dt[None], gridList)
            cm.CalStress(np, de, dw, SolidPhase, matList)
        SolidPhase.ResetParticleForce(np)


@ti.kernel
def UpdateStressStrainFluid(mmpm: ti.template()):
    gridList, LiquidPhase, matList, cellList = mmpm.lg, mmpm.lpf, mmpm.lm, mmpm.cell
    for np in range(LiquidPhase.particleNum[None]):
        LiquidPhase.CalLocalDvFluid(gridList, np)
        LiquidPhase.UpdateDeformationGrad(np, mmpm.Dt[None], matList, cellList, stablization=0, mode=0)

        LiquidPhase.UpdateStrainFluid(np, gridList)
        LiquidPhase.UpdateSolidStrainFluid(np, gridList)
        LiquidPhase.ComputePorePressure(np, mmpm.Dt[None], mmpm.threshold, matList)
        LiquidPhase.ResetParticleForce(np)


@ti.kernel
def CellAverageTech(mmpm: ti.template()):
    LiquidPhase, cellList = mmpm.lpf, mmpm.cell
    for cell in range(cellList.cellSum):
        cellList.CellReset(cell)

    for np in range(LiquidPhase.particleNum[None]):
        cellID = LiquidPhase.cellID[np]
        pvol = LiquidPhase.P[np] * LiquidPhase.vol[np]
        cellList.UpdateCellVolume(cellID, LiquidPhase.vol[np])
        cellList.UpdatePorePressure(cellID, pvol)

    for cell in range(cellList.cellSum):
        cellList.AveragePorePressure(cell)

    for np in range(LiquidPhase.particleNum[None]):
        cellID = LiquidPhase.cellID[np]
        P = cellList.P[cellID]
        LiquidPhase.ReComputePorePressure(np, P)
