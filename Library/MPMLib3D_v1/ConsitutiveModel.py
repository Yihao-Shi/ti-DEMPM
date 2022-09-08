import taichi as ti
from MPMLib2D_v1.Function import *


@ti.func
def Sigrot(np, dw, partList):
    partList.stress[np] += (dw @ partList.stress[np] - partList.stress[np] @ dw)


@ti.func
def Elastic_devi(np, de, partList, matList, matID):
    partList.stress[np] += 2 * matList.g[matID] * (de - de.trace() * ti.Matrix.identity(float, 2) / 2.)


@ti.func
def Elastic_p(np, de, partList, matList, matID):
    partList.stress[np] += matList.la[matID] * de.trace() * ti.Matrix.identity(float, 2)


@ti.func
def StressDecomposed(np, partList):
    sigma = partList.stress[np].trace() / 2.
    sd = partList.stress[np] - sigma * ti.Matrix.identity(float, 2)
    return sigma, sd


@ti.func
def EquivalentStress(sd):
    J2 = 0.5 * (sd * sd).sum()
    seqv = ti.sqrt(3 * J2)
    return seqv


# Constitutive Model
@ti.func
def Elastic(np, de, dw, partList, matList, matID):                                          # Hyper-Elastic
    Sigrot(np, dw, partList)
    Elastic_devi(np, de, partList, matList, matID)
    Elastic_p(np, de, partList, matList, matID)


@ti.func
def VonMises(np, de, dw, partList, matList, matID):                                 # J2 Flow Rule
    Sigrot(np, dw, partList)
    Elastic_devi(np, de, partList, matList, matID)
    sigma, sd = StressDecomposed(np, partList)
    seqv = EquivalentStress(sd)
    if seqv > matList.yield0[matID]:
        partList.plasticStrainEff[np] += (seqv - matList.yield0[matID]) / (3 * matList.g[matID] + matList.Ep[matID])
        ratio = matList.yield0[matID] / seqv
        sd *= ratio
        seqv *= ratio
        partList.stress[np] = sd + sigma * ti.Matrix.identity(float, 2)


@ti.func
def IsotropicHardening():
    pass


@ti.func
def MohrCoulomb(np, de, dw, partList, matList, matID):
    # !-- trial elastic stresses ----!
    Sigrot(np, dw, partList)
    Elastic_devi(np, de, partList, matList, matID)
    Elastic_p(np, de, partList, matList, matID)
    sigma, sd = StressDecomposed(np, partList)
    seqv = EquivalentStress(sd)
    J2 = (seqv / ti.sqrt(3)) ** 2
    J3 = sd.determinant()
    arg = -3*ti.sqrt(3) * J3 / (2 * J2 * ti.sqrt(J2))
    lode = -1/3 * ti.asin(arg)

    #Implicit method
    theta = 1.0
    epsilon = 1e-12

    sin_lode, cos_lode, sign_lode = ti.sin(lode), ti.cos(lode), sign(lode)
    sin_lodeT, cos_lodeT = ti.sin(matList.lodeT[np]), ti.cos(matList.lodeT[np])
    sin_3lodeT, cos_3lodeT = ti.sin(3 * matList.lodeT[np]), ti.cos(3 * matList.lodeT[np])
    sin_6lodeT, cos_6lodeT = ti.sin(6 * matList.lodeT[np]), ti.cos(6 * matList.lodeT[np])
    sin_fai, cos_fai, sin_psi = ti.sin(matList.fai[np]), ti.cos(matList.fai[np]), ti.sin(matList.psi[np])

    K = 0.
    if ti.abs(lode) < matList.lodeT[np]:
        K = cos_lode - ti.sqrt(1./3.) * sin_fai * sin_lode
    else:
        term1 = cos_lodeT - ti.sqrt(1./3.) * sin_fai * sign_lode
        term2 = sign_lode * sin_lodeT + ti.sqrt(1./3.) * sin_fai * cos_lodeT
        term3 = 18 * cos_3lodeT ** 3
        B = (sign_lode * sin_6lodeT * term1 - 6 * cos_6lodeT * term2) / term3
        C = (-cos_3lodeT * term1 - 3 * sign_lode * sin_3lodeT * term2) / term3
        A = -ti.sqrt(1./3.) * sin_fai * sign_lode * sin_lodeT - B * sign_lode * sin_3lodeT - C * sin_3lodeT ** 2 + cos_lodeT
        K = A + B * arg + C * arg ** 2
    sMC = sigma / 3 * sin_fai + ti.sqrt(J2 * K * K + (matList.tensile[np] * sin_fai) ** 2)
    f = sMC - matList.c[np] * cos_fai
    if f > 0.:
        pass
    '''U, sig, V = ti.svd(partList.stress[np])
    s1, s2 = sig[0, 0], sig[1, 1]
    sin0 = ti.sin(partList.phi[np])
    cos0 = ti.cos(partList.phi[np])
    sin1 = ti.sin(partList.psi[np])
    f = s1 + s1 * sin0 - 2. * partList.c[np] * cos0

    sc = ti.Matrix.zero(float, 2)
    if f > 1.e-18:
        sin01 = sin0 * sin1
        qA0 = (8. * partList.g[np] / 3. - 4. * partList.k[np]) * sin01
        qA1 = partList.g[np] * (1. + sin0) * (1. + sin1)
        qA2 = partList.g[np] * (1. - sin0) * (1. - sin1)
        qB0 = 2. * partList.c[np] * cos0

        gsl = 0.5 * (s1 - s2) / (partList.g[np] * (1. + sin1))
        gsr = 0.5 * s2 / (partList.g[np] * (1. - sin1))
        gla = 0.5 * (s1 + s2) / (partList.g[np] * (3. - sin1))
        gra = 0.5 * (2. * s1 - s2) / (partList.g[np] * (3. + sin1))

        qsA = qA0 - 4. * partList.g[np] * (1. + sin01)
        qsB = f
        qlA = qA0 - qA1 - 2. * qA2
        qlB = 0.5 * (1. + sin0) * (s1 + s2) - qB0
        qrA = qA0 - 2. * qA1 - qA2
        qrB = (1. + sin0) * s1 - 0.5 * (1. - sin0) * s2 - qB0
        qaA = -4. * partList.k[np] * sin01
        qaB = 2. * (s1 + s2) / 3. * sin0 - qB0

        minslsr = ti.min(gsl, gsr)
        maxlara = ti.max(gla, gra)

        if minslsr > 0 and qsA * minslsr + qsB < 0:
            dl = -qsB / qsA
            ds0 = -dl * (2. * partList.k[np] - 4. * partList.g[np] / 3.) * sin1
            sc[0] = s1 + ds0 - dl * (2. * partList.g[np] * (1. + sin1))
            sc[1] = s2 + ds0
        elif 0 < gsl <= gla and qlA * gsl + qlB >= 0 and qlA * gla + qlB <= 0:
            dl = -qlB / qlA
            ds0 = dl * (4. * partList.g[np] / 3. - 2. * partList.k[np]) * sin1
            sc[0] = sc[1] = 0.5 * (s1 + s2) + ds0 - dl * partList.g[np] * (1. + sin1)
        elif 0 < gsr <= gra and qrA * gsr + qrB >= 0 and qrA * gra + qrB <= 0:
            dl = -qrB / qrA
            ds0 = dl * (4. * partList.g[np] / 3. - 2. * partList.k[np]) * sin1
            sc[0] = s1 + ds0 - 2. * dl * partList.g[np] * (1. + sin1)
            sc[1] = 0.5 * s2 + ds0 + dl * partList.g[np] * (1.-sin1)
        elif maxlara > 0 and qaA * maxlara + qaB >= -1.e-24:
            sc[0] = sc[1] = partList.c[np] / ti.tan(partList.phi[np])
        sp = ti.Matrix([[sc[0], 0], [0, sc[1]]])
        partList.stress[np] = U @ sp @ U.transpose()'''


@ti.func
def DruckerPrager(np, de, dw, partList, matList, matID):
    # !-- trial elastic stresses ----!
    Sigrot(np, dw, partList)
    Elastic_devi(np, de, partList, matList, matID)
    Elastic_p(np, de, partList, matList, matID)
    sigma, sd = StressDecomposed(np, partList)
    seqv = EquivalentStress(sd)
    J2sqrt = seqv / ti.sqrt(3)
    dpFi = J2sqrt + matList.q_fai[matID] * sigma - matList.k_fai[matID]
    dpsig = sigma - matList.tensile[matID]

    if dpsig < 0.:
        if dpFi > 0.:
            dlamd = dpFi / (matList.g[matID] + matList.k[matID] * matList.q_fai[matID] * matList.q_psi[matID])
            sigma -= matList.k[matID] * matList.q_psi[matID] * dlamd
            ratio = (matList.k_fai[matID] - matList.q_fai[matID] * sigma) / J2sqrt
            sd *= ratio
            seqv *= ratio
            partList.stress[np] = sd + sigma * ti.Matrix.identity(float, 2)
            partList.plasticStrainEff[np] += dlamd * ti.sqrt(1./3. + (2./9.) * matList.q_psi[matID] ** 2)
    else:
        alphap = ti.sqrt(1 + matList.q_fai[matID] ** 2) - matList.q_fai[matID]
        J2sqrtp = matList.k_fai[matID] - matList.q_fai[matID] - matList.tensile[matID]
        dp_hfai = J2sqrt - J2sqrtp - alphap * dpsig

        if dp_hfai > 0.:
            dlamd = dpFi / (matList.g[matID] + matList.k[matID] * matList.q_fai[matID] * matList.q_psi[matID])
            sigma -= matList.k[matID] * matList.q_psi[matID] * dlamd
            ratio = (matList.k_fai[matID] - matList.q_fai[matID] * sigma) / J2sqrt
            sd *= ratio
            seqv *= ratio
            partList.stress[np] = sd + sigma * ti.Matrix.identity(float, 2)
            partList.plasticStrainEff[np] += dlamd * ti.sqrt(1./3. + (2./9.) * matList.q_psi[matID] ** 2)
        else:
            dlamd = (sigma - matList.tensile[matID]) / matList.k[matID]
            partList.stress[np] += (matList.tensile[matID] - sigma) * ti.Matrix.identity(float, 2)
            partList.plasticStrainEff[np] += dlamd * (1./3.) * ti.sqrt(2)
    '''# !-- trial elastic stresses ----!
    Sigrot(np, dw, partList)
    Elastic_devi(np, de, partList, matList, matID)
    Elastic_p(np, de, partList, matList, matID)
    sigma, sd = StressDecomposed(np, partList)
    seqv = EquivalentStress(sd)
    J2sqrt = seqv / ti.sqrt(3)
    dpFi = J2sqrt + matList.q_fai[matID] * sigma - matList.k_fai[matID]
    if dpFi > 0:
        if matList.q_fai[matID] * (sigma - J2sqrt / matList.g[matID] * matList.k[matID] * matList.q_psi[matID]) - matList.k_fai[matID] < 0:
            dlamd = dpFi / (matList.g[matID] + matList.q_fai[matID] * matList.k[matID] * matList.q_psi[matID])
            sigma -= matList.k[matID] * matList.q_psi[matID]
            sd -= matList.g[matID] / J2sqrt * sd
            partList.stress[np] += dlamd * (sd + sigma * ti.Matrix.identity(float, 2))
            partList.plasticStrainEff[np] += dlamd * ti.sqrt(1./3. + (2./9.) * matList.q_psi[matID] ** 2)
        else:
            partList.stress[np] = matList.k_fai[matID] / matList.q_fai[matID] * ti.Matrix.identity(float, 2)'''


@ti.func
def Newtonian(np, de, partList, matList, matID):
    partList.stress[np] = partList.P[np] * ti.Matrix.identity(float, 2) \
                          + 2 * matList.vis[matID] * (partList.StrainRate[np] - partList.StrainRate[np].trace() * ti.Matrix.identity(float, 2) / 2.)


@ti.func
def BinghamModel(np, de, partList, matList, matID):
    dnorm = ti.sqrt(2 * (partList.StrainRate[np] * partList.StrainRate[np]).sum())
    partList.stress[np] = partList.P[np] * ti.Matrix.identity(float, 2) \
                          + 2 * (matList.vis[matID] + partList.yieldStress[matID] / dnorm * (1 - ti.exp(-m * dnorm))) * partList.StrainRate[np]


@ti.func
def EOSMonaghan(np, partList, matList, matID):
    gamma = 7
    partList.P[np] = matList.k[matID] * ((partList.vol[np] / partList.vol0[np]) ** gamma - 1)


@ti.func
def Granular(np, de, partList, matList, matID):
    dnorm = ti.sqrt(2 * (partList.StrainRate[np] * partList.StrainRate[np]).sum())
    I = matList.diamp[matID] * dnorm / ti.sqrt(partList.P[np] / matList.rhop[matID])
    miu = matList.miu_s[matID] + (matList.miu_2[matID] - matList.miu_s[matID]) / (1. + matList.I0[matID] / I)
    if dnorm < 1e-12:
        matList.vis[matID] = miu * partList.P[np]
    else:
        matList.vis[matID] = miu * partList.P[np] / dnorm
    partList.stress[np] = - partList.P[np] * ti.Matrix.identity(float, 2) \
                          + 2 * matList.vis[matID] * (de - de.trace() * ti.Matrix.identity(float, 2) / 2.)


@ti.func
def ElasticViscoplastic(np, de, dw, partList, matList, matID):
    # !-- trial elastic stresses ----!
    Sigrot(np, dw, partList)
    Elastic_devi(np, de, partList, matList, matID)
    Elastic_p(np, de, partList, matList, matID)
    sigma, sd = StressDecomposed(np, partList)
    I1, p = 2 * sigma, ti.max(-sigma, 1e-5)
    seqv = EquivalentStress(sd)
    J2sqrt = seqv / ti.sqrt(3)

    if J2sqrt + matList.miu_s[matID] * p > 0:
        dnorm = de.determinant()
        I = 2. * matList.diamp[matID] * dnorm / ti.sqrt(p / matList.rhop[matID])
        miu = matList.miu_s[matID] + (matList.miu_2[matID] - matList.miu_s[matID]) / (1. + matList.I0[matID] / I)
        dpFi = J2sqrt + matList.q_fai[matID] * sigma 
        if dpFi < 0:
            sd = miu * p * partList.StrainRate[np] / dnorm
            partList.stress[np] = sigma * ti.Matrix.identity(float, 2) + sd
        if dpFi > 0:
            kapa = (matList.miu_2[matID] - matList.miu_s[matID]) * matList.I0[matID] \
                   / (3. * (matList.I0[matID] + I) ** 2 * ti.sqrt(p * matList.rhop[matID]) * dnorm * matList.diamp[matID])
            dlamd = (miu * matList.k[matID] * de.trace() + matList.k[matID] * kapa * I1 * de.trace() + (matList.g[matID] / J2sqrt) * (de * sd).sum()) \
                   / (miu ** 2 * matList.k[matID] + 2 * kapa * I1 * miu +  kapa ** 2 * I1 ** 2 + matList.g[matID])
            partList.stress[np] -= dlamd * (miu * matList.k[matID] * ti.Matrix.identity(float, 2) + (matList.g[matID] / J2sqrt) * sd + I1 * matList.k[matID] * kapa * ti.Matrix.identity(float, 2))
            sigma, sd = StressDecomposed(np, partList)
            I1, p = 2 * sigma, ti.max(-sigma, 1e-5)
            seqv = EquivalentStress(sd)
            J2sqrt = seqv / ti.sqrt(3)
            if miu * sigma < 0:
                partList.stress[np] -= sigma * ti.Matrix.identity(float, 2) 
            elif J2sqrt + miu * sigma > 0:
                r = -miu * sigma / J2sqrt
                partList.stress[np] = r * sd + sigma * ti.Matrix.identity(float, 2)


@ti.func
def NULLModel(np):
    partList.stress[np] = ti.Matrix.zero(float, 2, 2)


@ti.func
def CalStress(np, de, dw, partList, matList):
    matID = partList.materialID[np]
    if matList.matType[matID] == 0:
        Elastic(np, de, dw, partList, matList, matID)
    elif matList.matType[matID] == 1:
        MohrCoulomb(np, de, dw, partList, matList, matID)
    elif matList.matType[matID] == 2:
        DruckerPrager(np, de, dw, partList, matList, matID)
    elif matList.matType[matID] == 3:
        EOSMonaghan(np, partList, matList, matID)
        Newtonian(np, de, partList, matList, matID)
    elif matList.matType[matID] == 4:
        EOSMonaghan(np, partList, matList, matID)
        Granular(np, de, partList, matList, matID)
    elif matList.matType[matID] == 5:
        ElasticViscoplastic(np, de, dw, partList, matList, matID)
