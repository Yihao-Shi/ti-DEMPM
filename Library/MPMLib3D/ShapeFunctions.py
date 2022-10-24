import taichi as ti
from MPMLib3D_v1.Function import *


# ---------------------------------------------- GIMP ------------------------------------------------ #
# ========================================================= #
#                                                           #
#                  GIMP shape function                      #
#                                                           #
# ========================================================= #
@ti.func
def ShapeGIMP(xp, xg, dx, lp):
    nx = 0.
    d = xp - xg
    a = dx + lp
    b = dx * lp
    if d < a:
        if d > (dx - lp):
            nx = (a - d) * (a - d) / (4. * b)
        elif d > lp:
            nx = 1. - d / dx
        elif d > -lp:
            nx = 1. - (d * d + lp * lp) / (2. * b)
        elif d > (-dx + lp):
            nx = 1. + d / dx
        elif d > -a:
            nx = (a + d) * (a + d) / (4. * b)
    return nx


@ti.func
def GShapeGIMP(xp, xg, dx, lp):
    dnx = 0.
    d = xp - xg
    a = dx + lp
    b = dx * lp
    if d < a:
        if d > (dx - lp):
            dnx = (d - a) / (2. * b)
        elif d > lp:
            dnx = -1. / dx
        elif d > -lp:
            dnx = -d / b
        elif d > (-dx + lp):
            dnx = 1. / dx
        elif d > -a:
            dnx = (a + d) / (2. * b)
    return dnx


@ti.func
def ShapeGIMPCenter(xp, xg, dx, lp):
    nxc = 0.
    d = xp - xg
    a = dx + lp
    b = dx * lp
    if d < a:
        if d > (dx - lp):
            nxc = (a - d) / (4. * lp)
        elif d > (-dx + lp):
            nxc = 1./2.
        elif d > -a:
            nxc = (a + d) / (4. * lp)
    return nxc


# Used for tranditional GIMP
@ti.func
def GIMP1(xp, xg, dx, lp):
    SF0, SF1, SF2 = ShapeGIMP(xp[0], xg[0], dx[0], lp[0]), ShapeGIMP(xp[1], xg[1], dx[1], lp[1]), ShapeGIMP(xp[2], xg[2], dx[2], lp[2])
    GS0, GS1, GS2 = GShapeGIMP(xp[0], xg[0], dx[0], lp[0]), GShapeGIMP(xp[1], xg[1], dx[1], lp[1]), GShapeGIMP(xp[2], xg[2], dx[2], lp[2])

    SF = SF0 * SF1 * SF2
    GS = ti.Vector([GS0 * SF1 * SF2, SF0 * GS1 * SF2, SF0 * SF1 * GS2]).cast(float)
    return SF, GS


# Used for Anti-Locking GIMP (B-Bar)
@ti.func
def GIMP2(xp, xg, dx, lp):
    SF0, SF1, SF2 = ShapeGIMP(xp[0], xg[0], dx[0], lp[0]), ShapeGIMP(xp[1], xg[1], dx[1], lp[1]), ShapeGIMP(xp[2], xg[2], dx[2], lp[2])
    GS0, GS1, GS2 = GShapeGIMP(xp[0], xg[0], dx[0], lp[0]), GShapeGIMP(xp[1], xg[1], dx[1], lp[1]), GShapeGIMP(xp[2], xg[2], dx[2], lp[2])
    SFC0, SFC1, SFC2 = ShapeGIMPCenter(xp[0], xg[0], dx[0], lp[0]), ShapeGIMPCenter(xp[1], xg[1], dx[1], lp[1]), ShapeGIMPCenter(xp[2], xg[2], dx[2], lp[2])

    SF = SF0 * SF1 * SF2
    GS = ti.Vector([GS0 * SF1 * SF2, SF0 * GS1 * SF2, SF0 * SF1 * GS2]).cast(float)
    GSC = ti.Vector([GS0 * SFC1 * SFC2, SFC0 * GS1 * SFC2, SFC0 * SFC1 * GS2]).cast(float)
    return SF, GS, GSC


# Used for Anti-Locking GIMP (Fluid)
@ti.func
def GIMP3(xp, xg, dx, lp):
    SF0, SF1, SF2 = ShapeGIMP(xp[0], xg[0], dx[0], lp[0]), ShapeGIMP(xp[1], xg[1], dx[1], lp[1]), ShapeGIMP(xp[2], xg[2], dx[2], lp[2])
    GS0, GS1, GS2 = GShapeGIMP(xp[0], xg[0], dx[0], lp[0]), GShapeGIMP(xp[1], xg[1], dx[1], lp[1]), GShapeGIMP(xp[2], xg[2], dx[2], lp[2])
    SFC0, SFC1, SFC2 = ShapeGIMPCenter(xp[0], xg[0], dx[0], lp[0]), ShapeGIMPCenter(xp[1], xg[1], dx[1], lp[1]), ShapeGIMPCenter(xp[2], xg[2], dx[2], lp[2])

    SF = SF0 * SF1 * SF2
    GSC = ti.Vector([GS0 * SFC1 * SFC2, SFC0 * GS1 * SFC2, SFC0 * SFC1 * GS2]).cast(float)
    return SF, GSC


# ------------------------------------------- B-Spline ---------------------------------------------- #
# ========================================================= #
#                                                           #
#                  Linear shape function                    #
#                                                           #
# ========================================================= #
@ti.func
def ShapeLinear(xp, xg, dx):
    nx = 0.
    d = ti.abs(xp - xg) / dx
    if d < 1.:
        nx = 1. - d
    return nx


@ti.func
def GShapeLinear(xp, xg, dx):
    dnx = 0.
    d = ti.abs(xp - xg) / dx
    if d < 1.:
        dnx = -sign(xp - xg) / dx
    return dnx


@ti.func
def ShapeLinearCenter():
    return 1./2.


# Used for Classical MPM
@ti.func
def Linear1(xp, xg, dx):
    SF0, SF1, SF2 = ShapeLinear(xp[0], xg[0], dx[0]), ShapeLinear(xp[1], xg[1], dx[1]), ShapeLinear(xp[2], xg[2], dx[2])
    GS0, GS1, GS2 = GShapeLinear(xp[0], xg[0], dx[0]), GShapeLinear(xp[1], xg[1], dx[1]), GShapeLinear(xp[2], xg[2], dx[2])

    SF = SF0 * SF1 * SF2
    GS = ti.Vector([GS0 * SF1 * SF2, SF0 * GS1 * SF2, SF0 * SF1 * GS2]).cast(float)
    return SF, GS


# Used for Anti-Locking Classical MPM (B-Bar)
@ti.func
def Linear2(xp, xg, dx):
    SF0, SF1, SF2 = ShapeLinear(xp[0], xg[0], dx[0]), ShapeLinear(xp[1], xg[1], dx[1]), ShapeLinear(xp[2], xg[2], dx[2])
    GS0, GS1, GS2 = GShapeLinear(xp[0], xg[0], dx[0]), GShapeLinear(xp[1], xg[1], dx[1]), GShapeLinear(xp[2], xg[2], dx[2])
    SFC0, SFC1, SFC2 = ShapeLinearCenter(), ShapeLinearCenter(), ShapeLinearCenter()

    SF = SF0 * SF1 * SF2
    GS = ti.Vector([GS0 * SF1 * SF2, SF0 * GS1 * SF2, SF0 * SF1 * GS2]).cast(float)
    GSC = ti.Vector([GS0 * SFC1 * SFC2, SFC0 * GS1 * SFC2, SFC0 * SFC1 * GS2]).cast(float)
    return SF, GS, GSC


# Used for Anti-Locking Classical MPM (Fluid)
@ti.func
def Linear3(xp, xg, dx):
    SF0, SF1, SF2 = ShapeLinear(xp[0], xg[0], dx[0]), ShapeLinear(xp[1], xg[1], dx[1]), ShapeLinear(xp[2], xg[2], dx[2])
    GS0, GS1, GS2 = GShapeLinear(xp[0], xg[0], dx[0]), GShapeLinear(xp[1], xg[1], dx[1]), GShapeLinear(xp[2], xg[2], dx[2])
    SFC0, SFC1, SFC2 = ShapeLinearCenter(), ShapeLinearCenter(), ShapeLinearCenter()

    SF = SF0 * SF1 * SF2
    GSC = ti.Vector([GS0 * SFC1 * SFC2, SFC0 * GS1 * SFC2, SFC0 * SFC1 * GS2]).cast(float)
    return SF, GS, GSC


# ========================================================= #
#                                                           #
#           Quadratic B-spline shape function               #
#                                                           #
# ========================================================= #
@ti.func
def ShapeBsplineQ(xp, xg, dx):
    nx = 0.
    d = ti.abs(xp - xg) / dx
    if d < 1.5:
        if d > 0.5:
            nx = 0.5 * (1.5 - d) * (1.5 - d)
        else:
            nx = 0.75 - d * d
    return nx


@ti.func
def GShapeBsplineQ(xp, xg, dx):
    dnx = 0.
    d = ti.abs(xp - xg) / dx
    a = sign(xp - xg)
    if d < 1.5:
        if d > 0.5:
            dnx = (d - 1.5) * a / dx
        else:
            dnx = -2 * d * a  / dx
    return dnx


@ti.func
def BsplineQ(x, xg, dx, domain):
    LeftBound = 0
    RightBound = int(domain / dx)
    Loc = int(x / dx)
    SFx, GSx = 0., 0.
    '''if Loc == LeftBound:
        SFx, GSx = ShapeBsplineQ02(x, xg, dx), GShapeBsplineQ02(x, xg, dx)
    elif Loc == LeftBound + 1:
        SFx, GSx = ShapeBsplineQ12(x, xg, dx), GShapeBsplineQ12(x, xg, dx)
    elif Loc == RightBound - 1:
        SFx, GSx = ShapeBsplineQ32(x, xg, dx), GShapeBsplineQ32(x, xg, dx)
    elif Loc == RightBound:
        SFx, GSx = ShapeBsplineQ42(x, xg, dx), GShapeBsplineQ42(x, xg, dx)
    else:'''
    SFx, GSx = ShapeBsplineQ(x, xg, dx), GShapeBsplineQ(x, xg, dx)
    return SFx, GSx


@ti.func
def BsplineQ1(xp, xg, dx, domain):
    SF0, GS0 = BsplineQ(xp[0], xg[0], dx[0], domain[0])
    SF1, GS1 = BsplineQ(xp[1], xg[1], dx[1], domain[1])
    SF2, GS2 = BsplineQ(xp[2], xg[2], dx[2], domain[2])

    SF = SF0 * SF1 * SF2
    GS = ti.Vector([GS0 * SF1 * SF2, SF0 * GS1 * SF2, SF0 * SF1 * GS2]).cast(float)
    return SF, GS


@ti.func
def BsplineQ2(xp, xg, dx, domain):
    return 0., ti.Matrix.zero(float, 3), ti.Matrix.zero(float, 3)


# ========================================================= #
#                                                           #
#             Cubic B-spline shape function                 #
#                                                           #
# ========================================================= #
@ti.func
def ShapeBsplineC(xp, xg, dx):
    nx = 0.
    d = ti.abs(xp - xg) / dx
    if d < 2:
        if d > 1:
            nx = (2 - d) ** 3 / 6
        else:
            nx = 0.5 * d ** 3 - d ** 2 + 2. / 3.
    return nx


@ti.func
def GShapeBsplineC(xp, xg, dx):
    dnx = 0.
    d = ti.abs(xp - xg) / dx
    a = sign(xp - xg)
    if d < 2:
        if d > 1:
            dnx = -0.5 * (2 - d) ** 2 * a / dx
        else:
            dnx = (1.5 * d ** 2 - 2 * d) * a / dx
    return dnx


@ti.func
def BsplineC1(xp, xg, dx, domain):
    SF0, SF1, SF2 = ShapeBsplineC(xp[0], xg[0], dx[0]), ShapeBsplineC(xp[1], xg[1], dx[1]), ShapeBsplineC(xp[2], xg[2], dx[2])
    GS0, GS1, GS2 = GShapeBsplineC(xp[0], xg[0], dx[0]), GShapeBsplineC(xp[1], xg[1], dx[1]), GShapeBsplineC(xp[2], xg[2], dx[2])

    SF = SF0 * SF1 * SF2
    GS = ti.Vector([GS0 * SF1 * SF2, SF0 * GS1 * SF2, SF0 * SF1 * GS2]).cast(float)
    return SF, GS


@ti.func
def BsplineC2(xp, xg, dx, domain):
    return 0., ti.Matrix.zero(float, 3), ti.Matrix.zero(float, 3)


# ========================================================= #
#                                                           #
#           Quadratic B-spline shape function               #
#                                                           #
# ========================================================= #
@ti.func
def BSplineRecursion(knot, deg, cpos, index):
    val = 0.
    if deg == 0:
        if knor_coord[index] <= cpos < knor_coord[index + 1]:
            val = 1.
        else:
            cal = 0.
    else:
        a = (cpos - knor_coord[index]) / (knor_coord[index + deg] - knor_coord[index])
        b = (knor_coord[index + deg + 1] - cpos) / (knor_coord[index + deg + 1] - knor_coord[index + 1])
        val = a * BSpline(knot, deg - 1, cpos, index) + b * BSpline(knot, deg, cpos, index + 1)

    return val


# ========================================================= #
#                                                           #
#          Non-Uniform B-spline shape function              #
#                                                           #
# ========================================================= #
@ti.func
def NURBSRecursion(knot, deg, cpos, index):
    pass


# ---------------------------------------- Course Grain --------------------------------------------- #
@ti.func
def KernelFunc():
    pass
