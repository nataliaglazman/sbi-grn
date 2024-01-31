""" Tests for the scoring module. """

import numpy as np

from scoring import jac, ss


def test_jac():
    """ Test the Jacobian for a completely random position in state
    space and a completely random parametrisation. """
    y = np.random.random(3)
    par = random_par_dict()
    S = np.random.random()
    P = 1.0
    j = jac(y=y, par=par, S=S)

    # Diagonal elements
    assert j[0][0] == -1
    assert j[1][1] == -par["delta_Y_tilde"]
    assert j[2][2] == -par["delta_Z_tilde"]

    # Off-diagonal zero elements
    assert j[0][1] == 0
    assert j[1][2] == 0

    # Off-diagonal non-zero elements
    X, Y, Z = y

    c1 = par["K_X_P"] * P + par["c_X_PS"] * par["K_X_P"] * par["K_X_S"] * P * S
    c2 = 1 + par["K_X_S"] * S
    c3 = par["K_X_Z"]
    dphiXdZ = -c1*c2*c3*par["n"]*(c3*Z+1)**(par["n"]-1) / \
        (c1+c2*(c3*Z+1)**par["n"])**2
    assertclose(j[0][2], dphiXdZ)

    c1 = par["K_Y_P"] * P + par["c_Y_PS"] * par["K_Y_P"] * par["K_Y_S"] * P * S
    c2 = 1 + par["K_Y_S"] * S
    c3 = par["K_Y_X"]
    dphiYdX = -c1*c2*c3*par["n"]*(c3*X+1)**(par["n"]-1) / \
        (c1+c2*(c3*X+1)**par["n"])**2
    assertclose(j[1][0], dphiYdX)

    c1 = par["K_Z_P"] * P
    c2 = (1 + par["K_Z_Y"] * Y)**par["n"]
    c3 = par["K_Z_X"]
    dphiZdX = -c1*c2*c3*par["n"]*(c3*X+1)**(par["n"]-1) / \
        (c1+c2*(c3*X+1)**par["n"])**2
    assertclose(j[2][0], dphiZdX)

    c1 = par["K_Z_P"] * P
    c2 = (1 + par["K_Z_X"] * X)**par["n"]
    c3 = par["K_Z_Y"]
    dphiZdY = -c1*c2*c3*par["n"]*(c3*Y+1)**(par["n"]-1) / \
        (c1+c2*(c3*Y+1)**par["n"])**2
    assertclose(j[2][1], dphiZdY)


def test_ss():
    """ For a random parametrisation find all steady states and check
    whether they fulfill the constraints placed on them via the
    system equations at equilibrium. """
    par = random_par_dict()
    S = np.random.random()
    P = 1.0
    equils = ss(par=par, S=S)

    for equil in equils:
        X, Y, Z = equil

        # eq (1)
        c1 = par["K_X_P"] * P + par["c_X_PS"] * par["K_X_P"] * par[
            "K_X_S"] * P * S
        c2 = 1 + par["K_X_S"] * S
        c3 = par["K_X_Z"]
        assertclose(c1 / (c1 + c2 * (1 + c3 * Z)**par["n"]), X)

        # eq (2)
        c1 = par["K_Y_P"] * P + par["c_Y_PS"] * par["K_Y_P"] * par[
            "K_Y_S"] * P * S
        c2 = 1 + par["K_Y_S"] * S
        c3 = par["K_Y_X"]
        assertclose(c1 / (c1 + c2 * (1 + c3 * X)**par["n"]),
                    par["delta_Y_tilde"] * Y)

        # eq (3)
        c1 = par["K_Z_P"] * P
        c2 = par["K_Z_Y"]
        c3 = par["K_Z_X"]
        factor1 = (1 + c2 * Y)**par["n"]
        factor2 = (1 + c3 * X)**par["n"]
        assertclose(c1 / (c1 + factor1 * factor2), par["delta_Z_tilde"] * Z)


def random_par_dict():
    """ Helper for testing. Generates a random parameter dictionary. """
    names = [
        "delta_Y_tilde", "delta_Z_tilde", "K_X_P", "K_X_S", "K_X_Z", "K_Y_P",
        "K_Y_S", "K_Y_X", "K_Z_P", "K_Z_Y", "K_Z_X", "c_X_PS", "c_Y_PS", "n"
    ]
    values = np.random.uniform(0, 5, len(names))
    return dict(zip(names, values))


def assertclose(x, y, tol=1e-3):
    """ Helper for testing equality with floating point errors. """
    assert np.abs(x - y) < tol
