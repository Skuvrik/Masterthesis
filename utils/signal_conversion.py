import numpy as np

def SPGR_s2r(S, TR, FA):
    FA = FA*(np.pi/180)  # degree -> rad
    TR = TR / 1000  # ms -> s
    T_10 = 1.4
    S_0 = np.mean(S[:5])
    S = S-S_0
    E_10 = np.exp(-TR/T_10)
    B = (1-E_10)/(1-E_10*np.cos(FA))
    A = np.nan_to_num(B*S/S_0)

    C = np.log((1-A)/(1-A*np.cos(FA)))
    C[C > 0] = -C[C > 0]

    R1 = np.nan_to_num(-1/TR * C)
    return R1

def SR_s2r(S):
    R_10 = 1/1.4
    S_0 = np.mean(S[:5])
    R1 = R_10*(S/S_0 - 1)
    return R1

def NLC_SR_s2r(S, TD):
    TD = TD / 1000
    R_10 = 1/1.4
    S_0 = np.mean(S[:5])
    E_0 = 1-np.exp(-TD*R_10)
    A = 1-E_0*(S/S_0)
    R = (-1/TD) * np.log(A) - R_10
    return R

def pass_through(x):
    return x