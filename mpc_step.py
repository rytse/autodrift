import numpy as np

# Car general parameters
M = 1.0  # mass
G = 9.81
L_F = 1.0
L_R = 1.0
E_SP = 1.0
R = 1.0
I_ZZ = 1.0
C_W = 1.0
RHO = 1.0
A = 1.0
I_T = 1.0
I_G_U = 1.0

# Car slip parameters
B_F = 1.0
B_R = 1.0
C_F = 1.0
C_R = 1.0
D_F = 1.0
D_R = 1.0
E_F = 1.0
E_R = 1.0

# Dummy function for looking up track curvature
def track_curvature(sigma):
    return 1.0


# Engine torque helper functions
def f_1(phi):
    return 1 - np.exp(-3.0 * phi)


def f_2(w_mot):
    return -37.8 + 1.54 * w_mot - 0.0019 * w_mot ** 2


def f_3(w_mot):
    return -34.9 - 0.04775 * w_mot


# ODE Update Step
def step_physics(state, ctrl, track_curv):
    """
    """
    # Unpack state
    # c_x = state[0]
    # c_y = state[1]
    d = state[0]
    v = state[1]
    delta = state[2]
    beta = state[3]
    psi = state[4]
    w_z = state[5]
    t = state[6]
    sigma = state[7]

    # Unpack control
    w_delta = ctrl[0]
    F_B = ctrl[1]
    phi = ctrl[2]
    mu = ctrl[3]

    # New dynamics for d, t
    d_d = np.sin(psi - track_curv - beta)
    d_t = 1.0 / v

    # Yaw angle
    d_psi = w_z

    # Slip angle
    alpha_f = delta - np.arctan((L_F * d_psi - v * np.sin(beta)) / (v * np.cos(beta)))
    alpha_r = delta - np.arctan((L_R * d_psi + v * np.sin(beta)) / (v * np.cos(beta)))

    # Side lateral forces
    F_sf = D_F * np.sin(
        C_F
        * np.arctan(B_F * alpha_f - E_F * (B_F * alpha_f - np.arctan(B_F * alpha_f)))
    )
    F_sr = D_R * np.sin(
        C_R
        * np.arctan(B_R * alpha_r - E_R * (B_R * alpha_r - np.arctan(B_R * alpha_r)))
    )

    # Friction
    f_R = 9e-3 + 7.2e-5 * v + 5.038848e-10 * np.power(v, 4)
    F_Ax = 0.5 * C_W * RHO * A * v ** 2
    F_Ay = 0.0

    # Breaking force
    F_Bf = 2.0 / 3.0 * F_B
    F_Br = 1.0 / 3.0 * F_B
    F_Rf = f_R * M * L_R * G / (L_F + L_R)
    F_Rr = f_R * M * L_F * G / (L_F + L_R)
    F_lf = -F_Bf - F_Rf

    # Engine torque
    w_mot_u = I_G_U * I_T / R * v
    M_mot_u = f_1(phi) * f_2(w_mot_u) + (1.0 - f_1(phi)) * f_3(w_mot_u)

    # Longitudinal force
    F_lr_u = I_G_U * I_T / R * M_mot_u - F_Br - F_Rr

    # Regular dynamics
    d_v = (
        1.0
        / M
        * (
            (F_lr_u - F_Ax) * np.cos(beta)
            + F_lf * np.cos(delta + beta)
            - (F_sr - F_Ay) * np.sin(beta)
            - F_sf * np.sin(delta + beta)
        )
    )
    d_delta = w_delta
    d_beta = w_z - 1.0 / M / v * (
        (F_lr - F_Ax) * np.sin(beta)
        + F_lf * np.sin(delta + beta)
        + (F_sr - F_Ay) * np.cos(beta)
        + F_sf * np.cos(delta + beta)
    )
    d_w_z = (
        1.0
        / I_ZZ
        * (
            F_sf * L_F * np.cos(delta)
            - F_sr * LR
            - F_Ay * E_SP
            + F_lf * L_F * np.sin(delta)
        )
    )

    # Pack derivatives
    d_state = np.array([d_d, d_v, d_delta, d_beta, d_psi, d_w_z, d_t])  # note no d_sigma

    return d_state

def ode_step
