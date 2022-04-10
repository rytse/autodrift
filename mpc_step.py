import time
import torch
import torch.autograd

# Car general parameters
M = 1.239e3
G = 9.81
L_F = 1.19016
L_R = 1.37484
E_SP = 1.0  # has no effect
R = 0.302
I_ZZ = 1.752e3
C_W = 0.3
RHO = 1.249512
A = 1.4378946874
I_T = 3.91
I_G_U_DICT = {1: 3.09, 2: 2.002, 3: 1.33, 4: 1.0, 5: 0.805}
I_G_U_AVG = 1.65  # this is an average, it changes every time you shift
B_F = 1.096e1
B_R = 1.267e1
C_F = 1.3
C_R = 1.3
D_F = 4.5604e3
D_R = 3.94781e3
E_F = -0.5
E_R = -0.5

# Dummy function for looking up track curvature
def track_curvature(sigma):
    return 1.0


# Engine torque helper functions
def f_1(phi):
    return 1 - torch.exp(-3.0 * phi)


def f_2(w_mot):
    return -37.8 + 1.54 * w_mot - 0.0019 * w_mot ** 2


def f_3(w_mot):
    return -34.9 - 0.04775 * w_mot


def boring_step(state, ctrl, track_curv, gear):
    psi = state[4]
    beta = state[3]
    out = torch.sin(psi - track_curv - beta)
    return out


# ODE Update Step
def step_physics(state, ctrl, track_curv, gear):
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

    # New dynamics for d, t
    # d_d = Variable(torch.sin(state[4] - track_curv - state[3]), requires_grad=True)
    d_d = torch.sin(psi - track_curv - beta)
    d_t = 1.0 / v

    # Yaw angle
    d_psi = w_z

    # Slip angle
    alpha_f = delta - torch.arctan(
        (L_F * d_psi - v * torch.sin(beta)) / (v * torch.cos(beta))
    )
    alpha_r = delta - torch.arctan(
        (L_R * d_psi + v * torch.sin(beta)) / (v * torch.cos(beta))
    )

    # Side lateral forces
    F_sf = D_F * torch.sin(
        C_F
        * torch.arctan(
            B_F * alpha_f - E_F * (B_F * alpha_f - torch.arctan(B_F * alpha_f))
        )
    )
    F_sr = D_R * torch.sin(
        C_R
        * torch.arctan(
            B_R * alpha_r - E_R * (B_R * alpha_r - torch.arctan(B_R * alpha_r))
        )
    )

    # Friction
    f_R = 9e-3 + 7.2e-5 * v + 5.038848e-10 * torch.pow(v, 4)
    F_Ax = 0.5 * C_W * RHO * A * v ** 2
    F_Ay = 0.0

    # Breaking force
    F_Bf = 2.0 / 3.0 * F_B
    F_Br = 1.0 / 3.0 * F_B
    F_Rf = f_R * M * L_R * G / (L_F + L_R)
    F_Rr = f_R * M * L_F * G / (L_F + L_R)
    F_lf = -F_Bf - F_Rf

    # Engine torque
    i_g_u = I_G_U_DICT[gear]
    w_mot_u = i_g_u * I_T / R * v
    M_mot_u = f_1(phi) * f_2(w_mot_u) + (1.0 - f_1(phi)) * f_3(w_mot_u)

    # Longitudinal force
    F_lr_u = i_g_u * I_T / R * M_mot_u - F_Br - F_Rr

    # Regular dynamics
    d_v = (
        1.0
        / M
        * (
            (F_lr_u - F_Ax) * torch.cos(beta)
            + F_lf * torch.cos(delta + beta)
            - (F_sr - F_Ay) * torch.sin(beta)
            - F_sf * torch.sin(delta + beta)
        )
    )
    d_delta = w_delta
    d_beta = w_z - 1.0 / M / v * (
        (F_lr_u - F_Ax) * torch.sin(beta)  # TODO confirm F_lr_u
        + F_lf * torch.sin(delta + beta)
        + (F_sr - F_Ay) * torch.cos(beta)
        + F_sf * torch.cos(delta + beta)
    )
    d_w_z = (
        1.0
        / I_ZZ
        * (
            F_sf * L_F * torch.cos(delta)
            - F_sr * L_R
            - F_Ay * E_SP
            + F_lf * L_F * torch.sin(delta)
        )
    )

    # Pack derivatives
    d_sigma = torch.tensor(1.0)
    d_state = torch.stack(
        [d_d, d_v, d_delta, d_beta, d_psi, d_w_z, d_t, d_sigma], dim=0
    )  # note no d_sigma

    return d_state


if __name__ == "__main__":
    test_state = torch.tensor(
        [5.0, 50.0, 1.0, 0.1, 0.5, 0.2, 0.1, 0.1], requires_grad=True
    )
    test_action = torch.tensor([0.2, 0.0, 1.0], requires_grad=True)
    zero_state = torch.tensor(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True
    )
    zero_action = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
    track_curv = 0.3
    gear = 2

    # bs = boring_step(test_state, test_action, 3.0, 2)
    bs = step_physics(test_state, test_action, 0.2, 2)
    J_f_x_v = torch.autograd.grad(
        bs,
        test_state,
        grad_outputs=torch.ones_like(bs),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )
    J_f_u_v = torch.autograd.grad(
        bs,
        test_action,
        grad_outputs=torch.ones_like(bs),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )

    J_f_x = torch.autograd.functional.jacobian(
        lambda state: step_physics(state, zero_action, 0.2, 2),
        test_state,
        create_graph=True,
    )
    J_f_u = torch.autograd.functional.jacobian(
        lambda action: step_physics(test_state, action, 0.2, 2),
        test_action,
        create_graph=True,
    )

    print("bs.shape")
    print(bs.shape)
    print("test_state.shape")
    print(test_state.shape)
    print("J_f_x")
    print(J_f_x)
    print("J_f_u")
    print(J_f_u)
