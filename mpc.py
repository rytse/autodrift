import torch
from torchdiffeq import odeint

from mpc_step import step_physics

CENTERLINE_REWARD = 1.0
SPEED_REWARD = 0.1
DRIFT_REWARD = 0.3

L_STATECOST = torch.diag(
    torch.tensor([CENTERLINE_REWARD, -SPEED_REWARD, 0, -DRIFT_REWARD, 0, 0, 0])
)


def riccatti_diff(K, A, B, L):
    return K @ B @ B.T @ K - A.T @ K - K @ A - L

def get_jacobians(state_cur, action_last, track_curv, gear):
    J_f_x = torch.autograd.functional.jacobian(
        lambda state: step_physics(state, action_last, track_curv, gear),
        state_cur,
        create_graph=True,
    )
    J_f_u = torch.autograd.functional.jacobian(
        lambda action: step_physics(state_cur, action, 0.2, 2),
        action_last,
        create_graph=True,
    )

    return J_f_x, J_f_u


def optimize_next_frame(
    state_cur, action_last, track_curv, gear, mpc_horizon=1.0, mpc_sol_steps=10_000
):

    J_f_x, J_f_u = get_jacobians(state_cur, action_last, track_curv, gear)
    # Check controllability
    C = torch.concat([J_f_u,
                     J_f_x @ J_f_u,
                     J_f_x @ J_f_x @ J_f_u,
                     J_f_x @ J_f_x @ J_f_x @ J_f_u,
                     J_f_x @ J_f_x @ J_f_x @ J_f_x @ J_f_u,
                     J_f_x @ J_f_x @ J_f_x @ J_f_x @ J_f_x @ J_f_u,
                     J_f_x @ J_f_x @ J_f_x @ J_f_x @ J_f_x @ J_f_x @ J_f_u]
                 )
    print("C.shape")
    print(C.shape)
    print("\n\nrank of ctrlb")
    print(torch.matrix_rank(C))
    print("\n\n")


    print("J_f_x")
    print(J_f_x)
    print("J_f_u")
    print(J_f_u)

    t_sol = torch.linspace(mpc_horizon, 0.0, mpc_sol_steps)
    K_traj = odeint(
        lambda t, K: riccatti_diff(K, J_f_x, J_f_u, L_STATECOST),
        torch.zeros(7, 7),
        t_sol,
    )

    action_cur = -J_f_u.T @ K_traj[0, :, :] @ state_cur
    return action_cur


if __name__ == "__main__":
    state_cur = torch.tensor([-9.6022e+00,  1.4705e+01,
        -2.7156e-02,
        -1.2177e-02,
        -1.2591e-02, -1.6055e-03,
        0.0000e+00])

    ''' 
    state_cur = torch.tensor(
        [0.0,  # d
         0.1,  # v
         0.0,  # delta 
         0.1,  # beta
         0.1,  # psi
         0.1,  # w_z
         0.0,  # sigma
         0.0   # t
    ], requires_grad=True
    )
    '''
    action_last = torch.tensor([0.0, 0.0, 0.8], requires_grad=True)

    K_traj = optimize_next_frame(state_cur, action_last, 0.1, 2)
    print(K_traj)
