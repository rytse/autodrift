import torch
from torchdiffeq import odeint

from mpc_step import step_physics

CENTERLINE_REWARD = 1.0
SPEED_REWARD = 0.1
DRIFT_REWARD = 0.3

L_STATECOST = torch.diag(
    torch.tensor([CENTERLINE_REWARD, -SPEED_REWARD, 0, -DRIFT_REWARD, 0, 0, 0, 0])
)


def riccatti_diff(K, A, B, L):
    return K @ B @ B.T @ K - A.T @ K - K @ A - L
    """
    return (
        torch.mm(torch.mm(torch.mm(K, B), B.T), K)
        - torch.mm(K, A)
        - torch.mm(A.T, K)
        - L
    )
    """


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
    state_cur, action_last, track_curv, gear, mpc_horizon=1.0, mpc_sol_steps=1000
):
    J_f_x, J_f_u = get_jacobians(state_cur, action_last, track_curv, gear)

    print("J_f_x: ")
    print(torch.max(torch.abs(J_f_x)))
    print("J_f_u: ")
    print(torch.max(torch.abs(J_f_u)))

    t_sol = torch.linspace(mpc_horizon, 0.0, mpc_sol_steps)
    K_traj = odeint(
        lambda t, K: riccatti_diff(K, J_f_x, J_f_u, L_STATECOST),
        torch.zeros(8, 8),
        t_sol,
    )

    print("K_traj: ")
    print(K_traj.shape)
    return K_traj


if __name__ == "__main__":
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
    action_last = torch.tensor([0.0, 0.0, 0.8], requires_grad=True)

    K_traj = optimize_next_frame(state_cur, action_last, 0.1, 2)
