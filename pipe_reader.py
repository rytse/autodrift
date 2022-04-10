import os
import sys

import torch
import numpy as np

import mpc

read_path = "/tmp/pipe_sim2mpc"
write_path = "/tmp/pipe_mpc2sim"

state_cur = torch.zeros(8, dtype=torch.float32)
actions_last = torch.zeros(3, dtype=torch.float32)
actions_cur = torch.zeros(3, dtype=torch.float32)

while True:
    with open(read_path, "rb") as read_fifo:
        data = read_fifo.read()
        if len(data) == 0:
            continue
        state = torch.tensor(np.frombuffer(data, dtype=np.float32))
        state_cur[:-2] = state

        try:
            actions_cur = mpc.optimize_next_frame(state_cur, actions_last, 0.1, 3)
            print("state_cur")
            print(state_cur)
            print("actions_cur")
            print(actions_cur)
            print("\n\n")
        except Exception as e:
            actions_cur[0] = 0.0
            actions_cur[1] = 0.0
            actions_cur[2] = 0.0
        
        print(state_cur)
        with open(write_path, "wb") as write_fifo:
            write_fifo.write(actions_cur.detach().numpy().tobytes())

        actions_last = actions_cur
