# autodrift

Model predictive control for autonomously drifting cars simulated in [VDrift](https://vdrift.net/). Best open-source hack at Bitcamp 2022.

Current solver linearizes a car model at each step and solves a LQR problem for the short-time optimal control. Plans for eventually implmenting direct multiple shooting.

Communicates with [this fork](https://github.com/rytse/vdrift/tree/autodrift) of VDrift over pipe.
