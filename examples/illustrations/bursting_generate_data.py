from scipy.optimize import minimize, Bounds
import mpi4py.MPI as mpi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from pypacmensl.ssa.ssa import SSASolver

np.random.seed(mpi.COMM_WORLD.Get_rank())
# %% Experiment design
n_cells = 500
t_meas = np.linspace(0, 10, 5)
# %% Define model structure
stoich_mat = np.array([[-1, 1, 0],
                       [1, -1, 0],
                       [0, 0, 1],
                       [0, 0, -1]])
x0 = np.array([[1, 0, 0]])
p0 = np.array([1.0])
s0 = np.array([0.0])
constr_init = np.array([1, 1, 100])

k_off = 0.5
k_on = 0.1
k_r = 50.0
gamma = 1.0
theta_true = np.array([k_off, k_on, k_r, gamma])


def propensity(reaction, x, out):
    if reaction is 0:
        out[:] = x[:, 0]
    if reaction is 1:
        out[:] = x[:, 1]
    if reaction is 2:
        out[:] = x[:, 1]
    if reaction is 3:
        out[:] = x[:, 2]


def t_fun_factory(theta):
    def t_fun(t, out):
        out[:] = theta[:]

    return t_fun


def dt_fun_factory(theta):
    dt_list = []

    def d_t_fun(i):
        def d_t_(t, out):
            out[i] = 1.0

        return d_t_

    for i in range(0, 4):
        dt_list.append(d_t_fun(i))
    return dt_list


# %% Simulate a data set

def simulate_data():
    ssa = SSASolver(mpi.COMM_WORLD)
    ssa.SetModel(stoich_mat, t_fun_factory(theta_true), propensity)
    observations = []
    for t in t_meas:
        observations.append(ssa.Solve(t, x0, n_cells, send_to_root=True))
    return observations


observations = simulate_data()

if mpi.COMM_WORLD.Get_rank() == 0:
    np.savez('bursting_simulated_data', observations)


