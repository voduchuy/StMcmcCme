import numpy as np
from scipy.io import loadmat
from pypacmensl.fsp_solver import FspSolverMultiSinks
from pypacmensl.smfish.snapshot import SmFishSnapshot
import mpi4py.MPI as mpi
from math import exp


class Il1bModel:
    t_meas = np.array([0, 0.5, 1, 2, 4]) * 3600

    # %% Load the necessary data
    def load_data(self, modelid):
        with np.load("il1b_data.npz", allow_pickle=True) as file:
            X = file["mrna_counts"]

        copymax = self.copymaxs[modelid]
        smfish_data = []
        for i in range(0, len(self.t_meas)):
            smfish_data.append(SmFishSnapshot(np.minimum(X[i], copymax)))
        return smfish_data

    NUM_PARAMETERS = 11

    # Prior mean and std in log10-transformed space
    PRIOR_MEAN = np.array([-2, -2, -3, -2, 3, -3, -2, -3, -0, -4, 4])

    PRIOR_STD = 1.0 / 3.0

    PAR_MLE = [
        -2.39942472,
        -2.0,
        -3.20775953,
        -1.66146085,
        3.69897,
        -3.18677747,
        -2.4970468,
        -3.11371105,
        -0.43187213,
        -4.24823665,
        4.936513742478893,
    ]  # the best I could find with optimization

    # %% Define the gene expression model
    STOICH_MATRIX = np.array(
        [
            [-1, 1, 0, 0],
            [1, -1, 0, 0],
            [0, -1, 1, 0],
            [0, 1, -1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, -1],
        ],
        dtype=np.intc,
    )
    X0 = np.array([[2, 0, 0, 0]])
    P0 = np.array([1.0])

    num_surrogates = 10
    copymaxs = 10 + 2 ** (np.arange(0, num_surrogates) + 2)
    copymaxs = np.array(copymaxs, dtype=int)
    print(copymaxs)
    ode_solver = "petsc"

    def __init__(self):
        print("Il1beta model.\n")

    def propensity_factory(self, theta, modelid):

        [r1, r2, k01, k10a, k10b, k12, k21, kr1, kr2, gamma, T0] = theta[:]
        copymax = self.copymaxs[modelid]

        def propensity_x(reaction, states, out):
            if reaction == 0:
                out[:] = k01 * states[:, 0] * (states[:, 3] < copymax)
            elif reaction == 1:
                out[:] = states[:, 1] * (states[:, 3] < copymax)
            elif reaction == 2:
                out[:] = k12 * states[:, 1] * (states[:, 3] < copymax)
            elif reaction == 3:
                out[:] = k21 * states[:, 2] * (states[:, 3] < copymax)
            elif reaction == 4:
                out[:] = kr1 * states[:, 1] * (states[:, 3] < copymax)
            elif reaction == 5:
                out[:] = kr2 * states[:, 2] * (states[:, 3] < copymax)
            elif reaction == 6:
                out[:] = gamma * states[:, 3] * (states[:, 3] < copymax)

        def propensity_t(t, out):
            if t <= T0:
                signal = 0.0
            else:
                signal = exp(-r1 * (t - T0)) * (
                    1.0 - exp(-r2 * (t - T0))
                )  # normalized hump function
            out[:] = 1.0
            out[1] = max(0.0, k10a - k10b * signal)

        return propensity_t, propensity_x

    # %% Likelihood function
    def solve_model(self, log10_theta, modelid):
        def fsp_constr(X, out):
            out[:, 0] = X[:, 0]
            out[:, 1] = X[:, 1]
            out[:, 2] = X[:, 2]
            out[:, 3] = X[:, 3]
            out[:, 4] = X[:, 3]

        theta = np.power(10.0, log10_theta)
        copymax = self.copymaxs[modelid]
        constr_init = np.array([2, 2, 2, 5, copymax])
        propensity_t, propensity_x = self.propensity_factory(theta, modelid)
        SOLVER0 = FspSolverMultiSinks(mpi.COMM_SELF)
        SOLVER0.SetModel(self.STOICH_MATRIX, propensity_t, propensity_x, [1])
        SOLVER0.SetFspShape(
            constr_fun=fsp_constr,
            constr_bound=constr_init,
            exp_factors=np.array([0.0, 0.0, 0.0, 0.2, 0.0]),
        )
        SOLVER0.SetInitialDist(self.X0, self.P0)
        SOLVER0.SetOdeSolver(self.ode_solver)
        SOLVER0.SetOdeTolerances(1.0e-4, 1.0e-10)
        # SOLVER0.SetVerbosity(2)
        SOLVER0.SetUp()
        solutions = SOLVER0.SolveTspan(self.t_meas + theta[-1], 1.0e-8)
        SOLVER0.ClearState()
        return solutions

    def loglike(self, log10_thetas, dataz, modelid):
        nsamp = log10_thetas.shape[0]
        loglike = np.zeros((nsamp, 1))
        data = dataz[modelid]
        for jnc in range(0, nsamp):
            try:
                solutions = self.solve_model(log10_thetas[jnc, :], modelid)
                ll = 0.0
                for i in range(0, len(self.t_meas)):
                    ll = ll + data[i].LogLikelihood(solutions[i], np.array([3]))
            except RuntimeError:
                ll = -1.0e8
            loglike[jnc, 0] = ll
        return loglike

    def prior_gen(self, nsamp):
        thetas = np.empty([nsamp, self.NUM_PARAMETERS, 1])
        for i in range(0, self.NUM_PARAMETERS):
            thetas[:, i, 0] = np.random.normal(
                self.PRIOR_MEAN[i], self.PRIOR_STD, nsamp
            )
        return thetas

    def log_prior(self, log10_thetas):
        nsamp = log10_thetas.shape[0]
        lp = np.zeros((1, nsamp))
        for i in range(0, self.NUM_PARAMETERS):
            lp = (
                lp
                - 1.0
                / (2 * self.PRIOR_STD * self.PRIOR_STD)
                * (log10_thetas[:, i] - self.PRIOR_MEAN[i]) ** 2
            )
        return lp.T
