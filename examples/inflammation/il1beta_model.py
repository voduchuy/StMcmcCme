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
        with np.load('il1b_data.npz', allow_pickle=True) as file:
            X = file['mrna_counts']

        copymax = self.copymaxs[modelid]
        smfish_data = []
        for i in range(0, len(self.t_meas)):
            smfish_data.append(SmFishSnapshot(np.minimum(X[i], copymax)))
        return smfish_data

    # #%% Get the parameter bounds
    # r1_lower = 1.0E-4
    # r2_lower = 1.0E-4
    # k01_lower = 1.0E-4
    # k10a_lower = 1.0E-4
    # k10b_lower = 1.0E-9
    # k12_lower = 1.0E-9
    # k21_lower = 1.0/(4*3600)
    # kr1_lower = 1.0E-6
    # kr2_lower = 1.0/3600
    # gamma_lower = 1.0/(10*3600)
    # T0_lower = 60.0
    #
    # r1_upper = 1.0E-2
    # r2_upper = 1.0E-2
    # k01_upper = 10.0
    # k10a_upper = 5000
    # k10b_upper = 5000.0
    # k12_upper = 10.0
    # k21_upper = 10.0
    # kr1_upper = 1.0
    # kr2_upper = 1.0
    # gamma_upper = 1/(30*60)
    # T0_upper = 36.0*3600
    #
    # theta_lb = np.array(
    #         [r1_lower, r2_lower, k01_lower, k10a_lower, k10b_lower, k12_lower, k21_lower, kr1_lower, kr2_lower, gamma_lower, T0_lower])
    # theta_ub = np.array(
    #         [r1_upper, r2_upper, k01_upper, k10a_upper, k10b_upper, k12_upper, k21_upper, kr1_upper, kr2_upper, gamma_upper, T0_upper])

    num_parameters = 11

    # Prior mean and std in log10-transformed space
    prior_mean = np.array([
            -2,
            -2,
            -3,
            -2,
            3,
            -3,
            -2,
            -3,
            -0,
            -4,
            4
    ])

    prior_std = 1.0/3.0

    par_opt = [-2.39942472, -2., -3.20775953, -1.66146085, 3.69897,
    -3.18677747, -2.4970468, -3.11371105, -0.43187213, -4.24823665, 4.936513742478893] # the best I could find with optimization

    # %% Define the gene expression model
    stoichm = np.array([[-1, 1, 0, 0],
                        [1, -1, 0, 0],
                        [0, -1, 1, 0],
                        [0, 1, -1, 0],
                        [0, 0, 0, 1],
                        [0, 0, 0, 1],
                        [0, 0, 0, -1]],
                       dtype=np.intc)
    X0 = np.array([[2, 0, 0, 0]])
    P0 = np.array([1.0])

    num_surrogates = 10
    copymaxs = 10 + 2**(np.arange(0, num_surrogates)+2)
    copymaxs = np.array(copymaxs, dtype=int)
    print(copymaxs)
    ode_solver = "petsc"

    def __init__(self):
        print('Il1beta model.\n')

    def propensity_factory(self, theta, modelid):

        [r1, r2, k01, k10a, k10b, k12, k21, kr1, kr2, gamma, T0] = theta[:]
        copymax = self.copymaxs[modelid]

        def propensity_x(reaction, states, out):
            if reaction == 0:
                out[:] = k01*states[:, 0] * (states[:, 3] < copymax)
            elif reaction == 1:
                out[:] = states[:, 1] * (states[:, 3] < copymax)
            elif reaction == 2:
                out[:] = k12*states[:, 1] * (states[:, 3] < copymax)
            elif reaction == 3:
                out[:] = k21*states[:, 2] * (states[:, 3] < copymax)
            elif reaction == 4:
                out[:] = kr1*states[:, 1] * (states[:, 3] < copymax)
            elif reaction == 5:
                out[:] = kr2*states[:, 2] * (states[:, 3] < copymax)
            elif reaction == 6:
                out[:] = gamma*states[:, 3] * (states[:, 3] < copymax)

        def propensity_t(t, out):
            if t <= T0:
                signal = 0.0
            else:
                signal = exp(-r1 * (t - T0)) * (1.0 - exp(-r2 * (t - T0)))  # normalized hump function
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
        SOLVER0.SetModel(self.stoichm, propensity_t, propensity_x, [1])
        SOLVER0.SetFspShape(constr_fun=fsp_constr, constr_bound=constr_init, exp_factors=np.array([0.0, 0.0, 0.0, 0.2, 0.0]))
        SOLVER0.SetInitialDist(self.X0, self.P0)
        SOLVER0.SetOdeSolver(self.ode_solver)
        SOLVER0.SetOdeTolerances(1.0e-4, 1.0e-10)
        # SOLVER0.SetVerbosity(2)
        SOLVER0.SetUp()
        solutions = SOLVER0.SolveTspan(self.t_meas + theta[-1], 1.0E-8)
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
                ll = -1.0E8
            loglike[jnc, 0] = ll
        return loglike

    def prior_gen(self, nsamp):
        thetas = np.empty([nsamp, self.num_parameters, 1])
        for i in range(0, self.num_parameters):
            thetas[:, i, 0] = np.random.normal(self.prior_mean[i], self.prior_std, nsamp)
        return thetas

    def log_prior(self, log10_thetas):
        nsamp = log10_thetas.shape[0]
        lp = np.zeros((1, nsamp))
        for i in range(0, self.num_parameters):
            lp = lp - 1.0 / (2 * self.prior_std * self.prior_std) * (log10_thetas[:, i] - self.prior_mean[i]) ** 2
        return lp.T
    # # Uniform prior on the bounded parameter domain
    # def prior_gen(self, nsamp):
    #     thetas = np.zeros((nsamp, self.num_parameters, 1))
    #     for i in range(0, nsamp):
    #         thetas[i, :, 0] = np.random.uniform(low=np.log10(self.theta_lb), high=np.log10(self.theta_ub))
    #     return thetas
    # 
    # def log_prior(self, log10_thetas):
    #     nsamp = log10_thetas.shape[0]
    #     lprior = np.zeros((nsamp,1))
    #     for i in range(0, len(lprior)):
    #         lprior[i,0] = np.prod(1.0*(log10_thetas[i,:] >= np.log10(self.theta_lb))*(log10_thetas[i,:] <= np.log10(self.theta_ub)))
    #         if lprior[i, 0] == 0.0:
    #             lprior[i, 0] = -1.0E8
    #         else:
    #             lprior[i,0] = 0.0
    #     return lprior
