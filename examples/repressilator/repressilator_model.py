import numpy as np
from pypacmensl.ssa.ssa import SSASolver
from pypacmensl.fsp_solver import FspSolverMultiSinks
from pypacmensl.smfish.snapshot import SmFishSnapshot
import mpi4py.MPI as mpi
import matplotlib.pyplot as plt

# %% Define the sequence of multifidelity models
class RepressilatorModel:
    # %% Experiment design parameters
    t_meas = np.linspace(10.0 / 5, 10.0, 5)
    n_cells = 1000

    stoich_mat = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    )

    x0 = np.array([[0, 0, 0]])
    constr_init = np.array([10, 10, 10])

    # % Max number of molecules in the 'hard' FSP reduction
    num_surrogates = 10
    copymaxs = np.zeros((3, num_surrogates), dtype=int)
    copymaxs[0, :] = np.linspace(20, 50, num_surrogates, dtype=int)
    copymaxs[1, :] = np.linspace(40, 100, num_surrogates, dtype=int)
    copymaxs[2, :] = np.linspace(40, 100, num_surrogates, dtype=int)

    k0_true = 10.0
    gamma0_true = 0.01
    a0_true = 0.1
    b0_true = 2.0
    k1_true = 7.5
    gamma1_true = 0.02
    a1_true = 0.01
    b1_true = 2.5
    k2_true = 10.0
    gamma2_true = 0.05
    a2_true = 0.05
    b2_true = 3.0

    theta_true = np.array(
        [
            k0_true,
            gamma0_true,
            a0_true,
            b0_true,
            k1_true,
            gamma1_true,
            a1_true,
            b1_true,
            k2_true,
            gamma2_true,
            a2_true,
            b2_true,
        ]
    )

    # Reference parameter, this is an initial guess for the true parameter
    theta_ref = np.array(
        [10.0, 0.1, 0.1, 0.1, 10.0, 0.1, 0.1, 0.1, 10.0, 0.1, 0.1, 0.1]
    )

    def __init__(self):
        print("Repressilator model")

    def prop_factory(self, theta, modelid):
        [k0, gamma0, a0, b0, k1, gamma1, a1, b1, k2, gamma2, a2, b2] = theta[:]

        copymax = self.copymaxs[:, modelid]

        def propensity(reaction, x, out):
            if reaction == 0:
                out[:] = (
                    k0 / (1.0 + a0 * x[:, 1] ** b0) * (x[:, 0] < copymax[0])
                )
            if reaction == 1:
                out[:] = gamma0 * x[:, 0]
            if reaction == 2:
                out[:] = (
                    k1 / (1.0 + a1 * x[:, 2] ** b1) * (x[:, 1] < copymax[1])
                )
            if reaction == 3:
                out[:] = gamma1 * x[:, 1]
            if reaction == 4:
                out[:] = (
                    k2 / (1.0 + a2 * x[:, 0] ** b2) * (x[:, 2] < copymax[2])
                )
            if reaction == 5:
                out[:] = gamma2 * x[:, 2]

        return propensity

    def t_fun_factory(self, theta):
        def t_fun(t, out):
            out[:] = 1.0

        return t_fun

    def load_data(self, modelid):
        npzdat = np.load("repressilator_data.npz")
        X = npzdat["arr_0"]
        data = []
        copymax = self.copymaxs[:, modelid]

        for i in range(0, X.shape[0]):
            data.append(SmFishSnapshot(np.minimum(X[i][:, 0:3], copymax)))
        return data

    # Normal prior in log10-transformed space, surrounding the reference (not the true) parameters
    sigma0 = 0.3
    mu0 = np.log10(theta_ref)

    def prior_gen(self, nsamp):
        thetas = np.empty([nsamp, 12, 1])
        for i in range(0, 12):
            thetas[:, i, 0] = np.random.normal(self.mu0[i], self.sigma0, nsamp)
        return thetas

    def logprior(self, log10_thetas):
        nsamp = log10_thetas.shape[0]
        lp = np.zeros((1, nsamp))
        for i in range(0, 12):
            lp = (
                lp
                - 1.0
                / (2 * self.sigma0 * self.sigma0)
                * (log10_thetas[:, i] - self.mu0[i]) ** 2
            )

        return lp.T

    # %% Set up the log-likelihood calculation

    def loglike(self, log10_thetas, dataz, modelid):
        thetas = np.power(10.0, log10_thetas)
        nsamp = thetas.shape[0]
        loglike = np.zeros((nsamp, 1))
        data = dataz[modelid]
        solver = FspSolverMultiSinks(mpi.COMM_SELF)
        for jnc in range(0, nsamp):
            p0 = np.array([1.0])
            prop = self.prop_factory(thetas[jnc, :], modelid)
            solver.SetModel(self.stoich_mat, None, prop)
            solver.SetFspShape(None, self.constr_init)
            solver.SetInitialDist(self.x0, p0)
            solver.SetOdeSolver("KRYLOV")
            solver.SetKrylovOrthLength(2)
            solutions = solver.SolveTspan(self.t_meas, 1.0e-8)
            ll = 0.0
            for i in range(0, len(self.t_meas)):
                ll = ll + data[i].LogLikelihood(
                    solutions[i], np.array([0, 1, 2])
                )
            loglike[jnc, 0] = ll
        return loglike

    def simulate_data(self, t_meas, n_cells):
        """Simulate SmFish observations"""
        # For generating Data high copy max
        ssa = SSASolver(mpi.COMM_SELF)
        ssa.SetModel(
            self.stoich_mat,
            self.t_fun_factory(self.theta_true),
            self.prop_factory(self.theta_true, self.num_surrogates - 1),
        )
        observations = []
        for t in t_meas:
            observations.append(
                ssa.Solve(t, self.x0, n_cells, send_to_root=True)
            )

        np.savez("repressilator_data", observations)


if __name__ == "__main__":
    model = RepressilatorModel()
    rank = mpi.COMM_WORLD.Get_rank()
    np.random.seed(rank)
    if rank == 0:
        model.simulate_data(model.t_meas, model.n_cells)
        with np.load("repressilator_data.npz", allow_pickle=True) as data:
            observations = data["arr_0"]

        pairs = [[0, 1], [1, 2], [2, 0]]
        fig, axes = plt.subplots(1, 3)
        for i in range(0, len(pairs)):
            axes[i].scatter(
                observations[len(model.t_meas) - 1][:, pairs[i][0]],
                observations[len(model.t_meas) - 1][:, pairs[i][1]],
            )
        plt.show()
