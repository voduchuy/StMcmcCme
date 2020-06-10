import sys
sys.path.append('../')
import numpy as np
import mpi4py.MPI as mpi
from stmcmc import StMcmc
from pypacmensl.fsp_solver import FspSolverMultiSinks
from pypacmensl.ssa.ssa import SSASolver
from pypacmensl.smfish.snapshot import SmFishSnapshot

# %% Experiment design
t_meas = np.array([2.0, 6.0, 8.0]) * 3600.0
n_cells = 500


# %% Model structure
class ToggleModel:
    stoich_mat = np.array([[1, 0],
                           [1, 0],
                           [-1, 0],
                           [0, 1],
                           [0, 1],
                           [0, -1]])

    x0 = np.array([[0, 0]])
    constr_init = np.array([100, 100])

    ayx = 2.6e-3
    axy = 6.1e-3
    nyx = 3
    nxy = 2.1
    k0x = 10 ** (-2.66)
    k1x = 10 ** (-1.77)
    gammax = 10 ** (-3.42)
    k0y = 10 ** (-4.17)
    k1y = 10 ** (-1.8)
    gammay = 10 ** (-3.42)

    theta_true = np.array([k0x, k1x, gammax, k0y, k1y, gammay])

    # Reference parameter, this is an initial guess for the true parameter
    theta_ref = theta_true
    # theta_ref = np.array([1.0e-2, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-2, 1.0e-3])

    def __init__(self):
        print("Toggle model")

    def prop_factory(self):
        def propensity(reaction, x, out):
            if reaction == 0:
                out[:] = 1.0
            if reaction == 1:
                out[:] = np.reciprocal(1.0 + self.ayx * np.power(x[:, 1], self.nyx))
            if reaction == 2:
                out[:] = x[:, 0]
            if reaction == 3:
                out[:] = 1.0
            if reaction == 4:
                out[:] = np.reciprocal(1.0 + self.axy * np.power(x[:, 0], self.nxy))
            if reaction == 5:
                out[:] = x[:, 1]

        return propensity

    def t_fun_factory(self, theta):
        def t_fun(t, out):
            out[:] = theta[:]

        return t_fun


model = ToggleModel()


# %%
def simulate_data(t_meas, n_cells):
    """Simulate SmFish observations"""
    ssa = SSASolver(mpi.COMM_WORLD)
    ssa.SetModel(ToggleModel.stoich_mat, model.t_fun_factory(model.theta_true), model.prop_factory())
    observations = []
    for t in t_meas:
        observations.append(ssa.Solve(t, ToggleModel.x0, n_cells, send_to_root=True))
    if mpi.COMM_WORLD.Get_rank() == 0:
        np.savez('toggle_data', observations)
    mpi.COMM_WORLD.Barrier()


# %%
# Normal prior in log10-transformed space, surrounding the reference (not the true) parameters
sigma = 0.3
mu = np.log10(ToggleModel.theta_ref)


def prior_gen(nsamp):
    thetas = np.empty([nsamp, 6, 1])
    for i in range(0, 6):
        thetas[:, i, 0] = np.random.normal(mu[i], sigma, nsamp)
    return thetas


def logprior(log10_thetas):
    nsamp = log10_thetas.shape[0]
    ploglike = np.zeros((1, nsamp))
    for i in range(0, 6):
        ploglike = ploglike - 1.0 / (2 * sigma * sigma) * (log10_thetas[:, i] - mu[i]) ** 2

    # invalid = np.logical_or((thetas[:,3] > 250.0), (thetas[:,3] < 0))
    # ploglike[invalid,0] = -np.inf

    return ploglike.T


# %% Set up the log-likelihood calculation

def loglike(log10_thetas, data):
    thetas = np.power(10.0, log10_thetas)
    nsamp = thetas.shape[0]
    ndim = thetas.shape[1]
    loglike = np.zeros((nsamp, 1))

    for jnc in range(0, nsamp):
        #print(thetas[jnc, :], loglike[jnc, 0])
        p0 = np.array([1.0])

        t_fun = model.t_fun_factory(thetas[jnc, :])
        prop = model.prop_factory()

        solver = FspSolverMultiSinks(mpi.COMM_SELF)
        solver.SetFspShape(None, model.constr_init)
        solver.SetModel(model.stoich_mat, t_fun, prop)
        solver.SetFspShape(None, model.constr_init)
        solver.SetInitialDist(model.x0, p0)
        solver.SetUp()
        solutions = solver.SolveTspan(t_meas, 1.0e-4)
        solver.ClearState()
        ll = 0.0
        for i in range(0, len(t_meas)):
            ll = ll + data[i].LogLikelihood(solutions[i])
        loglike[jnc, 0] = ll
        print(thetas[jnc, :], loglike[jnc, 0])
    return loglike


# %%
def load_data():
    npzdat = np.load('toggle_data.npz')
    X = npzdat['arr_0']
    data = []
    for i in range(0, X.shape[0]):
        data.append(SmFishSnapshot(X[i][:, :]))
    return data


# %%
if __name__ == "__main__":
    rank = mpi.COMM_WORLD.Get_rank()
    np.random.seed(rank)
    simulate_data(t_meas, n_cells)
    data = load_data()

    nsamp = 144
    if rank == 0:
        theta0 = prior_gen(nsamp)
    else:
        theta0 = []
    my_mcmc = StMcmc()
    my_mcmc.Run(logprior, loglike, data, theta0, 'toggle_mcmc.npz', nsamp)
