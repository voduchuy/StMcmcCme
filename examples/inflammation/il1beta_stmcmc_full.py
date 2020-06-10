import sys
sys.path.append('../')
sys.path.append('../../')
import numpy as np
import mpi4py.MPI as mpi
from inflammation.il1beta_model import Il1bModel
from stmcmc import StMcmc
from pypacmensl.fsp_solver import FspSolverMultiSinks
from pypacmensl.smfish.snapshot import SmFishSnapshot


# %%
# The following code will run the multifidelity STMCMC to sample from the posterior distribution of IL1b parameters given
# observations at 0, 0.5, 1, and 2 hr.
if __name__ == "__main__":
    rank = mpi.COMM_WORLD.Get_rank()
    np.random.seed(rank)

    n_percore = 1
    num_cores = mpi.COMM_WORLD.Get_size()

    nsamp = n_percore * num_cores

    model = Il1bModel()
    n_surrogates = model.num_surrogates
    dataz = []
    for surrogate_id in range(0, n_surrogates):
        dataz.append(model.load_data(surrogate_id))
    if rank == 0:
        theta0 = model.prior_gen(nsamp)
    else:
        theta0 = []
    my_mcmc = StMcmc()
    my_mcmc.Run(model.log_prior, model.loglike, dataz, theta0, f'il1b_mcmc_full.npz', nsamp, n_surrogates-1)

