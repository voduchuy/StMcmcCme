import sys
sys.path.append('../')
sys.path.append('../../')
import numpy as np
import mpi4py.MPI as mpi
from stmcmc import StMcmc_multifi, StMcmc
from pypacmensl.fsp_solver import FspSolverMultiSinks
from pypacmensl.ssa.ssa import SSASolver
from pypacmensl.smfish.snapshot import SmFishSnapshot
from repressilator_model import RepressilatorModel

MODEL = RepressilatorModel()


# %%
if __name__ == "__main__":
    rank = mpi.COMM_WORLD.Get_rank()
    np.random.seed(rank)

    nmodel = MODEL.num_surrogates
    dataz = []
    for modelid in range(0, nmodel):
        dataz.append(MODEL.load_data(modelid))

    n_percore = 10
    num_cores = mpi.COMM_WORLD.Get_size()

    nsamp = n_percore * num_cores
    
    if rank == 0:
        theta0 = MODEL.prior_gen(nsamp)
    else:
        theta0 = []


    # Run with multifidelity ST-MCMC
    my_mcmc = StMcmc_multifi()
    my_mcmc.Run(MODEL.logprior, MODEL.loglike, dataz, theta0, 'repressilator_mcmc_multifi.npz', nsamp, nmodel)

    # Run with full STMCMC
    my_mcmc_full = StMcmc()
    my_mcmc_full.Run(MODEL.logprior, MODEL.loglike, dataz, theta0, 'repressilator_mcmc_full.npz', nsamp, nmodel-1)
