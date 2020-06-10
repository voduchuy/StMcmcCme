import sys
sys.path.append('../')
sys.path.append('../../')
import numpy as np
import mpi4py.MPI as mpi
from stmcmc import StMcmc_multifi, StMcmc, StMcmc_restart
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

    nsamp = 1044

    # Run with full STMCMC
    my_mcmc_full = StMcmc_restart()
    my_mcmc_full.Run(MODEL.logprior, MODEL.loglike, dataz, 'repressilator_mcmc_full.npz', nsamp, nmodel-1)
