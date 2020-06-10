import sys
sys.path.append('../')
sys.path.append('../../')
import numpy as np
import mpi4py.MPI as mpi
from comparmental_gene_expression.gene_expr_model import GeneExpressionModel
from stmcmc import StMcmc_multifi, StMcmc
from pypacmensl.fsp_solver import FspSolverMultiSinks
from pypacmensl.ssa.ssa import SSASolver
from pypacmensl.smfish.snapshot import SmFishSnapshot


# %%
if __name__ == "__main__":
    rank = mpi.COMM_WORLD.Get_rank()
    np.random.seed(rank)

    n_percore = 1
    num_cores = mpi.COMM_WORLD.Get_size()

    nsamp = n_percore * num_cores

    ARGV = sys.argv

    if len(ARGV) > 1:
        num_gene_states = int(ARGV[1])
    else:
        num_gene_states = 2


    if len(ARGV) == 3:
        use_full = (ARGV[2].lower() == "full")
    else:
        use_full = False

    print(f"Number of gene states: {num_gene_states} \n")
    print(f"Use full: {use_full}")

    # Perform multifidelity STMCMC on three classes of gene expression models
    if not use_full:
        model = GeneExpressionModel(num_gene_states=num_gene_states)
        n_surrogates = model.num_surrogates
        dataz = []
        for surrrogate_id in range(0, n_surrogates):
            dataz.append(model.load_data(surrrogate_id))
        if rank == 0:
            theta0 = model.prior_gen(nsamp)
        else:
            theta0 = []
        my_mcmc = StMcmc_multifi()
        my_mcmc.Run(model.log_prior, model.loglike, dataz, theta0, f'gene_expression_{num_gene_states}_mcmc_multifi.npz', nsamp, n_surrogates)
    else:
        model = GeneExpressionModel(num_gene_states=num_gene_states)
        n_surrogates = model.num_surrogates
        dataz = []
        for surrrogate_id in range(0, n_surrogates):
            dataz.append(model.load_data(surrrogate_id))

        if rank == 0:
            theta0 = model.prior_gen(nsamp)
        else:
            theta0 = []
        my_mcmc_full = StMcmc()
        my_mcmc_full.Run(model.log_prior, model.loglike, dataz, theta0, f'gene_expression_{num_gene_states}_mcmc_full.npz', nsamp, n_surrogates-1)
