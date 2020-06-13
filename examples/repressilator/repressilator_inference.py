import sys

sys.path.append("../")
sys.path.append("../../")
import numpy as np
import mpi4py.MPI as mpi
from stmcmc import (
    StMcmcMultiFidelityIT,
    StMcmcMultiFidelityITTuned,
    StMcmcMultiFidelityEssBridge,
    StMcmcSingleFidelity,
)
from repressilator_model import RepressilatorModel

MODEL = RepressilatorModel()


# %%
if __name__ == "__main__":
    rank = mpi.COMM_WORLD.Get_rank()
    np.random.seed(rank)

    # Number of samples per CPU
    num_samples_percore = 10

    # Number of CPUs available
    num_cores = mpi.COMM_WORLD.Get_size()

    num_samples_global = num_samples_percore * num_cores

    # Parse command-line arguments
    ARGV = sys.argv

    if len(ARGV) > 1:
        num_gene_states = int(ARGV[1])
    else:
        num_gene_states = 2

    if len(ARGV) == 3:
        method = ARGV[2].lower()
    else:
        method = "full"

    if mpi.COMM_WORLD.Get_rank() == 0:
        print(f"Number of gene states: {num_gene_states} \n")
        print(f"Method: {method}")

    model = RepressilatorModel()
    num_surrogates = model.num_surrogates
    dataz = []
    for surrrogate_id in range(0, num_surrogates):
        dataz.append(model.load_data(surrrogate_id))

    if rank == 0:
        theta0 = model.prior_gen(num_samples_global)
    else:
        theta0 = []

    if method == "ess":
        sampler = StMcmcMultiFidelityEssBridge()
        output_appendix = "mcmc_bridge"
    elif method == "it":
        sampler = StMcmcMultiFidelityIT()
        output_appendix = "mcmc_multifi"
    elif method == "it_tuned":
        sampler = StMcmcMultiFidelityITTuned
        output_appendix = "mcmc_tuned"
    else:
        sampler = StMcmcSingleFidelity()
        output_appendix = "mcmc_full"

    sampler.Run(
        model.logprior,
        model.loglike,
        dataz,
        theta0,
        f"gene_expression_{num_gene_states}_mcmc_full.npz",
        num_samples_global,
        num_surrogates - 1,
    )
