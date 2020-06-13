import sys

sys.path.append("../")
import numpy as np
import mpi4py.MPI as mpi
from scipy.stats import norm, poisson
from stmcmc import StMcmcSingleFidelity

#%%
WORLD = mpi.COMM_WORLD
RANK = WORLD.Get_rank()
NCPUS = WORLD.Get_size()
#%%
def generate_data():
    if RANK == 0:
        samples = np.random.poisson(1.0, size=100)
    else:
        samples = None
    return samples


#%%
data = generate_data()

data = WORLD.bcast(data, root=0)

#%%
PRIOR_MEAN = 1.0
PRIOR_STD = 1.0


def log_prior(log10thetas):
    return norm.pdf(log10thetas, loc=PRIOR_MEAN, scale=PRIOR_STD)


def log_likelihood(log10thetas, data, modelid):
    loglike_values = np.zeros((log10thetas.shape[0], 1))
    for i in range(0, log10thetas.shape[0]):
        loglike_values[i] = np.sum(
            poisson.logpmf(data[:], mu=10.0 ** log10thetas[i, :])
        )
    return loglike_values


def sample_prior(num_samples):
    return np.random.normal(
        loc=PRIOR_MEAN, scale=PRIOR_STD, size=(num_samples, 1, 1)
    )


#%%
SAMPLES_PER_CPUS = 1000
num_samples = SAMPLES_PER_CPUS * NCPUS
log10thetas_init = sample_prior(num_samples)

sampler = StMcmcSingleFidelity(WORLD)
sampler.Run(
    log_prior,
    log_likelihood,
    data,
    log10thetas_init,
    "poisson_mcmc.npz",
    num_samples,
)
#%%
import seaborn as sns
import matplotlib.pyplot as plt

if RANK == 0:
    with np.load("poisson_mcmc.npz") as f:
        stmcmc_samples = f["theta"]
    fig, ax = plt.subplots(1,1)
    sns.distplot(stmcmc_samples[:, 0, 0], ax=ax, color='red', label="Prior")
    sns.distplot(stmcmc_samples[:, 0, -1], ax=ax, color='blue', label="Posterior")
    ax.axvline(x=0, label="True")
    ax.legend()
    plt.show()





