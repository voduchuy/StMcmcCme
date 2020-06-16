import sys

sys.path.append("../")
import numpy as np
import mpi4py.MPI as mpi
from scipy.stats import norm, poisson
from stmcmc import StMcmcSingleFidelity, StMcmcMultiFidelityIT, StMcmcMultiFidelityITTuned, StMcmcMultiFidelityEssBridge

#%%
WORLD = mpi.COMM_WORLD
RANK = WORLD.Get_rank()
NCPUS = WORLD.Get_size()
#%%
true_theta = 1000
def generate_data():
    if RANK == 0:
        samples = np.random.poisson(true_theta, size=100)
    else:
        samples = None
    return samples


#%%
data = generate_data()

data = WORLD.bcast(data, root=0)

#%%
PRIOR_MEAN = 3.0
PRIOR_STD = 0.5

def log_prior(log10thetas):
    return norm.pdf(log10thetas, loc=PRIOR_MEAN, scale=PRIOR_STD)

# multifidelity models are defined by bounds that are 50 units apart, starting at 10
def model_id_to_bound(id):
    return 10 + id*50
MAX_LEVEL = 40

def project_data(data, max_level):
    dataz = []
    for id in range(0, max_level+1):
        bound = model_id_to_bound(id)
        data_here = np.copy(data)
        data_here[data_here > bound] = bound-1
        dataz.append(data_here)
    return dataz

def log_likelihood(log10thetas, dataz, modelid):
    loglike_values = np.zeros((log10thetas.shape[0], 1))
    for i in range(0, log10thetas.shape[0]):
        loglike_values[i] = np.sum(
            poisson.logpmf(dataz[modelid][:], mu=10.0 ** log10thetas[i, :])
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

# Parse command-line arguments
ARGV = sys.argv

if len(ARGV) > 1:
    method = ARGV[1].lower()
else:
    method = "full"

if mpi.COMM_WORLD.Get_rank() == 0:
    print(f"Method: {method}")

if method == "ess":
    sampler = StMcmcMultiFidelityEssBridge()
    output_appendix = "mcmc_bridge"
elif method == "it":
    sampler = StMcmcMultiFidelityIT()
    output_appendix = "mcmc_multifi"
elif method == "it_tuned":
    sampler = StMcmcMultiFidelityITTuned()
    output_appendix = "mcmc_tuned"
else:
    sampler = StMcmcSingleFidelity()
    output_appendix = "mcmc_tuned"

dataz = project_data(data, MAX_LEVEL)

sampler.Run(
    log_prior,
    log_likelihood,
    dataz,
    log10thetas_init,
    f"poisson_{output_appendix}.npz",
    num_samples,
    MAX_LEVEL
)
#%%
import seaborn as sns
import matplotlib.pyplot as plt

if RANK == 0:
    with np.load(f"poisson_{output_appendix}.npz") as f:
        stmcmc_samples = f["theta"]
    fig, ax = plt.subplots(1,1)
    sns.distplot(stmcmc_samples[:, 0, 0], ax=ax, color='red', label="Prior")
    sns.distplot(stmcmc_samples[:, 0, -1], ax=ax, color='blue', label="Posterior")
    ax.axvline(x=np.log10(true_theta), label="True")
    ax.legend()
    plt.show()





