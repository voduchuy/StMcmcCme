import sys
sys.path.append('../')
sys.path.append('../../')
import numpy as np
import mpi4py.MPI as mpi
from inflammation.il1beta_model import Il1bModel
from stmcmc import StMcmc_multifi
from pypacmensl.fsp_solver import FspSolverMultiSinks
from pypacmensl.smfish.snapshot import SmFishSnapshot


# %%
# The following code loads the posterior samples of IL1b parameters given
# observations at 0, 0.5, 1, and 2 hr, and makes prediction on the probability distribution of mRNA copy number at 4 hr.
# The prediction consists of the predictive mean and standard deviation of the probabilities of observing mRNA counts from 0 to 2000
model = Il1bModel()

def predict_mrna_dist(theta):
    propensity_t, propensity_x = model.propensity_factory(theta)
    SOLVER0 = FspSolverMultiSinks(mpi.COMM_SELF)
    SOLVER0.SetVerbosity(0)
    SOLVER0.SetModel(model.stoichm, propensity_t, propensity_x)
    SOLVER0.SetFspShape(constr_fun=None, constr_bound=model.init_bounds, exp_factors=np.array([0.0, 0.0, 0.0, 0.2]))
    SOLVER0.SetInitialDist(model.X0, model.P0)
    SOLVER0.SetUp()
    solution = SOLVER0.Solve(4.0*3600.0 + theta[-1], 1.0E-8)
    SOLVER0.ClearState()
    pmrna1 = solution.Marginal(3)
    pmrna = np.zeros((2001,))
    pmrna[0:len(pmrna1)] = pmrna1
    return pmrna

if __name__ == "__main__":
    comm = mpi.COMM_WORLD
    nprocs = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        with np.load(f'il1b_mcmc_multifi.npz') as file:
            Xglobal = file['theta']
            Xglobal = Xglobal[:, :, -1]
            nglobal = Xglobal.shape[0]
            npar = Xglobal.shape[1]
    else:
        npar = 0
        nglobal = 0

    npar = comm.bcast(npar, root=0)
    nglobal = comm.bcast(nglobal, root=0)

    # Determine number of particles per process
    n1 = nglobal // nprocs
    r1 = nglobal % nprocs
    samples_per_proc = n1 * np.ones((nprocs,), dtype=int)
    samples_per_proc[0:r1] += 1
    num_samples_local = samples_per_proc[rank]

    # Distribute particles to processes
    X = np.zeros((num_samples_local, dim), dtype=float)
    if self.rank_ == 0:
        displacements = np.zeros((nprocs,), dtype=int)
        displacements[1:] = dim*np.cumsum(samples_per_proc[0:-1])
    else:
        displacements = None
    comm.Scatterv([Xglobal, dim*samples_per_proc, displacements, MPI.DOUBLE], X, root=0)

    # Compute mean and std of mRNA probabilities at t = 4 hr
    prna_mean = np.zeros((2000,))
    prna_std = np.zeros((2000,))

    # Take the local weighted sum of p_rna
    for i in range(0, num_samples_local):
        p_rna = predict_mrna_dist(10.0**X[i,:])
        prna_mean += (1.0/nglobal)*p_rna
        prna_std += (1.0/nglobal)*p_rna*p_rna

    # Reduce all of them
    if rank == 0:
        prna_mean0 = np.zeros((2000,))
        prna_std = np.zeros((2000,))
    else:
        prna_mean0 = None
        prna_std = None

    comm.Reduce(sendbuf=[prna_mean, 2000, MPI.DOUBLE], recvbuf=[prna_mean0, 2000, MPI.DOUBLE], op=mpi.SUM, root=0)
    comm.Reduce(sendbuf=[prna_std, 2000, MPI.DOUBLE], recvbuf=[prna_std0, 2000, MPI.DOUBLE], op=mpi.SUM, root=0)

    if rank == 0:
        prna_std0 = np.sqrt(prna_std0 - prna_mean0*prna_mean0)
        np.savez('il1b_prediction.npz', mean=prna_mean0, std=prna_std0)





