import sys
sys.path.append('../')
sys.path.append('../../')
import numpy as np
import mpi4py.MPI as mpi
from inflammation.il1beta_model import Il1bModel
from stmcmc import StMcmc_tuned
from pypacmensl.fsp_solver import FspSolverMultiSinks
from pypacmensl.smfish.snapshot import SmFishSnapshot


# %%
# The following code loads the posterior samples of IL1b parameters given
# observations at 0, 0.5, 1, and 2 hr, and makes prediction on the probability distribution of mRNA copy number at 4 hr.
# The prediction consists of the predictive mean and standard deviation of the probabilities of observing mRNA counts from 0 to 2000
model = Il1bModel()

def predict_mrna_dist(log10_theta):
    solutions = model.solve_model(log10_theta, model.num_surrogates-1)
    pmrna = np.zeros(( len(model.t_meas), model.copymaxs[-1]+1))
    for i in range(0, len(solutions)):
        pmrna1 = solutions[i].Marginal(3)
        pmrna[i, 0:len(pmrna1)] = pmrna1
    return pmrna

if __name__ == "__main__":
    comm = mpi.COMM_WORLD
    nprocs = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        with np.load(f'il1b_mcmc_tuned.npz') as file:
            Xglobal = file['theta']
            Xglobal = Xglobal[:, :, -1]
            nglobal = Xglobal.shape[0]
            npar = Xglobal.shape[1]
            Xglobal = np.ascontiguousarray(Xglobal)
    else:
        Xglobal = None
        npar = 0
        nglobal = 0

    npar = comm.bcast(npar, root=0)
    nglobal = comm.bcast(nglobal, root=0)

    veclenmax = model.copymaxs[-1]+ 1

    # Determine number of particles per process
    n1 = nglobal // nprocs
    r1 = nglobal % nprocs
    samples_per_proc = n1 * np.ones((nprocs,), dtype=int)
    samples_per_proc[0:r1] += 1
    num_samples_local = samples_per_proc[rank]

    # Distribute particles to processes
    X = np.zeros((num_samples_local, npar), dtype=float)
    if rank == 0:
        displacements = np.zeros((nprocs,), dtype=int)
        displacements[1:] = npar*np.cumsum(samples_per_proc[0:-1])
    else:
        displacements = None
    comm.Scatterv([Xglobal, npar*samples_per_proc, displacements, mpi.DOUBLE], X, root=0)

    # Compute mean and std of mRNA probabilities at t = 4 hr
    prna_mean = np.zeros((len(model.t_meas), veclenmax))
    prna_std = np.zeros((len(model.t_meas), veclenmax))

    # Take the local weighted sum of p_rna
    for i in range(0, num_samples_local):
        p_rna = predict_mrna_dist(X[i,:])
        prna_mean += (1.0/nglobal)*p_rna
        prna_std += (1.0/nglobal)*p_rna*p_rna

    # Reduce all of them
    if rank == 0:
        prna_mean0 = np.zeros((len(model.t_meas), veclenmax))
        prna_std0 = np.zeros((len(model.t_meas), veclenmax))
    else:
        prna_mean0 = None
        prna_std0 = None

    comm.Reduce(sendbuf=[prna_mean, len(model.t_meas)*veclenmax, mpi.DOUBLE], recvbuf=[prna_mean0, len(model.t_meas)*veclenmax, mpi.DOUBLE], op=mpi.SUM, root=0)
    comm.Reduce(sendbuf=[prna_std, len(model.t_meas)*veclenmax, mpi.DOUBLE], recvbuf=[prna_std0, len(model.t_meas)*veclenmax, mpi.DOUBLE], op=mpi.SUM, root=0)

    if rank == 0:
        prna_std0 = np.sqrt(prna_std0 - prna_mean0*prna_mean0)
        np.savez('il1b_prediction_tuned.npz', mean=prna_mean0, std=prna_std0)





