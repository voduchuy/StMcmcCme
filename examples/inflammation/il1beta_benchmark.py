#%% Compare the performance of different settings for the ODE module in solving the CME of IL1Beta
import sys

sys.path.append("../")
sys.path.append("../../")
import numpy as np
import mpi4py.MPI as mpi
import matplotlib.pyplot as plt
from inflammation.il1beta_model import Il1bModel
# %%
model = Il1bModel()

if __name__ == "__main__":
    comm = mpi.COMM_WORLD
    nprocs = comm.Get_size()
    rank = comm.Get_rank()

    log10theta = model.PAR_MLE

    cput_multi_petsc = np.zeros((model.num_surrogates,))
    cput_multi_cvode = np.zeros((model.num_surrogates,))
    for i in range(0, model.num_surrogates):
        model.ODE_METHOD = "petsc"
        t1 = mpi.Wtime()
        model.solve_model(log10theta, i)
        cput_multi_petsc[i] = mpi.Wtime() - t1

        model.ODE_METHOD = "cvode"
        t2 = mpi.Wtime()
        model.solve_model(log10theta, i)
        cput_multi_cvode[i] = mpi.Wtime() - t2

    np.savez("il1b_bench.npz", tpetsc=cput_multi_petsc, tcvode=cput_multi_cvode)

    f = np.load("il1b_bench.npz")
    cput_multi_cvode = f["tcvode"]
    cput_multi_petsc = f["tpetsc"]
    fig, ax = plt.subplots(1, 1)
    ax.plot(
        np.arange(0, model.num_surrogates),
        cput_multi_petsc,
        color="darkgreen",
        label="PETSC",
    )
    ax.plot(
        np.arange(0, model.num_surrogates),
        cput_multi_cvode,
        color="r",
        label="CVODE",
    )
    ax.set_xlabel("Model id")
    ax.set_ylabel("Solve time (sec)")
    ax.legend()
    fig.savefig("il1beta_benchmark.pdf")
    plt.show()
