# This script compares the performance of two FSP-based approaches to computing the surrogate log-likelihood of
# single-cell data for the repressilator gene circuit.
#
# The first approach is based on the classical FSP. One static state space is generated from the onset, and the FSP
# solution is obtained by solving the ODEs associated with this static state space.
#
# The second approach is based on a combination of classical FSP and its adaptive variants. In particular,
# by modifying the propensities, we implicitly define a limit on the maximum size of the FSP. However, an adaptive
# FSP method is deployed that gradually expand the projection along the integration time. This means that we may not
# need to solve one big ODEs system from the onset of the integration, but can let the rigorous FSP error bound
# guides the computation.
#
# The major difference between the two approach is that the classic FSP requires solving a full set of ODEs from the
# start, while the second approach approximate the classic with high precision using a more effecient adaptive
# expansion.
#
# This script should be ran with one core only, since we're only using the serial version of the FSP solver.
import numpy as np
import mpi4py.MPI as mpi
import matplotlib.pyplot as plt
from pypacmensl.fsp_solver import FspSolverMultiSinks
from repressilator_model import RepressilatorModel
from time import time
from numba import jit

#%%
class RepressilatorModelExt(RepressilatorModel):
    def loglike_classic(self, log10_thetas, dataz, modelid):
        """
        Evaluate the surrogate log-likelihood function but using a classic (i.e. static) FSP approach on the
        surrogate CME.
        """
        thetas = np.power(10.0, log10_thetas)
        nsamp = thetas.shape[0]
        loglike = np.zeros((nsamp, 1))
        data = dataz[modelid]

        # Use the largest FSP bounds from the onset, this is the main difference with the adaptive FSP approach
        constr_full = np.zeros((3,))
        constr_full[:] = self.copymaxs[:, modelid]

        for jnc in range(0, nsamp):
            p0 = np.array([1.0])
            prop = self.prop_factory(thetas[jnc, :], modelid)
            solver = FspSolverMultiSinks(mpi.COMM_SELF)
            solver.SetModel(self.stoich_mat, None, prop)
            solver.SetFspShape(None, constr_full)
            solver.SetInitialDist(self.x0, p0)
            solver.SetOdeSolver("KRYLOV")
            solver.SetKrylovOrthLength(2)
            solver.SetKrylovDimRange(30, 30)
            solutions = solver.SolveTspan(self.t_meas, 1.0e-8)
            ll = 0.0
            for i in range(0, len(self.t_meas)):
                ll = ll + data[i].LogLikelihood(
                        solutions[i], np.array([0, 1, 2])
                )
            loglike[jnc, 0] = ll
        return loglike


#%%
model = RepressilatorModelExt()

log10_thetas = np.array(np.log10(model.theta_true)).reshape((1, -1))
dataz = []
for surrogate_id in range(0, model.num_surrogates):
    dataz.append(model.load_data(surrogate_id))

performance = {
    "classic": np.zeros((model.num_surrogates,)),
    "adaptive": np.zeros((model.num_surrogates)),
}

for surrogate_id in range(0, model.num_surrogates):
    print(surrogate_id)

    tic = mpi.Wtime()
    model.loglike_classic(log10_thetas, dataz, surrogate_id)
    performance["classic"][surrogate_id] = mpi.Wtime() - tic

    tic = mpi.Wtime()
    model.loglike(log10_thetas, dataz, surrogate_id)
    performance["adaptive"][surrogate_id] = mpi.Wtime() - tic

    print(f" Classic time: {performance['classic'][surrogate_id]} \\ \
    Adaptive time: {performance['adaptive'][surrogate_id]}")

#%% Save and plot performance curves
np.savez("repressilator_fsp_performance.npz", classic=performance["classic"], adaptive=performance["adaptive"])

fig, ax = plt.subplots(1,1)

ax.plot(range(0, model.num_surrogates), performance["classic"], label="Classic FSP")
ax.plot(range(0, model.num_surrogates), performance["adaptive"], label="Adaptive FSP (our approach)")
ax.legend()
ax.set_xlabel("Surrogate ID")
ax.set_ylabel("CPU time (sec)")

plt.show()

