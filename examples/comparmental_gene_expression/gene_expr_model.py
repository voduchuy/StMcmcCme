import sys

sys.path.append("../")
sys.path.append("../../")
import numpy as np
import mpi4py.MPI as mpi
from stmcmc import StMcmcMultiFidelityIT
from pypacmensl.fsp_solver import FspSolverMultiSinks
from pypacmensl.ssa.ssa import SSASolver
from pypacmensl.smfish.snapshot import SmFishSnapshot

# %% Experiment design
t_meas = np.array([2.0, 4.0, 6.0, 8.0, 10.0]) * 60.0
n_cells = 200
# %% Define the sequence of multifidelity models
class GeneExpressionModel:

    copymaxs = np.linspace(10, 400, 10, dtype=int)
    num_surrogates = len(copymaxs)

    def __init__(self, num_gene_states=4):
        print(
            f"Comparmental gene expression model with {num_gene_states} gene states.\n"
        )

        self.num_gene_states = num_gene_states
        # Stoichiometry matrix, we have (num_gene_states - 1)*2 reactions for gene state switching, (num_gene_states - 1) reactions for nuclear mRNA production,
        # 1 reaction for ncluear mrNA translocation and 1 for cytoplasmic RNA degradation, state vector has the form (gene_state_0, .., gene_state_{num_gene_state}, RNA_nuc, RNA_cyt)
        self.stoich_mat = np.zeros(
            ((num_gene_states - 1) * 3 + 2, num_gene_states + 2)
        )
        for i in range(0, num_gene_states - 1):
            # Transition from gene state i to gene state i+1
            self.stoich_mat[i, i] = -1
            self.stoich_mat[i, i + 1] = 1
            # Transition from gene state i+1 to gene state i
            self.stoich_mat[num_gene_states - 1 + i, i] = 1
            self.stoich_mat[num_gene_states - 1 + i, i + 1] = -1
        # Production of nuclear mRNA
        self.stoich_mat[
            (num_gene_states - 1) * 2 : (num_gene_states - 1) * 2
            + (num_gene_states - 1),
            num_gene_states,
        ] = 1
        # Transport of nuclear mRNA
        self.stoich_mat[-2, -2] = -1
        self.stoich_mat[-2, -1] = 1
        # Decay of cytoplasmic mRNA
        self.stoich_mat[-1, -1] = -1

        # Initial state and initial FSP constraints
        self.x0 = np.zeros((1, num_gene_states + 2), dtype=int)
        self.x0[0, 0] = 1
        self.constr_init = np.ones((num_gene_states + 2,))
        self.p0 = np.array([1.0])

        # Define the Gaussian prior (in log10-transformed space)
        self.prior_mean = np.zeros((3 * (num_gene_states - 1) + 2,))
        self.prior_mean[0 : 2 * (num_gene_states - 1)] = -1.0
        self.prior_mean[
            2 * (num_gene_states - 1) : 3 * (num_gene_states - 1)
        ] = 1.0
        self.prior_mean[-2] = -1.0
        self.prior_mean[-1] = -1.0
        self.prior_std = 1.0 / 3.0

        # # Parameter space will have dimension 3*(num_gene_states - 1) + 2
        # # theta = [ k_{01}, .., k_{num_gene_state-2, num_gene_state-1}, k_{10}, k_{21}, .., k_{num_gene_states - 1, num_gene_states-2},
        # # kr_{1}, ..., kr_{num_gene_states-1}, ktrans, deg]
        self.num_parameters = 3 * (num_gene_states - 1) + 2

    def prop_factory(self, surrogate_id):

        copymax = self.copymaxs[surrogate_id]
        print(copymax)

        def propensity(reaction, x, out):
            active_states = (x[:, -1] < copymax) * (x[:, -2] < copymax)
            # Reaction that up the gene state
            if reaction < self.num_gene_states - 1:
                out[:] = x[:, reaction] * active_states
            # Reaction that down the gene state
            if (
                self.num_gene_states - 1
                <= reaction
                < 2 * (self.num_gene_states - 1)
            ):
                out[:] = (
                    x[:, reaction - self.num_gene_states + 2] * active_states
                )
            # mRNA production
            if (
                2 * (self.num_gene_states - 1)
                <= reaction
                < 2 * (self.num_gene_states - 1) + (self.num_gene_states - 1)
            ):
                out[:] = (
                    x[:, reaction - 2 * (self.num_gene_states - 1) + 1]
                    * active_states
                )
            if reaction == 3 * (self.num_gene_states - 1):
                out[:] = x[:, -2] * active_states
            if reaction == 3 * (self.num_gene_states - 1) + 1:
                out[:] = x[:, -1] * active_states

        return propensity

    def t_fun_factory(self, theta):
        def t_fun(t, out):
            out[:] = theta[:]

        return t_fun

    def load_data(self, surrogate_id):
        npzdat = np.load("gene_expression_data.npz")
        X = npzdat["arr_0"]
        data = []
        copymax = self.copymaxs[surrogate_id]

        for i in range(0, X.shape[0]):
            data.append(SmFishSnapshot(np.minimum(X[i][:, -2:], copymax)))
        return data

    def prior_gen(self, nsamp):
        thetas = np.empty([nsamp, self.num_parameters, 1])
        for i in range(0, self.num_parameters):
            thetas[:, i, 0] = np.random.normal(
                self.prior_mean[i], self.prior_std, nsamp
            )
        return thetas

    def logprior(self, log10_thetas):
        nsamp = log10_thetas.shape[0]
        lp = np.zeros((1, nsamp))
        for i in range(0, self.num_parameters):
            lp = (
                lp
                - 1.0
                / (2 * self.prior_std * self.prior_std)
                * (log10_thetas[:, i] - self.prior_mean[i]) ** 2
            )
        return lp.T

    # %% Set up the log-likelihood calculation

    def loglike(self, log10_thetas, dataz, surrogate_id):
        thetas = np.power(10.0, log10_thetas)
        nsamp = thetas.shape[0]
        loglike = np.zeros((nsamp, 1))
        data = dataz[surrogate_id]

        for jnc in range(0, nsamp):
            t_fun = self.t_fun_factory(thetas[jnc, :])
            prop = self.prop_factory(surrogate_id)
            solver = FspSolverMultiSinks(mpi.COMM_SELF)
            solver.SetModel(self.stoich_mat, t_fun, prop)
            solver.SetFspShape(None, self.constr_init)
            solver.SetInitialDist(self.x0, self.p0)
            solver.SetUp()
            solutions = solver.SolveTspan(t_meas, 1.0e-8)
            solver.ClearState()
            ll = 0.0
            for i in range(0, len(t_meas)):
                ll = ll + data[i].LogLikelihood(
                    solutions[i],
                    np.array([self.num_gene_states, self.num_gene_states + 1]),
                )
            loglike[jnc, 0] = ll
        return loglike


def simulate_data(t_meas, n_cells):
    """Simulate SmFish observations"""

    # The true model will have 3 gene states
    true_model = GeneExpressionModel(num_gene_states=3)

    k01 = 10.0 ** (-2.52)
    k12 = 10.0 ** (-3.1)
    k10 = 10.0 ** (-2.22)
    k21 = 10.0 ** (-1.5)
    kr1 = 10.0 ** (1.76e-1)
    kr2 = 10.0 ** (2.5e-1)
    trans = 10.0 ** (-2.0)
    gamma_cyt = 10.0 ** (-2.52)

    theta_true = np.array([k01, k12, k10, k21, kr1, kr2, trans, gamma_cyt])

    # For generating Data high copy max
    modelid = true_model.num_surrogates - 1
    ssa = SSASolver(mpi.COMM_SELF)
    ssa.SetModel(
        true_model.stoich_mat,
        true_model.t_fun_factory(theta_true),
        true_model.prop_factory(modelid),
    )
    print(true_model.stoich_mat)
    observations = []
    for t in t_meas:
        observations.append(
            ssa.Solve(t, true_model.x0, n_cells, send_to_root=True)
        )
    if mpi.COMM_WORLD.Get_rank() == 0:
        np.savez("gene_expression_data", observations)
    mpi.COMM_WORLD.Barrier()


# %%
if __name__ == "__main__":
    rank = mpi.COMM_WORLD.Get_rank()
    np.random.seed(rank)
    if rank == 0:
        simulate_data(t_meas, n_cells)
