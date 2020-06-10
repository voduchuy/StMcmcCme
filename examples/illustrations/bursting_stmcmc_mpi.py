#!/usr/bin/env python
# coding: utf-8
import numpy as np
import copy
import time
from mpi4py import MPI

from pypacmensl.smfish.snapshot import SmFishSnapshot
from pypacmensl.fsp_solver import FspSolverMultiSinks

# %% Experiment design
t_meas = np.linspace(0, 10, 5)
# %% True parameter values
k_off = 0.5
k_on = 0.1
k_r = 50.0
gamma = 1.0

theta_true = np.array([k_off, k_on, k_r, gamma])
theta_ref = np.array([1.0, 1.0, 10.0, 1.0])
# %% Define model structure
stoich_mat = np.array([[-1, 1, 0],
                       [1, -1, 0],
                       [0, 0, 1],
                       [0, 0, -1]])
x0 = np.array([[1, 0, 0]])
p0 = np.array([1.0])
s0 = np.array([0.0])
constr_init = np.array([1, 1, 10])


def propensity(reaction, x, out):
    if reaction is 0:
        out[:] = x[:, 0]
    if reaction is 1:
        out[:] = x[:, 1]
    if reaction is 2:
        out[:] = x[:, 1]
    if reaction is 3:
        out[:] = x[:, 2]


def t_fun_factory(theta):
    def t_fun(t, out):
        out[:] = theta[:]

    return t_fun


def dt_fun_factory(theta):
    dt_list = []

    def d_t_fun(i):
        def d_t_(t, out):
            out[i] = 1.0

        return d_t_

    for i in range(0, 4):
        dt_list.append(d_t_fun(i))
    return dt_list


# %% Load the simulated data
npzdat = np.load('bursting_simulated_data.npz')
X = npzdat['arr_0']
data = []
for i in range(0, X.shape[0]):
    data.append(SmFishSnapshot( X[i][ :, 2]))


# %%

# Normal prior in log10-transformed space, surrounding the reference (not the true) parameters
sigma = 0.3
mu = np.log10(theta_ref)
def prior_gen(nsamp):
    mu = np.log10(theta_ref)
    thetas = np.empty([nsamp, 4, 1])
    thetas[:, 0, 0] = np.random.normal(mu[0], sigma, nsamp)
    thetas[:, 1, 0] = np.random.normal(mu[1], sigma, nsamp)
    thetas[:, 2, 0] = np.random.normal(mu[2], sigma, nsamp)
    thetas[:, 3, 0] = np.random.normal(mu[3], sigma, nsamp)
    return thetas


def ploglike_cal(log10_thetas):
    nsamp = log10_thetas.shape[0]
    ploglike = np.zeros((1, nsamp))
    ploglike = ploglike - 1.0 / (2*sigma*sigma) * (log10_thetas[:, 0] - mu[0]) ** 2
    ploglike = ploglike - 1.0 / (2*sigma*sigma) * (log10_thetas[:, 1] - mu[1]) ** 2
    ploglike = ploglike - 1.0 / (2*sigma*sigma) * (log10_thetas[:, 2] - mu[2]) ** 2
    ploglike = ploglike - 1.0 / (2*sigma*sigma) * (log10_thetas[:, 3] - mu[3]) ** 2

    # invalid = np.logical_or((thetas[:,3] > 250.0), (thetas[:,3] < 0))
    # ploglike[invalid,0] = -np.inf

    return ploglike.T


# %% Set up the log-likelihood calculation

solver = FspSolverMultiSinks(MPI.COMM_SELF)
solver.SetFspShape(None, constr_init)
# solver.SetOdeSolver("Krylov")

def loglike_cal(log10_thetas, data):
    thetas = np.power(10.0, log10_thetas)
    nsamp = thetas.shape[0]
    ndim = thetas.shape[1]
    loglike = np.zeros((nsamp, 1))

    for jnc in range(0, nsamp):
        t_fun = t_fun_factory(thetas[jnc, :])
        solver.SetModel(stoich_mat, t_fun, propensity)
        solver.SetFspShape(None, constr_init)
        solver.SetInitialDist(x0, p0)
        solver.SetUp()
        solutions = solver.SolveTspan(t_meas, 1.0e-2)
        solver.ClearState()
        ll = 0.0
        for i in range(0, len(t_meas)):
            ll = ll + data[i].LogLikelihood(solutions[i], np.array([2]))
        loglike[jnc, 0] = ll
        # print(thetas[jnc,:], loglike[jnc, 0])
    return loglike


# %% Let's do it

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# make each rank of its own seed
np.random.seed(int(time.time()) + rank)

nsamp = size * 250
ncores = size  # Assumes nsamp/ncores is an integer
stepmax = 1000
targcov = 1.0
learnlev = 1
nchain = 1
step = 1

if rank == 0:
    tevals = np.zeros(1)
    times = np.zeros(1)
    t0 = time.time()

# if you are rank 0 generate the parameters and evaluate the prior
ndim = None
sendtheta = None
counts = None
dspls = None
if rank == 0:
    theta = prior_gen(nsamp)
    sendtheta = theta[:, :, 0]
    stheta = np.copy(theta)
    ploglike = ploglike_cal(theta[:, :, 0])
    ndim = theta.shape[1]
    counts = ndim * nsamp // size * np.ones(size, dtype=int)
    dspls = range(0, nsamp * ndim, ndim * nsamp // size)

# send dimension information to everyone
ndim = comm.bcast(ndim, root=0)

# holder for everyone to recive there part of the theta vector
recvtheta = np.zeros((nsamp // size, ndim))

# get your theta values to use for computing th loglike
comm.Scatterv([sendtheta, counts, dspls, MPI.DOUBLE], recvtheta, root=0)

# compute on the loglike on each core
sendloglike = loglike_cal(recvtheta, data)

scounts = nsamp // size
rcounts = nsamp // size * np.ones(size, dtype=int)
rdspls = range(0, nsamp, nsamp // size)

# send core information to the root
recvloglike = None
if rank == 0:
    recvloglike = np.zeros(nsamp)
comm.Gatherv([sendloglike, scounts, MPI.DOUBLE], [
        recvloglike, rcounts, rdspls, MPI.DOUBLE], root=0)

# define various variables we will need on the root node
if rank == 0:
    loglike = np.zeros((nsamp, 1))
    loglike[:, 0] = recvloglike

    beta = np.zeros(1)
    acr = np.zeros(1)
    rho = np.zeros(1)
    scal = np.zeros(1)

    scale = 0.3 * (2.38 ** 2) / ndim
    targacr = 0.25

# everyone needs targrho
targrho = 0.6
for istep in range(0, stepmax):

    # if you are root compute the new beta
    if rank == 0:
        dbeta0 = 0.0
        dbeta1 = 1.0

        maxl = max(loglike[:, istep])
        odbeta = 10.0
        ddbeta = odbeta

        while (ddbeta > (10.0 ** (-10.0))):
            dbeta = (dbeta0 + dbeta1) / 2.0

            wi = np.divide(np.exp(
                    dbeta * (loglike[:, istep] - maxl)), sum(np.exp(dbeta * (loglike[:, istep] - maxl))))
            fi = np.sqrt(
                    1.0 / nsamp * sum(np.power(wi - np.mean(wi), 2))) / np.mean(wi) - targcov

            if fi <= 0.0:
                dbeta0 = dbeta
            else:
                dbeta1 = dbeta

            ddbeta = np.abs(odbeta - dbeta)
            odbeta = dbeta

        if beta[istep] + dbeta >= 1.0:
            dbeta = 1.0 - beta[istep]
            wi = np.divide(np.exp(
                    dbeta * (loglike[:, istep] - maxl)), sum(np.exp(dbeta * (loglike[:, istep] - maxl))))
            beta = np.append(beta, 1.0)
        else:
            beta = np.append(beta, beta[istep] + dbeta)

        sampmean = np.average(theta[:, :, istep], 0, wi, False)
        sampcov = np.cov(theta[:, :, istep], None,
                         False, False, None, None, wi)

        sampidx = np.random.choice(nsamp, nsamp, True, wi)
        inittheta = theta[sampidx, :, istep]

        otheta = np.copy(inittheta)
        ologlike = loglike[sampidx, istep]
        oploglike = ploglike[sampidx, istep]
        startloglike = np.copy(ologlike)
        startploglike = np.copy(oploglike)

        sacc = 0.0

        start_theta = np.copy(otheta)
        teval = 0

    # each core goes thorugh this while look until we tell it to stop
    rest = 1.0
    while (rest > targrho):
        # determine how long a chain to run (nchain) right now we take this as given

        # generate random number seeds (This is now done at the beginning of the code
        # my random numbers are ndim*nsamp*nchain multivariate_normal and nsamp*nchain uniform

        # scatter otheta
        sendotheta = None
        counts = None
        dspls = None
        if rank == 0:
            sendotheta = otheta
            counts = ndim * nsamp // size * np.ones(size, dtype=int)
            dspls = range(0, nsamp * ndim, ndim * nsamp // size)

        local_otheta = np.zeros((nsamp // size, ndim))
        comm.Scatterv([sendotheta, counts, dspls, MPI.DOUBLE],
                      local_otheta, root=0)

        # broadcast scale*sampcov
        local_propcov = np.zeros((ndim, ndim))
        if rank == 0:
            local_propcov = scale * sampcov
        comm.Bcast(local_propcov, root=0)

        # broadcast data (we assume each core already has this)
        local_data = data

        # scatter ologlike
        sendologlike = None
        counts = None
        dspls = None
        if rank == 0:
            sendologlike = ologlike
            counts = nsamp // size * np.ones(size, dtype=int)
            dspls = range(0, nsamp, nsamp // size)

        local_ologlike = np.zeros(nsamp // size)
        comm.Scatterv([sendologlike, counts, dspls, MPI.DOUBLE],
                      local_ologlike, root=0)

        # scatter oploglike
        sendoploglike = None
        counts = None
        dspls = None
        if rank == 0:
            sendoploglike = oploglike
            counts = nsamp // size * np.ones(size, dtype=int)
            dspls = range(0, nsamp, nsamp // size)

        local_oploglike = np.zeros(nsamp // size)
        comm.Scatterv([sendoploglike, counts, dspls, MPI.DOUBLE],
                      local_oploglike, root=0)

        # broadcast beta[istep+1]
        sendbeta = None
        if rank == 0:
            sendbeta = beta[istep + 1]
        local_beta = comm.bcast(sendbeta, root=0)

        # local chain stats
        local_sacc = np.zeros(1, dtype=np.int)
        local_tevals = np.zeros(1, dtype=np.int)

        # run code in parallel for each chain
        for inc in range(0, nchain):

            ctheta = local_otheta + \
                     np.random.multivariate_normal(
                             np.zeros(ndim), local_propcov, nsamp // size)
            cploglike = ploglike_cal(ctheta)[:, 0]
            cloglike = loglike_cal(ctheta, data)[:, 0]

            like = (local_beta * cloglike + cploglike) - \
                   (local_beta * local_ologlike + local_oploglike)
            acc = ((np.log(np.random.rand(nsamp // size)) < like))
            local_otheta[acc, :] = ctheta[acc, :]
            local_ologlike[acc] = cloglike[acc]
            local_oploglike[acc] = cploglike[acc]
            local_sacc = local_sacc + sum(acc)
            local_tevals = local_tevals + nsamp // size

        # gather otheta
        scounts = ndim * nsamp // size
        rcounts = ndim * nsamp // size * np.ones(size, dtype=int)
        rdspls = range(0, nsamp * ndim, ndim * nsamp // size)
        recvotheta = None
        if rank == 0:
            recvotheta = np.zeros((nsamp, ndim))
        comm.Gatherv([local_otheta, scounts, MPI.DOUBLE], [
                recvotheta, rcounts, rdspls, MPI.DOUBLE], root=0)
        if rank == 0:
            otheta = recvotheta

        # gather ologlike
        scounts = nsamp // size
        rcounts = nsamp // size * np.ones(size, dtype=int)
        rdspls = range(0, nsamp, nsamp // size)

        recvologlike = None
        if rank == 0:
            recvologlike = np.zeros(nsamp)
        comm.Gatherv([local_ologlike, scounts, MPI.DOUBLE], [
                recvologlike, rcounts, rdspls, MPI.DOUBLE], root=0)
        if rank == 0:
            ologlike = recvologlike

        # gather oploglike
        scounts = nsamp // size
        rcounts = nsamp // size * np.ones(size, dtype=int)
        rdspls = range(0, nsamp, nsamp // size)

        recvoploglike = None
        if rank == 0:
            recvoploglike = np.zeros(nsamp)
        comm.Gatherv([local_oploglike, scounts, MPI.DOUBLE], [
                recvoploglike, rcounts, rdspls, MPI.DOUBLE], root=0)
        if rank == 0:
            oploglike = recvoploglike

        # reduce (add) sacc
        rsacc = np.zeros(1, dtype=np.int)
        comm.Reduce([local_sacc, MPI.INT], rsacc, root=0, op=MPI.SUM)
        if rank == 0:
            sacc = sacc + rsacc

        # reduce (add) tevals
        rtevals = np.zeros(1, dtype=np.int)
        comm.Reduce([local_tevals, MPI.INT], rtevals, root=0, op=MPI.SUM)
        if rank == 0:
            teval = teval + rtevals

        # compute correlation information
        if rank == 0:
            corrmat = np.corrcoef(start_theta, otheta, False)
            rest = max(np.diagonal(corrmat, ndim))
            t1 = time.time() - t0

            print(str(teval) + " " + str(rest) + " " + str(t1))

        # broadcast rest
        sendrest = None
        if rank == 0:
            sendrest = rest
        rest = comm.bcast(sendrest, root=0)

    if rank == 0:
        end_theta = np.copy(otheta)
        sacc = sacc / teval

        corrmat = np.corrcoef(start_theta, end_theta, False)
        rest_state = max(np.diagonal(corrmat, ndim))

        acr = np.append(acr, sacc)
        rho = np.append(rho, rest_state)
        scal = np.append(scal, scale)
        tevals = np.append(tevals, teval)
        times = np.append(times, time.time() - t0)

        scale = scale * np.exp(2.0 * 2.1 * (sacc - targacr))

        print(str(istep) + " " + str(teval) + " " +
              str(beta[istep + 1]) + " " + str(rest_state) + " " + str(time.time() - t0))

        stheta = np.dstack((stheta, start_theta))
        theta = np.dstack((theta, otheta))
        loglike = np.column_stack((loglike, ologlike))
        ploglike = np.column_stack((ploglike, oploglike))

        np.savez('st_mcmc_test.npz', stheta=stheta, theta=theta, loglike=loglike,
                 ploglike=ploglike, times=times, tevals=tevals, scal=scal, rho=rho, acr=acr, beta=beta)

    # broadcast beta to everyone to break
    # broadcast beta[istep+1]
    sendbeta = None
    if rank == 0:
        sendbeta = beta[istep + 1]
    local_beta = comm.bcast(sendbeta, root=0)

    if local_beta >= 1.0:
        break
