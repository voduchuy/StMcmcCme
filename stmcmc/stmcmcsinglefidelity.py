# coding: utf-8
import numpy as np
import copy
import time
from mpi4py import MPI
from typing import Callable, Any


class StMcmcSingleFidelity:
    """
    Sequential Tempered MCMC sampler with only one fidelity level (usually the full fidelity).
    """

    def __init__(self, comm: MPI.Comm = MPI.COMM_WORLD):
        self.comm_ = comm.Dup()
        self.num_procs_ = comm.Get_size()
        self.rank_ = comm.Get_rank()
        self.nsamp_ = 1044
        self.max_num_steps_ = 1000
        self.targcov_ = 1.0
        self.learnlev_ = 1
        self.nchain_ = 1
        self.step_ = 1
        self.modelid_ = 0
        self.maxcount_ = 100

    def Run(
        self,
        logprior_fun: Callable[[np.ndarray], float],
        loglike_fun: Callable[[np.ndarray, Any, int], float],
        data: Any,
        init_thetas: np.ndarray,
        outfile: str = None,
        nsamp: int = 128,
        modelid: int = 0,
        max_num_steps: int = 1000,
        maxcount: int = 100,
        start_new=True,
    ):
        """
        Sample from the posterior distribution using the STMCMC sampler.

        Parameters
        ----------
        logprior_fun : Callable
            Function to evaluate the logarithm (base 10) of the prior.
            Syntax: y = logprior_fun(thetas), where thetas is an array of samples, arranged row-wise, and y is an
            array of shape (number of samples, 1).

        loglike_fun : Callable
            Function to evaluate the logarithm (base 10) of the likelihood.
            Syntax:

        data : Any
            Object that contains the data needed for the likelihood evaluation.

        init_thetas: np.ndarray
            Samples from the prior. Could be None if start_new == False.

        outfile : str
            Name of the output file for the chains.

        nsamp : int
            Total number of samples. This number must be divisible by the number of calling processors.

        modelid : int
            Identity of the model in the multifidelity hierarchy. This sampler only uses one level.

        max_num_steps : int
            Maximum number of MCMC steps in the rejunevation phase.

        maxcount : int
            [???]

        start_new : bool
            Default is True. If False, the sampler will first load the chains saved in ``outfile`` and continue from
            there.

        Returns
        -------

            None

        """
        if outfile is None:
            outfile = "stmcmc_test.npz"

        if type(outfile) != str:
            raise RuntimeError("Output filename must be a string!")

        ndim = None

        if start_new:
            thetas = init_thetas
            if self.rank_ == 0:
                stheta = np.copy(thetas)
                ploglike = self.logprior_(thetas[:, :, 0])
                ndim = thetas.shape[1]
                tevals = np.zeros(1)
                times = np.zeros(1)
                sendtheta = thetas[:, :, 0]
                stheta = np.copy(thetas)
                ploglike = self.logprior_(thetas[:, :, 0])
                ndim = thetas.shape[1]
                counts = (
                    ndim
                    * nsamp
                    // self.num_procs_
                    * np.ones(self.num_procs_, dtype=int)
                )
                dspls = range(0, nsamp * ndim, ndim * nsamp // self.num_procs_)

            # send dimension information to everyone
            ndim = self.comm_.bcast(ndim, root=0)

            # holder for everyone to recive there part of the theta vector
            recvtheta = np.zeros((nsamp // self.num_procs_, ndim))

            # get your theta values to use for computing th loglike
            self.comm_.Scatterv(
                [sendtheta, counts, dspls, MPI.DOUBLE], recvtheta, root=0
            )

            # compute on the loglike on each core, modelid needs to be 0 since only one model
            sendloglike = self.loglike_(recvtheta, data, modelid)

            scounts = nsamp // self.num_procs_
            rcounts = (
                nsamp // self.num_procs_ * np.ones(self.num_procs_, dtype=int)
            )
            rdspls = range(0, nsamp, nsamp // self.num_procs_)
            # send core information to the root
            recvloglike = None
            if self.rank_ == 0:
                recvloglike = np.zeros(nsamp)

            self.comm_.Gatherv(
                [sendloglike, scounts, MPI.DOUBLE],
                [recvloglike, rcounts, rdspls, MPI.DOUBLE],
                root=0,
            )

            loglike = np.zeros((nsamp, 1))
            loglike[:, 0] = recvloglike

            beta = np.zeros(1)
            acr = np.zeros(1)
            rho = np.zeros(1)
            scal = np.zeros(1)

            scale = 0.3 * (2.38 ** 2) / ndim
            targacr = 0.25
        else:
            # load the theta from last run
            inputdat = np.load(outfile)
            thetas = inputdat["theta"]
            lastistep = thetas.shape[2] - 1
            inputdat.close()
            # define various variables we will need on the root node
            if self.rank_ == 0:
                loglike = inputdat["loglike"]
                beta = inputdat["beta"]
                acr = inputdat["acr"]
                rho = inputdat["rho"]
                scal = inputdat["scal"]
                scale = scal[lastistep]
                targacr = 0.25
                stheta = inputdat["stheta"]
                ploglike = inputdat["ploglike"]
                ndim = thetas.shape[1]
                tevals = inputdat["tevals"]
                times = inputdat["times"]

        self.nsamp_ = nsamp
        self.max_num_steps_ = max_num_steps
        self.maxcount = maxcount
        # make each rank of its own seed
        np.random.seed(int(time.time()) + self.rank_)

        # send dimension information to everyone
        ndim = self.comm_.bcast(ndim, root=0)

        # everyone needs targrho
        targrho = 0.6

        if self.comm_.Get_rank() == 0:
            t0 = time.time()

        for istep in range(lastistep, self.max_num_steps_):

            # if you are root compute the new beta
            if self.rank_ == 0:
                dbeta0 = 0.0
                dbeta1 = 1.0

                maxl = max(loglike[:, istep])
                odbeta = 10.0
                ddbeta = odbeta

                while ddbeta > (10.0 ** (-10.0)):
                    dbeta = (dbeta0 + dbeta1) / 2.0

                    wi = np.divide(
                        np.exp(dbeta * (loglike[:, istep] - maxl)),
                        sum(np.exp(dbeta * (loglike[:, istep] - maxl))),
                    )
                    fi = (
                        np.sqrt(
                            1.0 / nsamp * sum(np.power(wi - np.mean(wi), 2))
                        )
                        / np.mean(wi)
                        - self.targcov_
                    )

                    if fi <= 0.0:
                        dbeta0 = dbeta
                    else:
                        dbeta1 = dbeta

                    ddbeta = np.abs(odbeta - dbeta)
                    odbeta = dbeta

                if beta[istep] + dbeta >= 1.0:
                    dbeta = 1.0 - beta[istep]
                    wi = np.divide(
                        np.exp(dbeta * (loglike[:, istep] - maxl)),
                        sum(np.exp(dbeta * (loglike[:, istep] - maxl))),
                    )
                    beta = np.append(beta, 1.0)
                else:
                    beta = np.append(beta, beta[istep] + dbeta)

                sampmean = np.average(thetas[:, :, istep], 0, wi, False)
                sampcov = np.cov(
                    thetas[:, :, istep], None, False, False, None, None, wi
                )

                sampidx = np.random.choice(nsamp, nsamp, True, wi)
                inittheta = thetas[sampidx, :, istep]

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
            totalcount = 0
            while (rest > targrho) and (totalcount < self.maxcount_):
                totalcount = totalcount + 1
                # determine how long a chain to run (nchain) right now we take this as given

                # generate random number seeds (This is now done at the beginning of the code
                # my random numbers are ndim*nsamp*nchain multivariate_normal and nsamp*nchain uniform

                # scatter otheta
                sendotheta = None
                counts = None
                dspls = None
                if self.rank_ == 0:
                    sendotheta = otheta
                    counts = (
                        ndim
                        * nsamp
                        // self.num_procs_
                        * np.ones(self.num_procs_, dtype=int)
                    )
                    dspls = range(
                        0, nsamp * ndim, ndim * nsamp // self.num_procs_
                    )

                local_otheta = np.zeros((nsamp // self.num_procs_, ndim))
                self.comm_.Scatterv(
                    [sendotheta, counts, dspls, MPI.DOUBLE],
                    local_otheta,
                    root=0,
                )

                # broadcast scale*sampcov
                local_propcov = np.zeros((ndim, ndim))
                if self.rank_ == 0:
                    local_propcov = scale * sampcov
                self.comm_.Bcast(local_propcov, root=0)

                # scatter ologlike
                sendologlike = None
                counts = None
                dspls = None
                if self.rank_ == 0:
                    sendologlike = ologlike
                    counts = (
                        nsamp
                        // self.num_procs_
                        * np.ones(self.num_procs_, dtype=int)
                    )
                    dspls = range(0, nsamp, nsamp // self.num_procs_)

                local_ologlike = np.zeros(nsamp // self.num_procs_)
                self.comm_.Scatterv(
                    [sendologlike, counts, dspls, MPI.DOUBLE],
                    local_ologlike,
                    root=0,
                )

                # scatter oploglike
                sendoploglike = None
                counts = None
                dspls = None
                if self.rank_ == 0:
                    sendoploglike = oploglike
                    counts = (
                        nsamp
                        // self.num_procs_
                        * np.ones(self.num_procs_, dtype=int)
                    )
                    dspls = range(0, nsamp, nsamp // self.num_procs_)

                local_oploglike = np.zeros(nsamp // self.num_procs_)
                self.comm_.Scatterv(
                    [sendoploglike, counts, dspls, MPI.DOUBLE],
                    local_oploglike,
                    root=0,
                )

                # broadcast beta[istep+1]
                sendbeta = None
                if self.rank_ == 0:
                    sendbeta = beta[istep + 1]
                local_beta = self.comm_.bcast(sendbeta, root=0)

                # local chain stats
                local_sacc = np.zeros(1, dtype=np.int)
                local_tevals = np.zeros(1, dtype=np.int)

                # run code in parallel for each chain
                for inc in range(0, self.nchain_):

                    ctheta = local_otheta + np.random.multivariate_normal(
                        np.zeros(ndim), local_propcov, nsamp // self.num_procs_
                    )
                    cploglike = logprior_fun(ctheta)[:, 0]
                    cloglike = loglike_fun(ctheta, data, modelid)[:, 0]

                    like = (local_beta * cloglike + cploglike) - (
                        local_beta * local_ologlike + local_oploglike
                    )
                    acc = (
                        np.log(np.random.rand(nsamp // self.num_procs_)) < like
                    )
                    local_otheta[acc, :] = ctheta[acc, :]
                    local_ologlike[acc] = cloglike[acc]
                    local_oploglike[acc] = cploglike[acc]
                    local_sacc = local_sacc + sum(acc)
                    local_tevals = local_tevals + nsamp // self.num_procs_

                # gather otheta
                scounts = ndim * nsamp // self.num_procs_
                rcounts = (
                    ndim
                    * nsamp
                    // self.num_procs_
                    * np.ones(self.num_procs_, dtype=int)
                )
                rdspls = range(0, nsamp * ndim, ndim * nsamp // self.num_procs_)
                recvotheta = None
                if self.rank_ == 0:
                    recvotheta = np.zeros((nsamp, ndim))
                self.comm_.Gatherv(
                    [local_otheta, scounts, MPI.DOUBLE],
                    [recvotheta, rcounts, rdspls, MPI.DOUBLE],
                    root=0,
                )
                if self.rank_ == 0:
                    otheta = recvotheta

                # gather ologlike
                scounts = nsamp // self.num_procs_
                rcounts = (
                    nsamp
                    // self.num_procs_
                    * np.ones(self.num_procs_, dtype=int)
                )
                rdspls = range(0, nsamp, nsamp // self.num_procs_)

                recvologlike = None
                if self.rank_ == 0:
                    recvologlike = np.zeros(nsamp)
                self.comm_.Gatherv(
                    [local_ologlike, scounts, MPI.DOUBLE],
                    [recvologlike, rcounts, rdspls, MPI.DOUBLE],
                    root=0,
                )
                if self.rank_ == 0:
                    ologlike = recvologlike

                # gather oploglike
                scounts = nsamp // self.num_procs_
                rcounts = (
                    nsamp
                    // self.num_procs_
                    * np.ones(self.num_procs_, dtype=int)
                )
                rdspls = range(0, nsamp, nsamp // self.num_procs_)

                recvoploglike = None
                if self.rank_ == 0:
                    recvoploglike = np.zeros(nsamp)
                self.comm_.Gatherv(
                    [local_oploglike, scounts, MPI.DOUBLE],
                    [recvoploglike, rcounts, rdspls, MPI.DOUBLE],
                    root=0,
                )
                if self.rank_ == 0:
                    oploglike = recvoploglike

                # reduce (add) sacc
                rsacc = np.zeros(1, dtype=np.int)
                self.comm_.Reduce(
                    [local_sacc, MPI.INT], rsacc, root=0, op=MPI.SUM
                )
                if self.rank_ == 0:
                    sacc = sacc + rsacc

                # reduce (add) tevals
                rtevals = np.zeros(1, dtype=np.int)
                self.comm_.Reduce(
                    [local_tevals, MPI.INT], rtevals, root=0, op=MPI.SUM
                )
                if self.rank_ == 0:
                    teval = teval + rtevals

                # compute correlation information
                if self.rank_ == 0:
                    corrmat = np.corrcoef(start_theta, otheta, False)
                    rest = max(np.diagonal(corrmat, ndim))
                    t1 = time.time() - t0

                    print(str(teval) + " " + str(rest) + " " + str(t1))

                # broadcast rest
                sendrest = None
                if self.rank_ == 0:
                    sendrest = rest
                rest = self.comm_.bcast(sendrest, root=0)

            if self.rank_ == 0:
                end_theta = np.copy(otheta)
                sacc = sacc / teval

                corrmat = np.corrcoef(start_theta, end_theta, False)
                rest_state = max(np.diagonal(corrmat, ndim))

                acr = np.append(acr, sacc)
                rho = np.append(rho, rest_state)
                scal = np.append(scal, scale)
                tevals = np.append(tevals, teval)
                times = np.append(times, time.time() - t0 + times[lastistep])

                scale = scale * np.exp(2.0 * 2.1 * (sacc - targacr))

                print(
                    str(istep)
                    + " "
                    + str(teval)
                    + " "
                    + str(beta[istep + 1])
                    + " "
                    + str(rest_state)
                    + " "
                    + str(time.time() - t0)
                )

                stheta = np.dstack((stheta, start_theta))
                thetas = np.dstack((thetas, otheta))
                loglike = np.column_stack((loglike, ologlike))
                ploglike = np.column_stack((ploglike, oploglike))

                np.savez(
                    outfile,
                    stheta=stheta,
                    theta=thetas,
                    loglike=loglike,
                    ploglike=ploglike,
                    times=times,
                    tevals=tevals,
                    scal=scal,
                    rho=rho,
                    acr=acr,
                    beta=beta,
                )

            # broadcast beta to everyone to break
            # broadcast beta[istep+1]
            sendbeta = None
            if self.rank_ == 0:
                sendbeta = beta[istep + 1]
            local_beta = self.comm_.bcast(sendbeta, root=0)

            if local_beta >= 1.0:
                break
