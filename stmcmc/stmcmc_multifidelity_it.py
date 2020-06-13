# coding: utf-8
import numpy as np
import time
from mpi4py import MPI
from typing import Tuple


class StMcmcMultiFidelityIT:
    def __init__(self, comm=MPI.COMM_WORLD):
        self.comm_ = comm.Dup()
        self.num_procs_ = comm.Get_size()
        self.rank_ = comm.Get_rank()
        self.nsamp_ = 1044
        self.ncores_ = self.num_procs_
        self.max_num_steps_ = 1000
        self.targcov_ = 1.0
        self.learnlev_ = 1
        self.nchain_ = 1
        self.step_ = 1
        self.logprior_ = []
        self.loglike_ = []
        self.maxcount_ = 100

    def Run(
        self,
        logprior_fun,
        loglike_fun,
        data,
        init_thetas,
        outfile=None,
        nsamp=128,
        max_level=0,
        max_num_steps=1000,
        maxcount=100,
    ):
        if outfile is None:
            outfile = "stmcmc_test.npz"

        if type(outfile) != str:
            raise RuntimeError("Output filename must be a string!")

        theta = init_thetas
        self.nsamp_ = nsamp
        self.max_num_steps_ = max_num_steps
        self.maxcount_ = maxcount

        # make each rank of its own seed
        np.random.seed(int(time.time()) + self.rank_)

        self.logprior_ = logprior_fun
        self.loglike_ = loglike_fun

        if self.rank_ == 0:
            tevals = np.zeros(1)
            times = np.zeros(1)
            modelused = np.zeros(1)
            t0 = time.time()

        # if you are rank 0 generate the parameters and evaluate the prior
        ndim = None
        sendtheta = None
        counts = None
        dspls = None
        modelid = None

        if self.rank_ == 0:
            # theta = prior_gen(nsamp)
            sendtheta = theta[:, :, 0]
            stheta = np.copy(theta)
            ploglike = self.logprior_(theta[:, :, 0])
            ndim = theta.shape[1]
            counts = (
                ndim
                * nsamp
                // self.num_procs_
                * np.ones(self.num_procs_, dtype=int)
            )
            dspls = range(0, nsamp * ndim, ndim * nsamp // self.num_procs_)
            modelid = 0

        # send dimension information to everyone
        ndim = self.comm_.bcast(ndim, root=0)

        # send model id to start with
        modelid = self.comm_.bcast(modelid, root=0)

        # holder for everyone to recive there part of the theta vector
        recvtheta = np.zeros((nsamp // self.num_procs_, ndim))

        # get your theta values to use for computing th loglike
        self.comm_.Scatterv(
            [sendtheta, counts, dspls, MPI.DOUBLE], recvtheta, root=0
        )

        # compute on the loglike on each core
        sendloglike = self.loglike_(recvtheta, data, modelid)

        scounts = nsamp // self.num_procs_
        rcounts = nsamp // self.num_procs_ * np.ones(self.num_procs_, dtype=int)
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

        # define various variables we will need on the root node
        if self.rank_ == 0:
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
        for istep in range(0, self.max_num_steps_):
            oldmodel = modelid
            # before we start lets have everyone compute the full model so we can look at whether we need to change
            if modelid != max_level:
                sendtheta = None
                counts = None
                dspls = None

                if self.rank_ == 0:
                    sendtheta = np.copy(theta[:, :, istep])
                    counts = (
                        ndim
                        * nsamp
                        // self.num_procs_
                        * np.ones(self.num_procs_, dtype=int)
                    )
                    dspls = range(
                        0, nsamp * ndim, ndim * nsamp // self.num_procs_
                    )

                # holder for everyone to recive there part of the theta vector
                recvtheta = np.zeros((nsamp // self.num_procs_, ndim))

                # get your theta values to use for computing th loglike
                self.comm_.Scatterv(
                    [sendtheta, counts, dspls, MPI.DOUBLE], recvtheta, root=0
                )

                # compute on the loglike on each core using the find modelid
                sendloglike = self.loglike_(recvtheta, data, max_level)

                scounts = nsamp // self.num_procs_
                rcounts = (
                    nsamp
                    // self.num_procs_
                    * np.ones(self.num_procs_, dtype=int)
                )
                rdspls = range(0, nsamp, nsamp // self.num_procs_)

                # send core information to the root
                fullloglike = None
                if self.rank_ == 0:
                    fullloglike = np.zeros(nsamp)

                self.comm_.Gatherv(
                    [sendloglike, scounts, MPI.DOUBLE],
                    [fullloglike, rcounts, rdspls, MPI.DOUBLE],
                    root=0,
                )

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

                if modelid != max_level:
                    # Determine if we need to change models based upon the full likeihood and stuff
                    b0 = beta[istep]
                    db0 = dbeta
                    loglike0 = np.copy(loglike[:, istep])
                    loglikek = np.copy(fullloglike)

                    pkb0 = np.exp(loglikek - max(loglikek))
                    pb0 = np.exp(b0 * loglike0 - max(b0 * loglike0))
                    pdb0 = np.exp(db0 * loglike0 - max(db0 * loglike0))
                    mdb0 = np.mean(pdb0)

                    if (
                        np.mean(
                            np.multiply(
                                np.divide(pkb0, pb0), np.log(pdb0 / mdb0)
                            )
                        )
                        < 0
                    ):
                        modelid = modelid + 1
                        beta[istep + 1] = beta[istep]

            # recompute the likeihood with the new model if needed
            modelid = self.comm_.bcast(modelid, root=0)
            if modelid != oldmodel:
                sendtheta = None
                counts = None
                dspls = None

                if self.rank_ == 0:
                    sendtheta = np.copy(theta[:, :, istep])
                    counts = (
                        ndim
                        * nsamp
                        // self.num_procs_
                        * np.ones(self.num_procs_, dtype=int)
                    )
                    dspls = range(
                        0, nsamp * ndim, ndim * nsamp // self.num_procs_
                    )

                # holder for everyone to recive there part of the theta vector
                recvtheta = np.zeros((nsamp // self.num_procs_, ndim))

                # get your theta values to use for computing th loglike
                self.comm_.Scatterv(
                    [sendtheta, counts, dspls, MPI.DOUBLE], recvtheta, root=0
                )

                # compute on the loglike on each core using the find modelid
                sendloglike = self.loglike_(recvtheta, data, modelid)

                scounts = nsamp // self.num_procs_
                rcounts = (
                    nsamp
                    // self.num_procs_
                    * np.ones(self.num_procs_, dtype=int)
                )
                rdspls = range(0, nsamp, nsamp // self.num_procs_)

                # send core information to the root
                newloglike = None
                if self.rank_ == 0:
                    newloglike = np.zeros(nsamp)

                self.comm_.Gatherv(
                    [sendloglike, scounts, MPI.DOUBLE],
                    [newloglike, rcounts, rdspls, MPI.DOUBLE],
                    root=0,
                )

                # find write over the loglikes and update weights
                if self.rank_ == 0:
                    maxl = max(
                        beta[istep + 1] * (newloglike - loglike[:, istep])
                    )
                    wi = np.divide(
                        np.exp(
                            beta[istep + 1] * (newloglike - loglike[:, istep])
                            - maxl
                        ),
                        sum(
                            np.exp(
                                beta[istep + 1]
                                * (newloglike - loglike[:, istep])
                                - maxl
                            )
                        ),
                    )
                    loglike[:, istep] = newloglike

            if self.rank_ == 0:
                sampmean = np.average(theta[:, :, istep], 0, wi, False)
                sampcov = np.cov(
                    theta[:, :, istep], None, False, False, None, None, wi
                )

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
                    cploglike = self.logprior_(ctheta)[:, 0]
                    cloglike = self.loglike_(ctheta, data, modelid)[:, 0]

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
                modelused = np.append(modelused, modelid)
                times = np.append(times, time.time() - t0)

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
                    + str(modelid)
                    + " "
                    + str(time.time() - t0)
                )

                stheta = np.dstack((stheta, start_theta))
                theta = np.dstack((theta, otheta))
                loglike = np.column_stack((loglike, ologlike))
                ploglike = np.column_stack((ploglike, oploglike))

                np.savez(
                    outfile,
                    stheta=stheta,
                    theta=theta,
                    loglike=loglike,
                    ploglike=ploglike,
                    times=times,
                    tevals=tevals,
                    scal=scal,
                    rho=rho,
                    acr=acr,
                    beta=beta,
                    modelused=modelused,
                )

            # broadcast beta to everyone to break
            # broadcast beta[istep+1]
            sendbeta = None
            if self.rank_ == 0:
                sendbeta = beta[istep + 1]
            local_beta = self.comm_.bcast(sendbeta, root=0)

            if local_beta >= 1.0:
                break

    def _compute_next_beta(
        self, loglike: np.ndarray, beta_prev: float
    ) -> Tuple[float, np.ndarray]:
        """
        Helper function to determine the next inverse temperature.

        Parameters
        ----------
        loglike : 1-d numpy array
            loglikelihood values of current samples.

        beta_prev : float
            previous value of beta (i.e., the inverse temperature).

        Returns
        -------

        beta_next : float
            the next value for the inverse temperature.

        """
        nsamp = len(loglike)

        dbeta0 = 0.0
        dbeta1 = 1.0

        maxl = max(loglike)
        odbeta = 10.0
        ddbeta = odbeta

        while ddbeta > (10.0 ** (-10.0)):
            dbeta = (dbeta0 + dbeta1) / 2.0

            wi = np.divide(
                np.exp(dbeta * (loglike - maxl)),
                sum(np.exp(dbeta * (loglike - maxl))),
            )
            fi = (
                np.sqrt(1.0 / nsamp * sum(np.power(wi - np.mean(wi), 2)))
                / np.mean(wi)
                - self.targcov_
            )

            if fi <= 0.0:
                dbeta0 = dbeta
            else:
                dbeta1 = dbeta

            ddbeta = np.abs(odbeta - dbeta)
            odbeta = dbeta

        if beta_prev + dbeta >= 1.0:
            dbeta = 1.0 - beta_prev
            wi = np.divide(
                np.exp(dbeta * (loglike - maxl)),
                sum(np.exp(dbeta * (loglike - maxl))),
            )
            beta_next = 1.0
        else:
            beta_next = beta_prev + dbeta
        return beta_next, wi

    def _scatter_data(self, send_data: np.ndarray, mode: str = "forward"):
        """
        Convenient method to scatter/gather data from root to other processors. This is required at various steps of the sampling process.

        Parameters
        ----------

        send_data : numpy array
            data to be communicated. Must follow 'C' ordering. The first dimension must be the 'batch' dimension,
            i.e.,
            data.shape[0] is
            the number of entities (samples, numbers,...) to be communicated across processors.

        mode : str (default: "forward")
            if "forward", scatter data from root to other processors. Otherwise, gather data from all processes
            into the root.

        Returns
        -------

        recv_data: numpy array
            data received on each proces.

        """
        comm = self.comm_
        rank = comm.Get_rank()
        num_procs = comm.Get_size()

        if mode == "forward":
            if rank == 0:
                num_samples_per_cpu = send_data.shape[0]
                data_type = send_data.dtype
                if len(send_data.shape) == 1:
                    sample_dim = 1
                    sample_shape = ()
                else:
                    sample_dim = int(np.prod(send_data.shape[1:]))
                    sample_shape = list(send_data.shape[1:])
            else:
                num_samples_per_cpu = None
                sample_dim = None
                data_type = None
                sample_shape = None
            sample_dim = comm.bcast(sample_dim)
            data_type = comm.bcast(data_type)
            num_samples_per_cpu = comm.bcast(num_samples_per_cpu)
            sample_shape = tuple(comm.bcast(sample_shape))

            counts = None
            displacements = None
            if rank == 0:
                counts = (
                    sample_dim
                    * (num_samples_per_cpu // num_procs)
                    * np.ones(num_procs, dtype=int)
                )
                displacements = np.zeros((num_procs,), dtype=int)
                displacements[1:] = np.cumsum(counts[0:-1])

            recv_data = np.zeros(
                (num_samples_per_cpu // num_procs,) + sample_shape,
                dtype=data_type,
            )

            comm.Scatterv(
                [
                    send_data,
                    counts,
                    displacements,
                    MPI._typedict[data_type.char],
                ],
                recv_data,
                root=0,
            )
        else:  # gather mode
            num_samples_per_cpu = send_data.shape[0]
            data_type = send_data.dtype
            if len(send_data.shape) == 1:
                sample_shape = ()
                sample_dim = 1
            else:
                sample_shape = send_data.shape[1:]
                sample_dim = int(np.prod(send_data.shape[1:]))

            sample_dim = comm.bcast(sample_dim)
            data_type = comm.bcast(data_type)

            send_counts = num_samples_per_cpu * sample_dim

            if rank == 0:
                receive_counts = (
                    num_samples_per_cpu
                    * sample_dim
                    * np.ones(num_procs, dtype=int)
                )
                receive_displacements = np.zeros((num_procs,), dtype=int)
                receive_displacements[1:] = np.cumsum(receive_counts[0:-1])
            else:
                receive_counts = None
                receive_displacements = None

            # send core information to the root
            recv_data = None
            if rank == 0:
                recv_data = np.zeros(
                    (num_samples_per_cpu * num_procs,) + sample_shape,
                    dtype=data_type,
                )

            comm.Gatherv(
                [send_data, send_counts, MPI._typedict[data_type.char]],
                [
                    recv_data,
                    receive_counts,
                    receive_displacements,
                    MPI._typedict[data_type.char],
                ],
                root=0,
            )

        return recv_data
