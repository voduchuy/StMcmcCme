# coding: utf-8
import numpy as np
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
        self.targcov_ = 1.0
        self.learnlev_ = 1
        self.nchain_ = 1
        self.step_ = 1
        self.modelid_ = 0
        self.maxcount_ = 100

    def Run(
        self,
        logprior_fun: Callable[[np.ndarray], np.ndarray],
        loglike_fun: Callable[[np.ndarray, Any, int], np.ndarray],
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
            Maximum number of tempering and bridging steps.

        maxcount : int
            Maximum number of MCMC iterations in the rejunevation phase.

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

        if start_new:
            lastistep = 0
            thetas = init_thetas
            if self.rank_ == 0:
                ndim = thetas.shape[1]
                tevals = np.zeros(1)
                times = np.zeros(1)
                stheta = np.copy(thetas)
                ploglike = logprior_fun(thetas[:, :, 0])
                betas = np.zeros(1)
                acr = np.zeros(1)
                rho = np.zeros(1)
                scal = np.zeros(1)
                scale = 0.3 * (2.38 ** 2) / ndim
                targacr = 0.25
            else:
                ndim = None

            # send dimension information to everyone
            ndim = self.comm_.bcast(ndim, root=0)

            # scatter initial samples to all processes
            if self.rank_ == 0:
                sendtheta = thetas[:, :, 0]
            else:
                sendtheta = None
            recvtheta = self._scatter_data(sendtheta, "forward")

            # compute on the log-likelihood values on each core
            sendloglike = loglike_fun(recvtheta, data, modelid)

            # Gather back all log-likelihood values at root
            recvloglike = self._scatter_data(sendloglike, "backward")

            if self.rank_ == 0:
                loglike = np.zeros((nsamp, 1))
                loglike[:, 0] = recvloglike[:, 0]
        else:
            # load the theta from last run
            inputdat = np.load(outfile)
            thetas = inputdat["theta"]
            lastistep = thetas.shape[2] - 1
            ndim = None
            # define various variables we will need on the root node
            if self.rank_ == 0:
                loglike = inputdat["loglike"]
                betas = inputdat["beta"]
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

        self.maxcount_ = maxcount
        # make each rank of its own seed
        np.random.seed(int(time.time()) + self.rank_)

        # send dimension information to everyone
        ndim = self.comm_.bcast(ndim, root=0)

        # Target sample correlation for the MCMC step
        target_correlation = 0.6

        if self.comm_.Get_rank() == 0:
            t0 = time.time()

        for istep in range(lastistep, max_num_steps):

            if self.rank_ == 0:
                # if you are root compute the new beta
                beta_next, wi = self._compute_next_beta(
                    loglike[:, istep], betas[istep]
                )
                betas = np.append(betas, beta_next)

                sampmean = np.average(thetas[:, :, istep], 0, wi, False)
                sampcov = np.cov(
                    thetas[:, :, istep], None, False, False, None, None, wi
                )

                saMPIdx = np.random.choice(nsamp, nsamp, True, wi)
                inittheta = thetas[saMPIdx, :, istep]

                otheta = np.copy(inittheta)
                ologlike = loglike[saMPIdx, istep]
                oploglike = ploglike[saMPIdx, istep]

                sacc = 0.0

                start_theta = np.copy(otheta)
                teval = 0

            # each core goes thorugh this while look until we tell it to stop
            r_est = 1.0
            totalcount = 0
            while (r_est > target_correlation) and (
                totalcount < self.maxcount_
            ):
                totalcount = totalcount + 1
                # determine how long a chain to run (nchain) right now we take this as given

                # generate random number seeds (This is now done at the beginning of the code
                # my random numbers are ndim*nsamp*nchain multivariate_normal and nsamp*nchain uniform

                # scatter otheta
                sendotheta = None
                if self.rank_ == 0:
                    sendotheta = otheta
                local_otheta = self._scatter_data(sendotheta, "forward")

                # scatter ologlike
                sendologlike = None
                if self.rank_ == 0:
                    sendologlike = ologlike
                local_ologlike = self._scatter_data(sendologlike, "forward")

                # scatter oploglike
                sendoploglike = None
                if self.rank_ == 0:
                    sendoploglike = oploglike
                local_oploglike = self._scatter_data(sendoploglike, "forward")

                # broadcast scale*sampcov
                local_propcov = np.zeros((ndim, ndim))
                if self.rank_ == 0:
                    local_propcov = scale * sampcov

                self.comm_.Bcast(local_propcov, root=0)
                if ndim == 1:
                    local_propcov = local_propcov.reshape((1,1))

                # broadcast beta[istep+1]
                sendbeta = None
                if self.rank_ == 0:
                    sendbeta = betas[istep + 1]
                local_beta = self.comm_.bcast(sendbeta, root=0)

                # local chain stats
                local_sacc = np.zeros(1, dtype=np.int)
                local_tevals = np.zeros(1, dtype=np.int)

                # run parallel chains to diversify samples
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

                # gather the computed samples, loglikelihood and logprior values to the root process
                otheta = self._scatter_data(local_otheta, "backward")
                ologlike = self._scatter_data(local_ologlike, "backward")
                oploglike = self._scatter_data(local_oploglike, "backward")

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
                    r_est = max(np.diagonal(corrmat, ndim))
                    t1 = time.time() - t0

                    print(str(teval) + " " + str(r_est) + " " + str(t1))

                # broadcast r_est
                sendrest = None
                if self.rank_ == 0:
                    sendrest = r_est
                r_est = self.comm_.bcast(sendrest, root=0)

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
                    + str(betas[istep + 1])
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
                    beta=betas,
                )

            # broadcast beta to everyone to break
            # broadcast beta[istep+1]
            sendbeta = None
            if self.rank_ == 0:
                sendbeta = betas[istep + 1]
            local_beta = self.comm_.bcast(sendbeta, root=0)

            if local_beta >= 1.0:
                break

    def _compute_next_beta(
        self, loglike: np.ndarray, beta_prev: float
    ) -> float:
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
