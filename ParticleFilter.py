import copy
import numpy as np
from enum import Enum
from typing import ClassVar
from dataclasses import dataclass
from scipy.stats import norm, expon


@dataclass
class Config:
    '''
    Configuration object for periodicity particle filter
    Args: 
        signal : list 
            the signal to be analyse

        num_particles : int 
            number of particles to use

        c : float
            ratio of the numbr of noise events to the number of periodic events. Used 
            for decaying the likelihood weights for particles that have a high noise rate 
            compared to the periodic signal, for which any useful inference would be likely
            to be elusive. 

        prior_T_scale : float 
            the hyperparameter to be used for the exponential distribution 
            that is used as the prior distribution for period T. 

        prior_lambda_scale : float 
            the hyperparameter to be used for the exponential distribution 
            that is used as the prior distribution for noise event rate Lambda.

        x_0 : float 
            initial value for periodic event variable

        z_0 : float
            initial value for noise event variable

        T_cauchy_scale : float 
            scale parameter for cauchy distribution used as the hypothesis distribution

        Lambda_cauchy_scale : float 
            scale parameter for cauchy distribution used as the hypothesis distribution

        Sigma_cauchy_scale : float 
            scale parameter for cauchy distribution used as the hypothesis distribution

        w_mult_decay_rate : float
            decay rate for particle reweighting function for those lambda x T > c. 
    '''
    signal: list
    num_particles: int = 256
    c: int = 2
    prior_T_scale: float = 1.0
    prior_lambda_scale: float = 1.0
    x_0: float = 0.0
    z_0: float = 0.0
    T_cauchy_scale: float = 1.0
    Lambda_cauchy_scale: float = 1.0
    Sigma_cauchy_scale: float = 1.0
    w_mult_decay_rate: float = 1.0

    def __post_init__(self):
        self.num_timesteps = len(self.signal)


class ModelDebugInfo():

    def __init__(self):
        self.h_series = []  # particles
        self.L_series = []  # likelihood weights


@dataclass
class Particles:
    '''
    Dataclass for carrying around each variable defined in particle vector 
    h = [T, lambda, sigma, xhat, zhat]^t. 

    Args: 
        T: 
        Lambda: Rate parameter for false positive signals due to noise. 
        Sigma:
        X: timestamp of last identified signal generated by periodic signal process
        Z: timestamp of last identified signal generated by false positive noise process
        NumParticles: defaults to length of parameter vectors. Executes length typecheck. 
    '''
    ### public ###
    T: "np.ndarray"
    Lambda: "np.ndarray"
    Sigma: "np.ndarray"
    X: "np.ndarray"
    Z: "np.ndarray"
    NumParticles: int

    ### private class var ###

    # number of params we use in our model
    _NumParams: ClassVar[int] = 5

    class VariableIndex(Enum):
        T = 0
        Lambda = 1
        Sigma = 2
        X = 3
        Z = 4

    def __post_init__(self):
        self._X_last = copy.deepcopy(self.X)
        self._Z_last = copy.deepcopy(self.Z)
        self._assert_ctor_params_ndarray()
        self._assert_ctor_params_same_len()

    def _assert_ctor_params_ndarray(self):
        '''enforce type checking as a simple list does not support broadcasting
        user must provide numpy arrays'''
        assert isinstance(self.T, np.ndarray), "T is not a numpy array"
        assert isinstance(
            self.Lambda, np.ndarray), "Lambda is not a numpy array"
        assert isinstance(self.Sigma, np.ndarray), "Sigma is not a numpy array"
        assert isinstance(self.X, np.ndarray), "X is not a numpy array"
        assert isinstance(self.Z, np.ndarray), "Z is not a numpy array"

    def _assert_ctor_params_same_len(self):

        is_same_len = (len(self.T)
                       == len(self.Lambda)
                       == len(self.Sigma)
                       == len(self.X)
                       == len(self.Z)
                       == self.NumParticles)

        assert is_same_len, f"Input vectors are not of the same size. Size specified is {self.NumParticles}"

    def __len__(self):
        return self.NumParticles

    def __eq__(self, other):
        return np.array_equal(self.T, other.T) \
            and np.array_equal(self.Lambda, other.Lambda) \
            and np.array_equal(self.Sigma, other.Sigma) \
            and np.array_equal(self.X, other.X) \
            and np.array_equal(self.Z, other.Z)

    @staticmethod
    def _indexing_key_typecheck(key):
        if not (isinstance(key, int) or isinstance(key, slice) or isinstance(key, np.ndarray)):
            raise TypeError(
                f"Particles object behaves as a 1-D array or list. Invalid index of type {type(key)}")

    def _indexing_out_of_bounds_check(self, key):
        if (isinstance(key, slice)):
            if key.stop > self.NumParticles - 1:
                raise IndexError(
                    f"Particles object contains {self.NumParticles} particles. Cannot retrieve items for indices {key.start}:{key.stop}")

        elif isinstance(key, int):
            if (key > self.NumParticles - 1):
                raise IndexError(
                    f"Particles object contains {self.NumParticles} particles. Cannot retrieve items for index {key}")

    def __getitem__(self, key):
        self._indexing_key_typecheck(key)
        self._indexing_out_of_bounds_check(key)

        t = self.T[key]
        l = self.Lambda[key]
        s = self.Sigma[key]
        x = self.X[key]
        z = self.Z[key]

        if isinstance(key, int):
            N = 1
            # input needs to be np array
            t, l, s, x, z = np.array([t]), np.array(
                [l]), np.array([s]), np.array([x]), np.array([z])
        else:
            N = len(t)

        return Particles(t, l, s, x, z, N)

    def __setitem__(self, key, value: "Particles"):
        self._indexing_key_typecheck(key)
        self._indexing_out_of_bounds_check(key)

        if not isinstance(value, Particles):
            raise TypeError(
                f"Value to be set should be Particles object, not {type(value)}")

        self.T[key] = value.T
        self.Lambda[key] = value.Lambda
        self.Sigma[key] = value.Sigma
        self.X[key] = value.X
        self.Z[key] = value.Z

    def __delitem__(self, key):
        self._indexing_key_typecheck(key)
        self._indexing_out_of_bounds_check(key)
        self.T.__delitem__(key)
        self.Lambda.__delitem__(key)
        self.Sigma.__delitem__(key)
        self.X.__delitem__(key)
        self.Z.__delitem__(key)

    def __iter__(self):
        for idx in range(self.NumParticles):
            yield Particles(self.T[idx],
                            self.Lambda[idx],
                            self.Sigma[idx],
                            self.X[idx],
                            self.Z[idx],
                            1)

    @property
    def shape(self):
        return (self._NumParams, self.NumParticles)

    @property
    def X_last(self):
        return self._X_last

    @X_last.setter
    def X_last(self, val):
        '''enforce type checking as a simple list does not support broadcasting
        user must provide numpy arrays'''
        assert isinstance(val, np.ndarray), "X_last is not a numpy array"
        self._X_last = val

    @property
    def Z_last(self):
        return self._Z_last

    @Z_last.setter
    def Z_last(self, val):
        '''enforce type checking as a simple list does not support broadcasting
        user must provide numpy arrays'''
        assert isinstance(val, np.ndarray), "X_last is not a numpy array"
        self._Z_last = val

    def as_matrix(self):
        '''Return np.ndarray of vector h = [T, lambda, sigma, xhat, zhat]^t
        of shape (5, num_particles)'''
        return np.stack((self.T, self.Lambda, self.Sigma, self.X, self.Z))


class PeriodicityParticleFilter:
    def __init__(self, config: Config):
        self.config = config
        self.y = config.signal
        self.num_timesteps = self.config.num_timesteps
        self.K = self.config.num_particles
        self.s_T = self.config.prior_T_scale
        self.s_lambda = self.config.prior_lambda_scale
        self.s_cauchy_T = np.broadcast_to(
            self.config.T_cauchy_scale, self.K)  # shape = (num_particles,)
        self.s_cauchy_Sigma = np.broadcast_to(
            self.config.Sigma_cauchy_scale, self.K)  # shape = (num_particles,)
        self.s_cauchy_Lambda = np.broadcast_to(
            self.config.Lambda_cauchy_scale, self.K)  # shape = (num_particles,)
        self.c = self.config.c
        self.w_mult_decay_rate = self.config.w_mult_decay_rate

        self._h = None
        self._iter_cnt = 0

        self.debugInfo = ModelDebugInfo()

    def signal(self):
        '''
        Generator method for iterating through the signal. 
        Returns the next value in the signal. 
        '''
        while self._iter_cnt < self.num_timesteps:
            yield self.y[self._iter_cnt]
            self._iter_cnt += 1

    def initialise_samples(self) -> Particles:
        '''
        Sample K particles from prior distributions for h_0
        h = [T, lambda, sigma, x_0, z_0] 
        where: 
            T ~ exponential(scale) 
            lambda ~ exponential(scale)
            sigma ~ Uniform(0, T)
            x_0 = zero
            z_0 = zero
        '''
        T_0 = np.random.exponential(self.s_T, self.K)
        lambda_0 = np.random.exponential(self.s_lambda, self.K)
        sigma_0 = np.random.rand(self.K) * T_0
        x_0 = np.zeros(self.K)
        z_0 = np.zeros(self.K)
        return Particles(T_0, lambda_0, sigma_0, x_0, z_0, self.K)

    @staticmethod
    def sample_cauchy_1D(loc, scale):
        hnew_std = np.random.standard_cauchy(len(loc))
        hnew = scale * hnew_std + loc
        return hnew

    def sample_cauchy_1D_gt_zero(self, loc, scale):
        '''
        Parameters
        ----------
        loc : mean values. 
        scale : standard deviation. 

        Raises
        ------
        ValueError
            If the shape of loc and scale do not match. It wil not 
            attempt to broadcast input array shapes. 
        '''
        if (np.shape(loc) != np.shape(scale)):
            raise ValueError(
                f'loc and scale lengths do not match. {np.shape(loc)} != {np.shape(scale)}')
        samples = self.sample_cauchy_1D(loc, scale)
        while True:
            idx = np.where(samples < 0)[0]
            if len(idx) == 0:
                break
            resample_locs = loc[idx]
            resample_scale = scale[idx]

            new_samples = self.sample_cauchy_1D(resample_locs, resample_scale)

            samples[idx] = new_samples

        return samples

    def sample_event_provenance(self, y, h: Particles):
        '''
        Sample event provenance r_i. 
        Args: 
            h: Particles object, containing particles (or hypotheses) used in our model.

        Returns: 
            r: r values sampled from bernoulli distribution. Size is equal to number of 
            particles. 

        Calculate bernoulli distribution, p(r_i = val | y_i, h_i), using conditional probabilities:

        p(ri = 1 | yi, hi) = PeriodicSignalLikelihood / (PeriodicSignalLikelihood + FalsePositiveLikelihood)
        p(ri = 0 | yi, hi) = 1 - p(ri = 1 | yi, hi)
        '''

        logps = self.log_periodic_likelihood(
            y, h.T, h.Sigma, h.Lambda, h.X, h.Z)
        logfp = self.log_fp_likelihood(y, h.T, h.Sigma, h.Lambda, h.X, h.Z)

        # p = ps / (ps + fp)
        logp = logps - np.logaddexp(logps, logfp)

        # single trial binomial = bernoulli
        N = 1
        r = np.random.binomial(N, np.exp(logp))

        return r

    def update_hypothesis(self, y, h: Particles):
        '''
        Sample next set of particles from hypothesis distribution p(h_i|h_{i-1}).
        For the ith timestep, let j denote the jth particle:
        h^j = [T^j, lambda^j, sigma^j, x_j, z_j]

        where each variable in vector h_j are sampled from a cauchy distribution centred
        about the previous particle value. 
        '''
        # sample T, lambda, sigma from cauchy
        h.T = self.sample_cauchy_1D_gt_zero(h.T, self.s_cauchy_T)
        h.Sigma = self.sample_cauchy_1D_gt_zero(h.Sigma, self.s_cauchy_Lambda)
        h.Lambda = self.sample_cauchy_1D_gt_zero(h.Lambda, self.s_cauchy_Sigma)

        h.X_last = copy.deepcopy(h.X)
        h.Z_last = copy.deepcopy(h.Z)

        rsampled = self.sample_event_provenance(y, h)
        r_is_1_idx, r_is_0_idx = np.nonzero(rsampled), np.nonzero(1-rsampled)
        h.X[r_is_1_idx] = y
        h.Z[r_is_0_idx] = y

        return h

    @staticmethod
    def log_periodic_likelihood(y: float,
                                T: np.array,
                                Sigma: np.array,
                                Lambda: np.array,
                                X: np.array,
                                Z: np.array):
        '''
        The likelihood of the signal being generated by the signal process. 
        Returns log likeliood for floating point precision stability. 

        L = p(yi, ri=1|hi) = p(yi | hi, ri = 1) x (1 - p(ri = 0)) 
        = N(Ti + x_{i-1}, sigma_i) * (1 - F(yi - z_{i-1}; exp(lambda_i)))

        Args: 
            y : ith event timestamp
            h : particle object containing current parameter hypotheses 
        '''
        # p(yi | hi, ri=1) = N(Ti + x_{i-1}, sigma_i)
        # scale = standard deviation, not variance.
        logp1 = norm.logpdf(y, T + X, Sigma)

        # 1 - F(yi - z_{i-1}; exp(lambda_i)); F_exp(x; lambda) = 1 - exp(-lamba * x)
        # therefore, 1 - F(yi - z_{i-1}; exp(lambda_i)) = exp(-lambda * (yi - z_{i-1}))
        logp2 = -Lambda * (y - Z)

        return logp1 + logp2

    @staticmethod
    def log_fp_likelihood(y: float,
                          T: np.array,
                          Sigma: np.array,
                          Lambda: np.array,
                          X: np.array,
                          Z: np.array):
        '''
        The likelihood of the signal being a false positive generated by noise. 
        Returns log likeliood for floating point precision stability. 

        L = p(yi, ri = 0 | hi) = p(i | hi, ri = 0) x p(ri = 0) 
        = f(yi - z_{i-1}; exp(lambda_i)) * (1 - F(yi; N(x_{i-1} + Ti, sigma_i)))

        Args: 
            y: ith event timestamp
            h: particle object containing current parameter hypotheses. 
        '''
        # f(yi - z_{i-1}; exp(lambda_i)), expon takes parameter scale = 1/lambda
        logp1 = expon.logpdf(x=y - Z, loc=0, scale=1/Lambda)

        # 1 - Phi(y; x_{i-1} + Ti, sigma_i)
        # p2 = 1 - norm.cdf(y, loc = h.T + h.X, scale = h.Sigma)
        # survival function sf = 1 - cdf.
        logp2 = norm.logsf(y, loc=T + X, scale=Sigma)

        return logp1 + logp2

    def log_likelihood_weighting(self, y, h: Particles):
        '''
        Returns likelihood weighting for each particle. 

        '''
        # Lper = p(yi, ri=1 | hi)
        LogLperiodic = self.log_periodic_likelihood(
            y, h.T, h.Sigma, h.Lambda, h.X_last, h.Z_last)

        # Lfp = p(yi, ri0 | hi)
        LogLfp = self.log_fp_likelihood(
            y, h.T, h.Sigma, h.Lambda, h.X_last, h.Z_last)

        # L = p(yi | hi) = p(yi, ri=0 | hi) + p(yi, ri=1 | hi)
        # Use LogL = Log(exp(LogX) + exp(LogY))
        LogL = np.logaddexp(LogLperiodic, LogLfp)

        return LogL

    def resample_particles(self, h: Particles, w: np.ndarray):
        '''
        Resample particles, h, with probabilities according to their likelihood weights, w. 
        The number of particles resampled, k, is equal to the number of particles present in the 
        Particles object (equivalent to len(h)). k is equivalent to self.K, or config.num_particles. 

        Args: 
            h: Particles object
            w: np.array of length equal to the number of particles

        Raises ValueError if length of w does not equal to the number of particles. 
        Raises ValueError if length of h is not equal to self.K (which is eqv to config.num_particles).
        '''
        num_h = len(h)

        if (num_h != len(w)):
            msg = f"Iteration {self._iter_cnt} : Number of particles must match length of array of weights." + \
                f"Received Particles object of length {num_h} but likelihood weights given is " + \
                f"of length {len(w)}"
            raise ValueError(msg)

        if (num_h != self.K):
            msg = f"Iteration {self._iter_cnt} : Number of particles specified for this model is {self.K}, but Particles object" + \
                f" given has {num_h} particles."
            raise ValueError(msg)

        idx = np.random.choice(a=num_h, size=num_h, replace=True, p=w)
        h = h[idx]

        return h

    def get_log_reweight_multiplier(self, h, c):
        '''
        We forcfully decay the weights for particles where lambda x T > c.
        w = exp(-decay_rate * x)
        log(w) = -decay_rate * x
        '''
        lambda_thresh = c / h.T
        x = np.maximum(h.Lambda - lambda_thresh, 0)
        logy = -self.w_mult_decay_rate * x

        return logy

    def get_reweight_multiplier(self, h, c):
        '''
        We want to penalise 

        '''
        lambda_thresh = c / h.T
        x = np.maximum(h.Lambda - lambda_thresh, 0)
        y = np.exp(-self.w_mult_decay_rate * x)
        return y

    def fit(self):
        '''Run the particle filter on the input data.'''

        # initialise K particles by sampling from prior distributions
        # over hypothesis parameters.
        self._h = self.initialise_samples()

        for y in self.signal():
            # sample next set of particles from hypothesis distribution p(h_i|h_{i-1})
            self._h = self.update_hypothesis(y, self._h)
            self.debugInfo.h_series.append(copy.deepcopy(self._h))

            # likelihood weighting - compute likeliness of each particle
            loglw = self.log_likelihood_weighting(y, self._h)
            reweight_mult = self.get_log_reweight_multiplier(self._h, self.c)
            reweighted_loglw = loglw + reweight_mult

            # normalise likelihood weights
            # To prevent overflow, we do the following transformation:
            # We need to remind ourselves that resampling is done based on
            # relative weight sizes.
            # So, we subtract the max(log-likelihood) i.e. equiv to likelihood / max(likelihood)
            # then we convert back from log domain to normalise weights.
            rel_loglw = reweighted_loglw - max(reweighted_loglw)
            lw = np.exp(rel_loglw)
            lw_norm = lw / np.sum(lw)

            self.debugInfo.L_series.append(lw_norm)

            # resample
            self._h = self.resample_particles(self._h, lw_norm)

        return self._h
