import unittest
import numpy as np
from scipy.stats import norm, expon

from ParticleFilter import Particles, PeriodicityParticleFilter, Config

if __debug__:
    import matplotlib.pyplot as plt 


def override_initialise_particles(self):
    T_0 = np.array([self.T]*self.K)
    lambda_0 = np.array([self.Lambda]*self.K)
    sigma_0 = np.array([self.Sigma]*self.K)
    x_0 = np.zeros(self.K)
    z_0 = np.zeros(self.K)
    return Particles(T_0, lambda_0, sigma_0, x_0, z_0, self.K)


class TestIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.T = 97
        self.Sigma = 0.01
        self.Lambda = 0.005
        self.N = 50
        self.conf = Config(signal=self.signal_timestamps, num_particles=256, w_mult_decay_rate=10, c=2, T_cauchy_scale=1.0,
                           Lambda_cauchy_scale=0.0001, Sigma_cauchy_scale=0.001, prior_T_scale=5, prior_lambda_scale=0.001)
        
        self.signal_timestamps, self.periodic, self.noise = self.generate_test_signal(
            self.T, self.Sigma, self.Lambda, self.N)

        self.target_relative_error = 0.07288346629199467
        self.target_dcml_place = 10

    def generate_test_signal(self, T, Sigma, Lambda, N):
        np.random.seed(1)
        x = 0
        z = 0

        # Test parameter values
        periodic_series = []
        noise_series = []
        for _ in range(N):
            # sample periodic signal timestamp
            while True:
                # N.B. ensure our new sample xnew > x
                # otherwise sample again until we satisfy this.
                dper = norm.rvs(loc=T, scale=Sigma, size=1)[0]
                xnew = x + dper
                if (xnew > x):
                    x = xnew
                    periodic_series.append(xnew)
                    break

        while True:
            dnoise = expon.rvs(loc=0, scale=1/Lambda, size=1)[0]
            z += dnoise
            if z > periodic_series[-1]:
                break
            noise_series.append(z)

        signal_timestamps = sorted(periodic_series + noise_series)

        return signal_timestamps, periodic_series, noise_series

    def test_model(self):
        # np.random.seed(1)

        # PeriodicityParticleFilter.initialise_samples = override_initialise_particles
        model = PeriodicityParticleFilter(self.conf)

        res = model.fit()

        idx = -1

        h = model.debugInfo.h_series[idx]
        lw = model.debugInfo.L_series[idx]

        predicted_T = np.sum(np.array(h.T) * np.array(lw))
        actual_T = self.T
        actual_sigma = self.Sigma
        actual_lambda = self.Lambda
        relative_error = np.abs(np.log(predicted_T / actual_T))
        predicted_lambda = np.sum(np.array(h.Lambda) * np.array(lw))
        predicted_sigma = np.sum(np.array(h.Sigma) * np.array(lw))

        self.assertAlmostEqual(relative_error , self.target_relative_error, self.target_dcml_place)
        
        fig, ax = plt.subplots(7, figsize=(10, 20))
        ax[0].scatter(h.T, lw, alpha=0.2)
        ax[0].vlines(actual_T, ymin=0, ymax=max(lw), color='r')
        ax[1].scatter(h.Sigma, lw, alpha=0.2)
        ax[1].vlines(actual_sigma, ymin=0, ymax=max(lw), color='r')
        ax[2].scatter(1/h.Lambda, lw, alpha=0.2)
        ax[2].vlines(1/actual_lambda, ymin=0, ymax=max(lw), color='r')
        ax[2].set_xscale('log')
        ax[3].scatter(h.X, lw, alpha=0.2)
        ax[4].scatter(h.Z, lw, alpha=0.2)
        ax[5].hist(h.Lambda * h.T, alpha=0.2, bins=250)
        ax[6].hist(lw, density=True, bins=50)
        ax[0].set_title('T distribution (log10)')
        ax[1].set_title('Sigma distribution (log10)')
        ax[2].set_title('1/Lambda distribution (log10)')
        ax[3].set_title('X distribution')
        ax[4].set_title('Z distribution')
        ax[5].set_title('Lambda x T distribution')
        ax[6].set_title('weights distribution')
        ax[0].set_xscale('log')
        ax[5].set_xscale('log')
        ax[5].set_yscale('log')
        for axis in ax:
            axis.grid()

        predicted_T_change = []
        predicted_lambda_change = []
        predicted_sigma_change = []
        relative_error_change = []
        for h, lw in zip(model.debugInfo.h_series, model.debugInfo.L_series):
            predicted_T = np.sum(np.array(h.T) * np.array(lw))
            predicted_lambda = np.sum(np.array(h.Lambda * np.array(lw)))
            predicted_sigma = np.sum(np.array(h.Sigma * np.array(lw)))
            actual_T = self.T
            relative_error = np.abs(np.log(predicted_T / actual_T))
            predicted_T_change.append(predicted_T)
            predicted_lambda_change.append(predicted_lambda)
            predicted_sigma_change.append(predicted_sigma)
            relative_error_change.append(relative_error)

        fig, ax = plt.subplots(4, figsize=(10, 10), sharex=True)
        ax[0].plot(predicted_T_change)
        ax[0].hlines(actual_T, xmin=0, xmax=len(predicted_T_change))
        ax[1].plot(relative_error_change)
        ax[2].plot(predicted_lambda_change)
        ax[2].hlines(actual_lambda, xmin=0, xmax=len(predicted_lambda_change))
        ax[3].plot(predicted_sigma_change)
        ax[3].hlines(actual_sigma, xmin=0, xmax=len(predicted_sigma_change))
        ax[0].set_title('Predicted T per iteration')
        ax[1].set_title('Relative error per iteration for T')
        ax[2].set_title('Predicted Lambda per iteration')
        ax[3].set_title('Predicted Sigma per iteration')
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')
        ax[2].set_yscale('log')
        ax[3].set_yscale('log')
        for axis in ax:
            axis.grid()
