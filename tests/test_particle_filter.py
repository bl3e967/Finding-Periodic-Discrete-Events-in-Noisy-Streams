import scipy
import unittest
import numpy as np
from ParticleFilter import Particles, PeriodicityParticleFilter, Config

### unit tests ## 

# Test function - get_periodic_likelihood

class TestParticleFilter(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_periodic_likelihood_no_noise(self):
        '''
        Test case: 

        Let's have the signal as strictly periodic where the true
        period T = 100;
        
        i.e. signal timestamps = 0, 100, 200, 300, 400, 500,...

        Here, true lambda = 0 (rate of false positive = 0 as we have none)
        and true sigma = 0 (signal is strictly periodic with no uncertainty)

        For our particle vector h = [T, lambda, sigma, X, Z], let's have the following samples: 

        For this test case, y_{i-1} = 100, y_i = 200

        particles for test scenario: 

        h1 = [T = 100, lambda = 0, sigma = 0, x_last = 100, z_last = 0]
        This should return NaN. T = 100 is correct with zero uncertainty. sigma=0, so gaussian pdf 
        would have an infinite pdf value. Lambda = 0 so correct that we have no noise, and we also
        correctly know no noise has been found yet, z_last = 0. 

        h2 = [T = 100, lambda = 0, sigma = 0.01, x_last = 100, z_last = 0]
        Same as test 1, but now with non-zero uncertainty. This should return a non-NaN likelihood value 
        as the gaussian no longer has a divide by zero. Exact value is: 39.89422804
        
        h3 = [T = 1, lambda = 0, sigma = 0.01, x_last = 100, z_last = 0]
        T is way off, with small uncertainty. x_last = 100 while current value is 200. 
        This means likelihood of the observation generated by a periodic signal according to particle h3 is zero. 
        Again, noise process param lambda is set to zero. 

        h4 = [T = 90, lambda = 0, sigma = 25, x_last = 100, z_last = 0]
        T is a bit off, but sigma = 25 so that actual T falls within 1 std dev of our estimate. 
        Likelihood should be moderate value. Actual value is 0.014730805
        '''
        y_prev = 100
        y_curr = 200
        example_T = np.array([100, 100, 1, 90])
        example_lambda = np.array([0, 0, 0, 0])
        example_sigma = np.array([1e-19, 0.01, 0.01, 25])
        example_X_last = np.array([y_prev, y_prev, y_prev, y_prev])
        example_Z_last = np.array([0, 0, 0, 0])
        
        example_h = Particles(example_T, 
                            example_lambda, 
                            example_sigma, 
                            example_X_last,
                            example_Z_last,
                            4) # Z_curr not used in priodic likelihood so just set to zero

        # initialisation does not matter, we're just testing the get_periodic_likelihood func. 
        particle_filter = PeriodicityParticleFilter(Config([]))

        p = particle_filter.log_periodic_likelihood(y_curr, example_h.T, example_h.Sigma, example_h.Lambda, example_h.X, example_h.Z)

        p1, p2, p3, p4 = p 
        p1_expected, p2_expected, p3_expected, p4_expected = 42.8, 39.9, 0, 0.014731

        self.assertAlmostEqual(p1,p1_expected, 1, f"Expected {p1_expected} for test case 1, but received {p1}")
        self.assertAlmostEqual(np.exp(p2), p2_expected, 1, f"Expected {p2_expected} for test case 2, but received {p2}")
        self.assertAlmostEqual(np.exp(p3), p3_expected, 12, f"Expected {p3_expected} for test case 3, but received {p3}")
        self.assertAlmostEqual(np.exp(p4), p4_expected, 6, f"Expected {p4_expected} for test case 4, but received {p4}")


    def test_periodic_likelihood_with_noise(self):
        '''
        Test case: 

        Let's have the signal as periodic where the true period T = 100,
        with noise that occurs within 25 time units. Example signal could look like:
        
        i.e. signal timestamps = 0, 10, 100, 117, 200, 214, 300, 313, 400, 415, 500,...

        For our particle vector h = [T, lambda, sigma, X, Z], let's have the following samples: 

        h1 = [T = 100, lambda = ln(2)/100, sigma = 1E-19, x_last = 200, z_last = 114]
        We set the particle T = 100 and sigma = 1E-19 - i.e. this particle is close to the actual true signal
        parameters. We should get a zero value for the peridic series likelihood as the parameters indicate we are most certain
        that the period of the true signal is 100 with almost zero uncertainty, therefore 214 being a true signal is impossible. 
        We've set the current and last noise signal times to be 114 and 214; we set the lambda to ln(2)/100 such that 100 becomes 
        the half-life of our exponential distribution. We should get a pdf value of 0.5 * f_exp(0) = 0.5 * lambda ~= .0034657 from the exponential distribution. 
        Therefore, as the false-positve likelihood is expon-pdf * (1 - Phi), we will expect a fp value close to 0.0034657. 

        h2 = [T = 100, lambda = ln(2)/100, sigma = 86, x_last = 200, z_last = 114]
        Similar setup to first particle, h1, but now the uncertainty for the periodic signal has been increased, such that the 
        214 y_curr value now falls within 1 std dev (next expected periodic signal is at 300, but std dev is now set to 86). 
        We should expect a higher liklihood now for the periodic model. 
        For the noise model, we still expect a likelihood close to 0.0034657, but lower as now the probability of this signal being noise is 
        offset by the uncertainty now introduced by sigma (refer to Likelihood Functions section). 
        '''
        z_last = 114
        y_prev = 200
        y_curr = 214

        example_T = np.array([100, 100])
        example_lambda = np.array([np.log(2)/100, np.log(2)/100])
        example_sigma = np.array([1e-19, 86])
        example_X_last = np.array([y_prev, y_prev])
        example_Z_last = np.array([z_last, z_last])

        example_h = Particles(example_T, 
                            example_lambda, 
                            example_sigma, 
                            example_X_last,
                            example_Z_last, 
                            2) # Z_curr not used in priodic likelihood so just set to zero

        # initialisation does not matter, we're just testing the get_periodic_likelihood func. 
        particle_filter = PeriodicityParticleFilter(Config([]))

        p_periodic = particle_filter.log_periodic_likelihood(y_curr, example_h.T, example_h.Sigma, example_h.Lambda, example_h.X, example_h.Z)
        p_fp = particle_filter.log_fp_likelihood(y_curr, example_h.T, example_h.Sigma, example_h.Lambda, example_h.X, example_h.Z)

        p_per_1, p_per_2 = p_periodic
        p_fp_1, p_fp_2 = p_fp

        p_per_1_expected, p_per_2_expected = 0.0, 0.0014068
        p_fp_1_expected, p_fp_2_expected = np.round(np.log(2)/100 * 0.5, 6), 0.0029159

        # periodic series likelihood assertions
        self.assertEqual(np.exp(p_per_1), p_per_1_expected, f"Expected periodic likelihood = {p_per_1_expected} for test case 1, but got {p_per_1} instead")
        self.assertAlmostEqual(np.exp(p_per_2), p_per_2_expected, 7, f"Expected periodic likelihood = {p_per_2_expected} for test case 2, but got {np.round(p_per_2,7)} instead")
        
        # false positive likelihood assertions
        self.assertAlmostEqual(np.exp(p_fp_1), p_fp_1_expected, 6, f"Expected false posiive likelihood = {p_fp_1_expected} for test case 1, but got {p_fp_1} instead")
        self.assertAlmostEqual(np.exp(p_fp_2),  p_fp_2_expected, 7, f"Expected false positive likelihood = {p_fp_2_expected} for test case 2, but got {np.round(p_fp_2, 7)} instead")

    def test_event_provenance_sampling(self):
        '''
        Test our method for updating our Xi and Zi parameters according to the event
        provenance variable ri. 

        Test case 1:
        hi = [T = 100, lambda = 0, sigma = 1e-19, xlast = 100, zlast = 100]
        The parameters set for this particle are indicating that we are very sure that the current 
        observation is generated from our periodic signal while there is no change of it being a false positive.
        Therefore, bernoulli distrbution p will be equal to 1. Hence, we should get r = 1. 

        Test case 2: 
        hi = [T = 100, lambda = np.log(2)/50, sigma = 1e19, xlast = 100, zlast = 150]
        This particle has parameters for periodc signal likelihood set with period 100 
        and large sigma. Hence, periodic signal likelihood will be ~0.  
        
        Meanwhile, last Z observed is set to 150. We set lambda to ln(2)/50 such that ycurr=200 is the half life 
        of the exponential distribution pdf. Therefore, false positive likelihood given ri = 0 is = ln(2)50*0.5 = 0.006931471805599453. 
        p(ri = 0) = 1 - gausscdf(100, 100, 1e19), where gauss cdf = 0.5. So, false positive likelihood = 0.00693 * 0.5

        Therefore, overall bernoulli p = 0 / (0 + 0.003465...) = 0, so we should get r = 0. 

        Test case 3: 
        hi = [T = 100, lambda = np.log(2)/10, sigma = 25, xlast = 175, zlast = 190]

        Periodic signal model is way off, and so p_periodic_given_signal_is_periodic will be = 0. 
        False positive model parameter lambda is set to ln(2)/10 such that p_fp_given_signal_is_fp = ln(2)/10 * 0.5. 
        Given these parameters, periodic model is somewhat uncertain of whether current observation is 
        periodic, and the same for the false positive model. Given the setup, the likelihood of this
        being a false positive is greater (i.e. bernoulli p < 0.5) so we tend to get more zeros for r.
        Given the seed, we get a zero.

        Test case 4: 
        hi= [T = 10, lambda= np.log(2)/5, sigma = 7, xlast = 191, zlast = 195]

        ycurr = 200 and period is 10, xlast is 191 with sigma = 7 so a new signal is due soon. 
        ycurr = 200 and lambda is np.log(2)/10 is the half life of the exponential given zlast = 195. 
        The setup should be unclear to the model whether the signal is periodic or a false positive.
        p value that we get is close to 0.5. Given the seed 0, we get a r value of 1. 
        '''
        ycurr = 200
        example_T = np.array([100, 100, 100, 10])
        example_lambda = np.array([1e-19, np.log(2)/50, np.log(2)/10, np.log(2)/10])
        example_sigma = np.array([1e-19, 1e19, 50, 10])
        example_X_last = np.array([100, 100, 175, 191])
        example_Z_last = np.array([0, 150, 190, 195])
        N = 4

        example_h = Particles(example_T, 
                            example_lambda, 
                            example_sigma, 
                            example_X_last, # X_curr not used in priodic likelihood so just set to zero 
                            example_Z_last, 
                            N) # Z_curr not used in priodic likelihood so just set to zero

        example_h.X_last = example_X_last
        example_h.Z_last = example_Z_last

        # initialisation does not matter. 
        particle_filter = PeriodicityParticleFilter(Config([]))

        np.random.seed(2) # sampling is random so seed it
        r = particle_filter.sample_event_provenance(ycurr, example_h)
        r1, r2, r3, r4 = r
        
        self.assertEqual(r1, 1)
        self.assertEqual(r2, 0)
        self.assertEqual(r3, 0)
        self.assertEqual(r4, 1)

    def test_particle_resampling(self):
        n = 1000
        T = np.array(range(0,n))
        L = np.array(range(0,n))
        S = np.array(range(0,n))
        X = np.array(range(0,n))
        Z = np.array(range(0,n))
        
        h = Particles(T, L, S, X, Z, n)
        pf = PeriodicityParticleFilter(Config([], num_particles=n))
        
        # Test Case 1: 
        # Weights are 1 for idx = 0, else 0. h returned should only contain values from h[0]
        w1 = np.zeros(n)
        w1[0] = 1

        h1 = pf.resample_particles(h, w1)

        msg = f"Expected resampled particle to be all zeros, but received {h1}"
        self.assertTrue(np.array_equal(h1.as_matrix(), np.zeros(shape=(5,n))), msg)

        # Test Case 2: 
        # Weights are equal - sampling should be from uniform distribution 
        np.random.seed(0)
        w2 = np.ones(n) / n
        h2 = pf.resample_particles(h, w2)

        res = scipy.stats.kstest(h2.T, 'uniform', N=n) # all var's have same value so we can use just T

        # we are very certain this should be uniform so set expected pvalue to zero
        self.assertEqual(res.pvalue, 0.0, "Particles resampling should have uniform distribution, " + \
            f"but p-value for goodness-of-fit test aganst uniform distribution returned {res.pvalue}, not 0.0")
        
        
        # Test Case 3: 
        # Weights are normally distributed
        w3 = scipy.stats.norm.pdf(np.linspace(-3,3,n))
        w3 = w3 / np.sum(w3)
        h3 = pf.resample_particles(h, w3)

        res = scipy.stats.kstest(h3.T, 'norm', N=n)

        # we are very certain this should be normal so set expected pvalue to zero
        self.assertEqual(res.pvalue, 0.0, "Particles resampling should have normal distribution, " + \
            f"but p-value for goodness-of-fit test aganst normal distribution returned {res.pvalue}, not 0.0")

    def test_reweight_multiplier(self):
        '''
        Test the method for decaying particles whose lambda value is significantly larger than T by factor c.
        c = 2. Particles where effective c > 2 should be decayed according to exponential distribution, with 
        decay factor set to 1.0. 
        Test cases: 
        T = 100, lambda = 1/1000 -> effective c = 0.1
        T = 100, lambda = 1/100 -> effective c = 1
        T = 100, lambda = 1/50 -> effective c = 2
        T = 100, lambda = 1/10 -> effective c = 10
        T = 100, lambda = 1 -> effective c = 100
        T = 100, lambda = 10 -> effective c = 1000

        T = 0.01, lambda = 100 -> effective c = 1
        T = 0.001, lambda = 1000 -> effective c = 1
        '''
        c = 2
        n = 8
        T = np.array([100, 100, 100, 100, 100, 100, 0.01, 0.001])
        L = np.array([1/1000, 1/100, 1/50, 1/10, 1, 10, 100, 1000])
        S = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        X = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        Z = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        h = Particles(T, L, S, X, Z, n)
        model = PeriodicityParticleFilter(Config([]))

        # check the normal version
        ws = model.get_reweight_multiplier(h, c)

        expected_ws = [1, 1, 1, 9.2311635e-01, 3.7531110e-01, 4.632e-05, 1, 1]
        for i in range(n):
            expected_w = expected_ws[i]
            actual_w = round(ws[i],8)
            self.assertEqual(expected_w, actual_w, f'Test case {i}: Expected {expected_w} but received {actual_w}')

        # check the log version 
        ws = model.get_log_reweight_multiplier(h, c)

        expected_ws = [0, 0, 0, -0.08, -0.98, -9.98, 0, 0]
        for i in range(n):
            expected_w = expected_ws[i]
            actual_w = round(ws[i], 6)

            self.assertEqual(expected_w, actual_w, f'Log Test case {i}: Expected {expected_w} but received {actual_w}')

    def test_sample_cauchy_gt_zero(self):
        np.random.seed(1)
        model = PeriodicityParticleFilter(Config([]))

        T = np.array([1,5,10,25,50,100])
        L = np.array([2,4,6,8,10,12])
        S = np.array([2, 7, 10, 20, 30, 40])
        cauchy_scale = 100

        # this should fail as the function should expect arr and scale 
        # input shape to match. Here, arr.shape=(3,6) while cauchy_scale is float (shape=())
        try:
            samples = model.sample_cauchy_1D_gt_zero(T, cauchy_scale)
            raise AssertionError('Exception should have been thrown for sample_cauchy_gt_zero.')
        except ValueError as e:
            pass 

        scale = np.broadcast_to(cauchy_scale, len(T))
        samples = model.sample_cauchy_1D_gt_zero(T, scale)

        self.assertEqual(sum((samples < 0)), 0, 'Negative samples found after using method sample_cauchy_1D_gt_zero')