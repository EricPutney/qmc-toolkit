import numpy as np

import scipy
from scipy import special
from scipy import stats

from scipy.stats.qmc import Halton
from scipy.special import rel_entr
from scipy.special import erf
from scipy.optimize import root
from scipy.special import gamma
from scipy.special import gammaincc


def ndim_norm_percentile(d, sigmaRatio):
    """
    Estimate the radial "CDF" of an isotropic gaussian in d-dimensions.

    d: [int] number of dimensions.

    sigmaRatio: [float] number of sigmas to truncate at.
    """

    if d == 1:
        return 0.5 * (erf(sigmaRatio / np.sqrt(2)) - erf(-sigmaRatio / np.sqrt(2)))

    elif d > 1:
        return 1 - (gammaincc(d / 2, (1 / 2) * (sigmaRatio) ** 2))


class BaseGenerator:
    def __init__(
        self, d=3, randomize=False, offset=None, recenter=False, verbose=False
    ):
        """
        template for d-dimensional random sample generator for QMC.

        d: [int] specifies the number of dimensions. Default = 3.

        randomize: [bool] shuffling can improve quasi-randomness if d is large. Default = False since this is not usually needed.

        offset : np.ndarray with shape (d,), offset will be added (modulo 1) to the random numbers
        """

        self.d = d
        self.sampler = None
        self.offset = offset
        self.recenter = recenter
        self.verbose = verbose

    def reset(self):
        raise NotImplementedError

    def _sample(self, n_samples, reset=False):
        raise NotImplementedError

    def sample(self, n_samples, reset=False):
        output = self._sample(n_samples, reset)

        if self.offset is not None:
            if self.verbose:
                print("INFO: adding offset: {}".format(self.offset))
            output = (output + self.offset) % 1.0
        if self.recenter:
            mean_value = output.mean(axis=0)
            offset_recenter = mean_value - np.ones((self.d,)) * 0.5
            if self.verbose:
                print("INFO: recentering: adding offset: {}".format(offset_recenter))
            output = (output - offset_recenter) % 1.0
            if self.verbose:
                print("INFO: recentered: mean: {}".format(output.mean(axis=0)))

        # Mirror
        # output = np.concatenate(
        #    [
        #        output,
        #        np.array([0.,1.])+np.array([1.,-1.])*output,
        #        np.array([1.,0.])+np.array([-1.,1.])*output,
        #        1.0-output,
        #    ],
        #    axis=0
        # )
        return output

    def sample_gaussian(
        self,
        n_samples,
        loc=None,
        scale=None,
        reset=False,
        trunc_scale=None,
        sample_buffer=1.5,
    ):
        # draw samples from [0,1]^D
        n_samples_buffered = int(n_samples * sample_buffer)
        arr_unif = self.sample(n_samples_buffered, reset)

        # transform each components to standard normal
        arr_output = -np.sqrt(2.0) * scipy.special.erfcinv(2.0 * arr_unif)

        if trunc_scale is not None:
            mag = np.linalg.norm(arr_output, axis=-1)
            mask = mag < trunc_scale

            arr_output = arr_output[mask][:n_samples]
            assert arr_output.shape[0] == n_samples, "increase sample buffer"

        # shift and scale
        if scale is not None:
            arr_output = scale * arr_output
        if loc is not None:
            arr_output = arr_output + loc

        # return output
        return arr_output

    def test_gaussian_likelihood(self, n_samples):
        arr_samples = self.sample_gaussian(n_samples)
        dist = scipy.stats.multivariate_normal(
            mean=np.zeros(self.d), cov=np.eye(self.d)
        )
        arr_logpdf = dist.logpdf(arr_samples)
        return arr_logpdf.mean()


class HaltonGenerator(BaseGenerator):
    def __init__(
        self, d=3, randomize=False, offset=None, recenter=False, verbose=False
    ):
        """
        d-dimensional generator using scipy's implementation of the Halton sequence.

        d: [int] specifies the number of dimensions. Default = 3.

        randomize: [bool] shuffling can improve quasi-randomness if d is large. Default = False since this is not usually needed.
        recenter : [bool] subtract mean values from the center in order to mitigate the boundary effect
        verbose  : [bool] print out additional information

        offset : np.ndarray with shape (d,), offset will be added (modulo 1) to the random numbers
        """
        BaseGenerator.__init__(
            self,
            d=d,
            offset=offset,
            recenter=recenter,
            verbose=verbose,
        )
        self.sampler = scipy.stats.qmc.Halton(self.d, scramble=randomize)

    def reset(self):
        """
        Sampler needs to be reset each time sample is called, since the scipy Halton sequence generator always continues from where it left off.
        """
        self.sampler.reset()

    def _sample(self, n_samples, reset=False):
        """
        Generate uniform quasi-random samples from the Halton sequence over domain [0,1].

        n_samples: [int]  number of samples required.
        reset    : [bool] if true, reset the random state.
        """
        if reset:
            self.reset()

        output = self.sampler.random(n_samples + 1)[1:]
        return output


class SobolGenerator(BaseGenerator):
    def __init__(
        self, d=3, randomize=False, offset=None, recenter=False, verbose=False
    ):
        """
        d-dimensional generator using scipy's implementation of the Halton sequence.

        d: [int] specifies the number of dimensions. Default = 3.

        randomize: [bool] shuffling can improve quasi-randomness if d is large. Default = False since this is not usually needed.
        recenter : [bool] subtract mean values from the center in order to mitigate the boundary effect
        verbose  : [bool] print out additional information

        offset : np.ndarray with shape (d,), offset will be added (modulo 1) to the random numbers
        """
        BaseGenerator.__init__(
            self,
            d=d,
            offset=offset,
            recenter=recenter,
            verbose=verbose,
        )
        self.sampler = scipy.stats.qmc.Sobol(self.d, scramble=randomize)

    def reset(self):
        """
        Sampler needs to be reset each time sample is called, since the scipy Halton sequence generator always continues from where it left off.
        """
        self.sampler.reset()

    def _sample(self, n_samples, reset=False):
        """
        Generate uniform quasi-random samples from the Halton sequence over domain [0,1].

        n_samples: [int]  number of samples required.
        reset    : [bool] if true, reset the random state.
        """
        assert n_samples < 2**30  # approximate period of the default sobol sequence
        if reset:
            self.reset()

        output = self.sampler.random(n_samples)  # DO NOT SKIP THE FIRST POINT
        return output


class FibonacciLattice(BaseGenerator):
    def __init__(
        self, d=2, randomize=False, offset=None, recenter=False, verbose=False
    ):
        """
        d-dimensional generator for Fibonacci lattices.
        see also: http://extremelearning.com.au/evenly-distributing-points-on-a-sphere/

        d: [int] specifies the number of dimensions. Default = 3.

        randomize: [bool] shuffling can improve quasi-randomness if d is large. Default = False since this is not usually needed.
        recenter : [bool] subtract mean values from the center in order to mitigate the boundary effect
        verbose  : [bool] print out additional information

        offset : np.ndarray with shape (d,), offset will be added (modulo 1) to the random numbers
        """
        BaseGenerator.__init__(
            self,
            d=d,
            offset=offset,
            recenter=recenter,
            verbose=verbose,
        )
        assert d == 2, "currently Fibonacci lattice is only implemented for D=2"
        # self.sampler = scipy.stats.qmc.Halton(self.d, scramble = randomize)

        self.golden_ratio = (1.0 + 5.0**0.5) / 2

    def _sample(self, n_samples, reset=False):
        """
        Generate uniform quasi-random samples from the Halton sequence over domain [0,1].

        n_samples: [int]  number of samples required.
        reset    : [bool] if true, reset the random state.
        """

        arr_index = np.arange(0, n_samples, dtype=np.float32)
        arr_output = np.stack(
            [np.mod(arr_index / self.golden_ratio, 1.0), (arr_index + 0.5) / n_samples],
            axis=-1,
        )
        return arr_output


class randomGenerator:
    def __init__(self, d=3, randomize=False, offset=None):
        """
        d-dimensional generator using numpy's np.random.rand().

        d: [int] specifies the number of dimensions. Default = 3.

        randomize: [bool] unnecessary, will remove later. Default = False.

        offset : np.ndarray with shape (d,), offset will be added (modulo 1) to the random numbers
        """

        self.d = d

    def reset(self):
        """
        This sampler does not need to be reset.
        """

        pass

    def sample(self, n_samples, reset=False):
        """
        Sample from uniform distribution over domain [0,1]

        n_samples: [int] number of samples required.
        reset    : [bool] if true, reset the random state. (do nothing in this cae)
        """

        return np.random.rand(n_samples, self.d)


class exampleGenerator:
    def __init__(self, args):
        # initialize any class variables specified in args
        pass

    def reset(self):
        # sometimes sequence generators need to be reset each time,
        # or they will continue counting from where they left off
        # the last time sample was called
        pass

    def sample(self, n_samples):
        # sample uniform distribution n_samples times
        pass


class qmcSampler:
    def __init__(self, generator=HaltonGenerator, d=3, randomize=False):
        """
        Parent class for quasi-monte-carlo sampling from analytic distributions (iso/anisotropic gaussians, full/truncated gaussians) using inverse transform sampling.

        generator: [class] one of the generator classes (e.g. HaltonGenerator or randomGenerator) containing the common generator functions (sample and reset). Default = HaltonGenerator.

        d: [int] specifies the number of dimensions. Default = 3.

        randomize: [bool] some generators need to be randomized in certain edge cases to maintain uniformity (e.g. Halton for large values of d). Default = False since this is not usually needed.
        """

        self.d = d
        self.generator = generator(self.d, randomize)

    def sampleUniform(self, n_samples, resetGenerator=True):
        """
        Calls sample method from the selected generator, should return a quasi-random or pseudo-random uniform distribution

        n_samples: [int] number of samples required.

        resetGenerator: [bool] # sometimes sequence generators need to be reset after sampling. Default = True.
        """

        samples_uniform = self.generator.sample(n_samples)

        if resetGenerator == True:
            self.generator.reset()

        return samples_uniform

    def sampleUnitIsotropicGaussian(
        self, n_samples, truncate=False, truncScale=None, sampleBuffer=1.25
    ):
        """
        Sample from a d-dimensional isotropic gaussian with std=1 in each dimension.

        n_samples: [int] number of samples required.

        truncate: [bool] do truncation at some radius. Default = False.

        truncScale: [float] radius to truncate at. Interpreted as the number of sigmas to truncate at since sigma = 1. Default = None.

        sampleBuffer: [float] oversampling factor when doing rejection sampling. Efficiency is already estimated using ndim_norm_percentile, but random fluctuations often require further oversampling by a small factor. Default = 1.25.
        """

        if truncate == True:
            # the sigma of our gaussian is 1 by default. Don't change this,
            # since we'll just rescale later.
            sigma = 1.0

            # some fraction given by ndim_norm_percentile will fall outside
            # of the truncation window. We need to increase n_samples by a
            # proportionate factor. Buffer by some factor (25% by default)
            # to allow for random fluctuations.
            n_samples_eff = int(
                sampleBuffer
                * n_samples
                / ndim_norm_percentile(d=self.d, sigmaRatio=sigma * truncScale)
            )

            if n_samples / n_samples_eff < 0.5:
                print(
                    "Efficiency: "
                    + str(np.round(100 * n_samples / n_samples_eff, 2))
                    + "%"
                )

        else:
            # no truncation -> no rejection sampling, n_samples is sufficient
            n_samples_eff = n_samples

        # sample from the samplerClass's uniform sampler
        samples_uniform = self.sampleUniform(n_samples_eff)

        # prepare empty array for the gaussian samples
        samples_gaussian = np.zeros((n_samples_eff, self.d))

        # inverse-transform-sample each coordinate in samples_uniform
        for i in range(n_samples_eff):
            for j in range(self.d):
                samples_gaussian[i, j] = (
                    np.sqrt(2)
                    * root(
                        fun=lambda x: 0.5 * (1 + erf(x)) - samples_uniform[i][j], x0=0
                    ).x
                )

        if truncate == True:
            # only keep samples inside the truncation window, then cut to
            # the first n_samples number of samples as originally requested
            return samples_gaussian[
                np.linalg.norm(samples_gaussian, axis=1) < truncScale
            ][:n_samples]

        else:
            return samples_gaussian

    def sampleIsotropicGaussian(
        self, n_samples, scale, truncate=False, truncScale=None, sampleBuffer=1.25
    ):
        """
        Sample from an d-dimensional isotropic Gaussian by rescaling the unit isotropic Gaussian generated by sampleUnitIsotropicGaussian().

        n_samples: [int] number of samples required.

        scale: [float] standard deviation (for all dimensions).

        truncate: [bool] do truncation at some radius. Default = False.

        truncScale: [float] Number of sigmas to truncate at. Default = None.

        sampleBuffer: [float] oversampling factor when doing rejection sampling. Efficiency is already estimated using ndim_norm_percentile, but random fluctuations often require further oversampling by a small factor. Default = 1.25.
        """

        samples_gaussian = self.sampleUnitIsotropicGaussian(
            n_samples, truncate, truncScale, sampleBuffer
        )

        return samples_gaussian * scale

    def sampleAnisotropicGaussian(
        self, n_samples, scales, truncate=False, truncScale=None, sampleBuffer=1.25
    ):
        """
        Sample from an d-dimensional anisotropic Gaussian by rescaling the unit isotropic Gaussian generated by sampleUnitIsotropicGaussian().

        n_samples: [int] number of samples required.

        scales: [float] array of length d of each dimensions std.

        truncate: [bool] do truncation at some radius. Default = False.

        truncScale: [float] Number of sigmas to truncate at. Default = None.

        sampleBuffer: [float] oversampling factor when doing rejection sampling. Efficiency is already estimated using ndim_norm_percentile, but random fluctuations often require further oversampling by a small factor. Default = 1.25.
        """

        samples_gaussian = self.sampleUnitIsotropicGaussian(
            n_samples, truncate, truncScale, sampleBuffer
        )

        return samples_gaussian * scales[np.newaxis, :]
