import random
import typing
import math
import pyerf


class LaplaceDistribution:
    def __init__(self, rand, loc, scale):
        self.rand = rand
        self.loc = loc
        self.scale = scale

    def pdf(self, x):
        return (1 / (2 * self.scale)) * math.exp(-abs(x - self.loc) / self.scale)

    def cdf(self, x):
        if x < self.loc:
            return 0.5 * math.exp((x - self.loc) / self.scale)
        else:
            return 1 - 0.5 * math.exp(-(x - self.loc) / self.scale)

    def ppf(self, p):
        # Az aszimmetrikus Laplace eloszlás inverz kumulatív eloszlásfüggvénye
        if p < 0 or p > 1:
            raise ValueError("Az 'p' értéke 0 és 1 között kell legyen.")

        if p < 0.5:
            return self.loc + self.scale * math.log(2 * p)
        else:
            return self.loc - self.scale * math.log(2 - 2 * p)

    def gen_rand(self):
        u = self.rand.uniform(0, 1)
        if u < 0.5:
            return self.loc + self.scale * math.log(2 * u)
        else:
            return self.loc - self.scale * math.log(2 - 2 * u)

    def mean(self):
        return self.loc

    def variance(self):
        return 2 * self.scale**2

    def skewness(self):
        return 0

    def ex_kurtosis(self):
        return 3

    def mvsk(self):
        mean = self.mean()
        variance = self.variance()
        skewness = self.skewness()
        kurtosis = self.ex_kurtosis()

        return [mean, variance, skewness, kurtosis]



class ParetoDistribution:
    def __init__(self, rand, scale, shape):
        self.rand = rand
        self.scale = scale
        self.shape = shape

    def pdf(self, x):
        if x < self.scale:
            return 0
        else:
            return (self.shape * (self.scale**self.shape))/(x**(self.shape+1))

    def cdf(self, x):
        if x < self.scale:
            return 0
        else:
            return 1-((self.scale/x)**self.shape)

    def ppf(self, p):
        if p < 0 or p > 1:
            raise ValueError("p has to be between 0 and 1")
        else:
            return self.scale * ( (1 - p) ** (-1/self.shape) )

    def gen_rand(self):
        u = self.rand.uniform(0, 1)
        return self.ppf(u)

    def mean(self):
        if self.shape <= 1:
            return math.inf
        elif self.shape > 1:
            return (self.shape*self.scale)/(self.shape-1)
        else:
            raise ValueError("something is wrong")

    def variance(self):
        if self.shape <= 2:
            return math.inf
        elif self.shape > 2:
            return ((self.scale**2)*self.shape)/(((self.shape-1)**2)*(self.shape-2))

    def skewness(self):
        if self.shape < 3:
            raise Exception("Moment undefined")
        elif self.shape > 3:
            return ((2 * (1 + self.shape)) / (self.shape - 3)) * math.sqrt((1 - (2/self.shape)))
        else:
            return math.inf

    def ex_kurtosis(self):
        if self.shape > 4:
            return (6 * ((self.shape * 3) + (self.shape * 2) - (6 * self.shape) - 2)) / (self.shape * (self.shape - 3) * (self.shape - 4))
        elif self.shape <= 1:
            raise Exception("Moment undefined")
        else:
            return math.inf

    def mvsk(self):
        mean = self.mean()
        variance = self.variance()
        skewness = self.skewness()
        kurtosis = self.ex_kurtosis()

        return [mean, variance, skewness, kurtosis]