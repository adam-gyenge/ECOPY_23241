import random
import math
import typing
import pyerf


class SecondClass:
    def __init__(self, rand, a, b):
        self.random = rand
        self.a = a
        self.b = b


class UniformDistribution:
    def __init__(self, rand, a, b):
        self.rand = rand
        self.a = a
        self.b = b

    def pdf(self, x):
        if self.a <= x <= self.b:
            return 1 / (self.b - self.a)
        else:
            return 0

    def cdf(self, x):
        if x<self.a:
            return 0
        elif self.a<=x<=self.b:
            return (x-self.a)/(self.b-self.a)
        else:
            return 1

    def ppf(self, p):
        if 0 <= p <= 1:
            return self.a + p * (self.b - self.a)
        else:
            raise ValueError("Valószínűségi értéknek 0 és 1 között kell lennie.")

    def gen_rand(self):
        return random.uniform(self.a, self.b)

    def mean(self):
        if self.a == self.b:
            raise Exception("Moment undefined")
        return (self.a + self.b) / 2

    def median(self):
        return (self.a + self.b) / 2

    def variance(self):
        if self.a == self.b:
            raise Exception("Moment undefined")
        return ((self.b - self.a) ** 2) / 12

    def skewness(self):
        if self.a == self.b:
            raise Exception("Moment undefined")
        return 0

    def ex_kurtosis(self):
        if self.a == self.b:
            raise Exception("Moment undefined")
        return -(6 / 5)

    def mvsk(self):
        # Számítás az első 3 centrális momentumra és kurtózisra
        mean = self.mean()
        variance = self.variance()
        skewness = self.skewness()
        kurtosis = self.ex_kurtosis()

        return [mean, variance, skewness, kurtosis]


class NormalDistribution:
    def __init__(self, rand, loc, scale):
        self.rand = rand
        self.loc = loc
        self.scale = scale

    def pdf(self, x):
        exponent = -((x - self.loc) ** 2) / (2 * (self.scale ** 2))
        coefficient = 1 / (math.sqrt(self.scale) * math.sqrt(2 * math.pi))
        probability_density = coefficient * math.exp(exponent)
        return probability_density

    def cdf(self, x):
        # Számítás a normális eloszlás kumulatív eloszlásfüggvényéből
        z = (x - self.loc) / self.scale
        cumulative_probability = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        return cumulative_probability

    def ppf(self, p):
        if 0 <= p <= 1:
            # Inverz kumulatív eloszlásfüggvény számítása
            z = math.sqrt(2) * pyerf.erfinv(2 * p - 1)
            x = self.loc + self.scale * z
            return x
        else:
            raise ValueError("Valószínűségi értéknek 0 és 1 között kell lennie.")

    def gen_rand(self):
        # Normális eloszlású véletlen szám generálása
        return self.loc + self.scale * math.sqrt(-2 * math.log(self.rand.random())) * math.cos(2 * math.pi * self.rand.random())

    def mean(self):
        return self.loc

    def median(self):
        return self.loc

    def variance(self):
        return self.scale

    def skewness(self):
        return 0

    def ex_kurtosis(self):
        return 0

    def mvsk(self):
        mean = self.mean()
        variance = self.variance()
        skewness = self.skewness()
        kurtosis = self.ex_kurtosis()

        return [mean, variance, skewness, kurtosis]


class CauchyDistribution:
    def __init__(self, rand, loc, scale):
        self.rand = rand
        self.loc = loc
        self.scale = scale

    def pdf(self, x):
        return (1 / (math.pi * self.scale)) * (1 / (1 + ((x - self.loc) / self.scale) ** 2))

    def cdf(self, x):
        return (1/math.pi)*math.atan((x-self.loc)/self.scale) + 1/2

    def ppf(self, p):
        if 0<=p<=1:
            return self.loc + self.scale * math.tan(math.pi*(p - 1/2))
        else:
            raise ValueError("Valószínűségi érték nem 0 és 1 között van")

    def gen_rand(self):
        # Cauchy eloszlású véletlen szám generálása
        return self.loc + self.scale * math.tan(math.pi * (self.rand.random() - 0.5))

    def mean(self):
        return None  # A Cauchy eloszlásnak nincs véges várható értéke

    def median(self):
        return self.loc  # Medián az eloszlás középpontja

    def variance(self):
        return None  # A Cauchy eloszlásnak nincs véges varianciája

    def skewness(self):
        return None  # A Cauchy eloszlásnak nincs véges ferdesége

    def ex_kurtosis(self):
        return None  # A Cauchy eloszlásnak nincs véges kurtózisa

    def mvsk(self):
        raise Exception("Moments undefined")  # A Cauchy eloszlásnak nincsenek véges momentumai
