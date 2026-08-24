import math

class OneEuroFilter:
    def __init__(self, freq, min_cutoff, beta, d_cutoff):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0

    def _low_pass_filter(self, x, x_prev, alpha):
        return alpha * x + (1.0 - alpha) * x_prev

    def _alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def filter(self, x):
        if self.x_prev is None:
            self.x_prev = x
            return x
        te = 1.0 / self.freq
        dx = (x - self.x_prev) / te
        edx = self._low_pass_filter(dx, self.dx_prev, self._alpha(self.d_cutoff))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        result = self._low_pass_filter(x, self.x_prev, self._alpha(cutoff))
        self.x_prev = result
        self.dx_prev = edx
        return result