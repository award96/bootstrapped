class BootstrapResults(object):
    def __init__(self, lower_bound, value, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.value = value
        if self.lower_bound > self.upper_bound:
            raise ValueError(
                "lower_bound must be less than upper_bound. "
                "lower_bound: {0}, upper_bound: {1}".format(
                    self.lower_bound,
                    self.upper_bound
                )
            )
    def __str__(self):
        return '{1}    ({0}, {2})'.format(self.lower_bound, self.value,
                                          self.upper_bound)

    def __repr__(self):
        return self.__str__()
    """
        -------------------------------------------------
        All of the following functions allow a scalar to
        be added, multiplied, subtracted, or divided from
        the BootstrapResults object.
        -------------------------------------------------
        bs = BootstrapResults(3.5, 3.01, 4.45)
        print(bs)
        >> 3.5 (3.01, 4.45)
        print(bs + 1)
        >> 4.5 (4.01, 5.45)
    """
    def _apply(self, other, func):
        """
        Apply a function to the lower_bound, value, and upper_bound.
        """
        return BootstrapResults(func(self.lower_bound, other),
                                func(self.value, other),
                                func(self.upper_bound, other))
    def __add__(self, other):
        return self._apply(float(other), lambda x, other: other + x)

    def __radd__(self, other):
        return self._apply(float(other), lambda x, other: other + x)

    def __sub__(self, other):
        return self._apply(float(other), lambda x, other: x - other)

    def __rsub__(self, other):
        return self._apply(float(other), lambda x, other: other - x)

    def __mul__(self, other):
        return self._apply(float(other), lambda x, other: x * other)

    def __rmul__(self, other):
        return self._apply(float(other), lambda x, other: x * other)

    def error_width(self):
        """Returns: upper_bound - lower_bound"""
        return self.upper_bound - self.lower_bound

    def error_fraction(self):
        """Returns the error_width / value"""
        if self.value == 0:
            return np.inf # TODO
        else:
            return self.error_width() / self.value

    def is_significant(self):
        """Returns True if the upper_bound and lower_bound have the same sign."""
        # TODO more mathematically sound way to determine significance
        return self.upper_bound * self.lower_bound >= 0

    def get_result(self):
        """Returns:
            -1 if statistically significantly negative
            +1 if statistically significantly positive
            0 otherwise
        """
        # TODO
        return int(
            (int(self.is_significant()) * (self.value)) > 0
        )