"""
vektor/layers/__init__.py

NOTE:

New code should use the `standard_normal` method of a `Generator` instance instead.

# Old way: using np.random.randn
samples = np.random.randn(output_size, input_size)

# New way: using rng.standard_normal
rng = np.random.default_rng()
samples = rng.standard_normal(size=(output_size, input_size), dtype=np.float64)

# Specifying Mean and Standard Deviation
mu = ...  # your mean value
sigma = ...  # your standard deviation value
samples = mu + sigma * rng.standard_normal(size=...)

Source:
- https://numpy.org/doc/stable/reference/random/generated/numpy.random.randn.html#numpy.random.randn
- https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.standard_normal.html#numpy.random.Generator.standard_normal
"""
from vektor.layers.dense import Dense
from vektor.layers.sparse import Sparse

__all__ = [Dense, Sparse]
