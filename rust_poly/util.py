from . import FloatExpression
import numpy as np
from functools import reduce
def from_terms(terms):
    """
    Construct a polynomial from a dictionary of (powers, coef) pairs.
    """
    terms = dict(terms)
    dimension = reduce(lambda a, b: a if len(a) == len(b) else (), terms)
    if not dimension:
        raise ValueError("All terms must have same dimension.")
    shape = np.max(np.array(list(terms)), axis=0).astype(np.int32)
    shape += np.ones_like(shape)
    coefs = np.zeros(shape, dtype=np.float64)
    for powers, coef in terms.items():
        try:
            coefs[tuple(powers)] = coef
        except IndexError as e:
            raise ValueError("Indices must be integers", e)
    return FloatExpression(coefs)
