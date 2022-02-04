# Rust Poly

Fast polynomial and rational evaluation for Python written in Rust.


Quick example of computing a stiffness matrix:

``` python
import sys
sys.path.append("/home/jason/Projects/rust/rust_poly")
from rust_poly import FloatExpression as Polynomial, from_terms
import numpy as np

terms = {"x1": 0, "x2": 1, "x1a": 2, "x2a": 3,
         "x1b": 4, "x2b": 5, "x1c": 6, "x2c": 7,
         "nu": 8, "none": None}

def get_term(term):
    t = [0] * (len(terms) - 1)
    if terms[term] is not None:
        t[terms[term]] = 1
    return tuple(t)

def get_term_array(term):
    return np.array(get_term(term))

def get_poly(**kwargs):
    return from_terms(
        {get_term(key): val for key, val in kwargs.items()}
    )

# Stiffness matrix interpolation functions
Na = (get_poly(x2=1, x2b=-1) *get_poly(x1c=1, x1b=-1)
    -get_poly(x1=1, x1b=-1) *get_poly(x2c=1, x2b=-1)
) / (get_poly(x2a=1, x2b=-1) *get_poly(x1c=1, x1b=-1)
    -get_poly(x1a=1, x1b=-1) *get_poly(x2c=1, x2b=-1))
Nb = (get_poly(x2=1, x2c=-1) *get_poly(x1a=1, x1c=-1)
    -get_poly(x1=1, x1c=-1) *get_poly(x2a=1, x2c=-1)
) / (get_poly(x2b=1, x2c=-1) *get_poly(x1a=1, x1c=-1)
    -get_poly(x1b=1, x1c=-1) *get_poly(x2a=1, x2c=-1))
Nc = (get_poly(x2=1, x2a=-1) *get_poly(x1b=1, x1a=-1)
    -get_poly(x1=1, x1a=-1) *get_poly(x2b=1, x2a=-1)
) / (get_poly(x2c=1, x2a=-1) *get_poly(x1b=1, x1a=-1)
    -get_poly(x1c=1, x1a=-1) *get_poly(x2b=1, x2a=-1))

# Strain-displacement matrix
B = np.full((3, 6), Polynomial.zero(len(terms) - 1) / Polynomial.one(len(terms) - 1))
B[0, 0] = Na.deriv(get_term_array('x1'))
B[0, 2] = Nb.deriv(get_term_array('x1'))
B[0, 4] = Nc.deriv(get_term_array('x1'))
B[1, 1] = Na.deriv(get_term_array('x2'))
B[1, 3] = Nb.deriv(get_term_array('x2'))
B[1, 5] = Nc.deriv(get_term_array('x2'))
B[2, 0] = Na.deriv(get_term_array('x2'))
B[2, 1] = Na.deriv(get_term_array('x1'))
B[2, 2] = Nb.deriv(get_term_array('x2'))
B[2, 3] = Nb.deriv(get_term_array('x1'))
B[2, 4] = Nc.deriv(get_term_array('x2'))
B[2, 5] = Nc.deriv(get_term_array('x1'))

# Material constitutive matrix (plane stress)
D = np.full(
    (3, 3),
    Polynomial.zero(len(terms) - 1) / Polynomial.one(len(terms) - 1)
)
D[0, 0] = D[1, 1] = get_poly(nu=0, none=1)
D[1, 0] = D[0, 1] = get_poly(nu=1)
D[2, 2] = get_poly(nu=-0.5, none=0.5)
D = D / (get_poly(none=1, nu=1) * get_poly(none=1, nu=-1))

# Coordinate Determinant
Det = (
    get_poly(x1a=1) * get_poly(x2b=1, x2c=-1)
    + get_poly(x1b=1) * get_poly(x2c=1, x2a=-1)
    + get_poly(x1c=1) * get_poly(x2a=1, x2b=-1)
)


# Compute an abstract representation for the stiffness matrix of a constant
# strain triangular element
K = (B.T @ D @ B) / Det


# Specialize the stiffness matrix for a particular element
coords = np.array([0, -1, 2, 0, 0, 1, 0.25])
indices = np.array([2, 3, 4, 5, 6, 7, 8])
vfunc = np.vectorize(lambda a: a.partial(indices, coords).expand().to_constant())
K_special = vfunc(K)

expected = np.array([
    [ 2.5  ,  1.25 , -2.   , -1.5  , -0.5  ,  0.25 ],
    [ 1.25 ,  4.375, -1.   , -0.75 , -0.25 , -3.625],
    [-2.   , -1.   ,  4.   ,  0.   , -2.   ,  1.   ],
    [-1.5  , -0.75 ,  0.   ,  1.5  ,  1.5  , -0.75 ],
    [-0.5  , -0.25 , -2.   ,  1.5  ,  2.5  , -1.25 ],
    [ 0.25 , -3.625,  1.   , -0.75 , -1.25 ,  4.375]
]) / (0.9375 * 64)
assert np.allclose(expected, K_special)
```
