# Rust Poly

Fast polynomial and rational evaluation for Python written in Rust.


## Usage
### Basics

``` python
>>> from rust_poly import ExpressionTree

```


We can construct a polynomial with the `ExpressionTree` constructor.
``` python
>>> poly = ExpressionTree([1, 2, 3])
>>> print(poly)
1 + 2x₁¹ + 3x₁², dtype = i64

```
`ExpressionTree` objects can be called on arrays and broadcast over the last dimension.
``` python
>>> poly([2])
array(17)

>>> poly([[1], [2], [3]])
array([ 6, 17, 34])

>>> poly([[[1], [2]], [[3], [4]]])
array([[ 6, 17],
       [34, 57]])

```
The last dimension of the argument must equal the dimension of the expression.

``` python
>>> poly([1, 2, 3])
Traceback (most recent call last):
...
ValueError: Incompatible shape for evaluation

```


`ExpressionTree` objects can be composed and are lazily evaluated.
``` python
>>> print(poly * poly)
(1 + 2x₁¹ + 3x₁²) * (1 + 2x₁¹ + 3x₁²), dtype = i64
>>> (poly * poly)([2])
array(289)

```
`ExpressionTree` objects can be "expanded".
``` python
>>> print((poly * poly).expand())
1 + 4x₁¹ + 10x₁² + 12x₁³ + 9x₁⁴, dtype = i64

```
`ExpressionTree` objects can be scaled, added, subtracted, divided, and multiplied lazily.

```  python
>>> p1 = ExpressionTree([1, 2])
>>> print(p1)
1 + 2x₁¹, dtype = i64
>>> p2 = ExpressionTree([3, 4, 5])
>>> print(p2)
3 + 4x₁¹ + 5x₁², dtype = i64
>>> print(5 * p1)
5 * (1 + 2x₁¹), dtype = i64
>>> print(p1 + p2)
(1 + 2x₁¹) + (3 + 4x₁¹ + 5x₁²), dtype = i64
>>> print(p1 - p2)
(1 + 2x₁¹) - (3 + 4x₁¹ + 5x₁²), dtype = i64
>>> print(p1 * p2)
(1 + 2x₁¹) * (3 + 4x₁¹ + 5x₁²), dtype = i64
>>> print(p1 / p2)
(1 + 2x₁¹) / (3 + 4x₁¹ + 5x₁²), dtype = i64

```

Different data types (float, int, and complex) can be handled. 

``` python
>>> poly([2.0])
array(17.)

>>> poly([2.0 + 0j])
array(17.+0.j)

>>> floatpoly = ExpressionTree([1.0, 2.0])
>>> print((floatpoly + poly).expand())
2 + 4x₁¹ + 3x₁², dtype = f64

```



### Higher-dimensional polynomials

Polynomials can be defined with many dimensions.

``` python 
>>> p2d = ExpressionTree([[1, 2], [3, 4]])
>>> print(p2d)
1 + 2x₂¹ + 3x₁¹ + 4x₁¹x₂¹, dtype = i64

>>> p2d.dimension
2

```

When evaluating higher dimension arrays, the last dimension of the expression to be evaluated must match the dimension of the array.

``` python
>>> p2d([1, 2])
array(16)
>>> p2d([[1], [2]])
Traceback (most recent call last):
...
ValueError: Incompatible shape for evaluation

>>> p2d([[1, 2, 3], [2, 3, 4]])
Traceback (most recent call last):
...
ValueError: Incompatible shape for evaluation

```

Polynomial dimension must match for composition.

``` python
>>> p2d + p1
Traceback (most recent call last):
...
ValueError: Incompatible dimensions for composition

```

### Other operations 
Differentiation and integration are supported. Integer polynomials are automatically converted to floats with this method.

``` python
>>> print(p1.deriv([1]))
deriv(1 + 2x₁¹, [1]), dtype = f64
>>> print(p1.deriv([1]).expand())
2, dtype = f64
>>> print(p1.deriv([0]).expand())
1 + 2x₁¹, dtype = f64

>>> print(p2.deriv([1]))
deriv(3 + 4x₁¹ + 5x₁², [1]), dtype = f64
>>> print(p2.deriv([1]).expand())
4 + 10x₁¹, dtype = f64

>>> print(p1.integ([1]).expand())
1x₁¹ + 1x₁², dtype = f64
>>> print(p1.deriv([-1]).expand())
1x₁¹ + 1x₁², dtype = f64


```
The syntax for multi-dimensional derivatives and integrals is similar:

``` python
>>> print(p2d.deriv([1, 0]).expand())
3 + 4x₂¹, dtype = f64
>>> print(p2d.deriv([0, 1]).expand())
2 + 4x₁¹, dtype = f64

```



## More complex example
Quick example of computing a stiffness matrix:

``` python
from rust_poly import ExpressionTree as Polynomial, from_terms
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
