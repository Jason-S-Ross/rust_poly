// The MIT License (MIT)
//
//     Copyright (c) 2022 Jason Ross
//
//     Permission is hereby granted, free of charge, to any person obtaining a copy
//     of this software and associated documentation files (the "Software"), to
//     deal in the Software without restriction, including without limitation the
//     rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
//     sell copies of the Software, and to permit persons to whom the Software is
//     furnished to do so, subject to the following conditions:
//
//     The above copyright notice and this permission notice shall be included in
//     all copies or substantial portions of the Software.
//
//     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//     FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//     IN THE SOFTWARE.

use std::ops::{Mul, Add, Div, Sub};

use ndarray::{Array, Array1, ArrayView, Dimension, NdIndex, ScalarOperand};
use num_traits::{NumCast, Zero};
use numpy::npyffi::array::PY_ARRAY_API;
use numpy::npyffi::types::NPY_TYPES;
use numpy::{
    Element, IntoPyArray, PyArray1, PyReadonlyArray, PyReadonlyArray1, PyReadonlyArrayDyn,
};
use polynomial::PolynomialError;
use pyo3::conversion::FromPyPointer;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyType;
use pyo3::{prelude::*, AsPyPointer};

use num_complex::Complex;
use pyo3::{PyNumberProtocol, PyObjectProtocol};
use tree::Expression;

use crate::pow::Pow;
pub mod polynomial;
pub mod pow;
pub mod tree;

/// An abstract expression tree with concrete terms. Lazily evaluates arithmetic
/// operations.
#[derive(Clone)]
#[pyclass]
#[pyo3(text_signature = "(vals)")]
struct ExpressionTree {
    expression: ExpressionType,
}

#[derive(Clone)]
enum ExpressionType {
    Float(Expression<f64>),
    Complex(Expression<Complex<f64>>),
    Int(Expression<i64>),
}

/// Performs casting for binary operations between expression trees
macro_rules! impl_binop {
    ($fname:ident) => {
        fn $fname(&self, right: &Self) -> PyResult<Self> {
            match (self, right) {
                (ExpressionType::Float(f), ExpressionType::Complex(c))
                | (ExpressionType::Complex(c), ExpressionType::Float(f)) => {
                    Ok(ExpressionType::Complex(c.$fname(&f.astype()?)?))
                }
                (ExpressionType::Float(f), ExpressionType::Int(i))
                | (ExpressionType::Int(i), ExpressionType::Float(f)) => {
                    Ok(ExpressionType::Float(f.$fname(&i.astype()?)?))
                }
                (ExpressionType::Complex(c), ExpressionType::Int(i))
                | (ExpressionType::Int(i), ExpressionType::Complex(c)) => {
                    Ok(ExpressionType::Complex(c.$fname(&i.astype()?)?))
                }
                (ExpressionType::Float(l), ExpressionType::Float(r)) => {
                    Ok(ExpressionType::Float(l.$fname(r)?))
                }
                (ExpressionType::Complex(l), ExpressionType::Complex(r)) => {
                    Ok(ExpressionType::Complex(l.$fname(r)?))
                }
                (ExpressionType::Int(l), ExpressionType::Int(r)) => {
                    Ok(ExpressionType::Int(l.$fname(r)?))
                }
            }
        }
    };
}

/// Implement pass-through methods on the members of ExpressionType
macro_rules! impl_foreach {
    ($fname:ident ( $($arg: ident : $ty: ty ),* ) -> PyResult<Self>, ($firstmeth:ident, $($firstclose:tt)?) $(, ($meth: ident, $close: tt) )*) => {
        fn $fname(&self, $($arg: $ty), *) -> PyResult<Self> {
            match self {
                ExpressionType::Float(e) => Ok(ExpressionType::Float(e.$firstmeth($($arg),*)$($firstclose)?$(.$meth()$close)*)),
                ExpressionType::Complex(e) => Ok(ExpressionType::Complex(e.$firstmeth($($arg),*)$($firstclose)?$(.$meth()$close)*)),
                ExpressionType::Int(e) => Ok(ExpressionType::Int(e.$firstmeth($($arg),*)$($firstclose)?$(.$meth()$close)*)),
            }
        }
    };
    ($fname:ident ( $($arg: ident : $ty: ty ),* ) $( -> $ret: ty)*, ($firstmeth:ident, $($firstclose:tt)?) $(, ($meth: ident, $close: tt) )*) => {
        fn $fname(&self, $($arg: $ty), *) $(-> $ret )* {
            match self {
                ExpressionType::Float(e) => e.$firstmeth($($arg),*)$($firstclose)?$(.$meth()$close)*,
                ExpressionType::Complex(e) => e.$firstmeth($($arg),*)$($firstclose)?$(.$meth()$close)*,
                ExpressionType::Int(e) => e.$firstmeth($($arg),*)$($firstclose)?$(.$meth()$close)*,
            }
        }
    };
}

/// Extracts a numpy array from a python object
fn extract_array<'p, T, D>(
    values: &PyAny,
    py: &'p Python<'p>,
    mintype: NPY_TYPES,
) -> Result<PyReadonlyArray<'p, T, D>, PolynomialError>
where
    PyReadonlyArray<'p, T, D>: pyo3::FromPyObject<'p>,
    D: 'p,
{
    let vals = unsafe { array_from_any(values.to_object(*py), py, mintype) };
    vals.extract::<PyReadonlyArray<T, D>>()
        .map_err(|_| PolynomialError::Other("Could not convert type".to_string()))
}

fn cast_array<T, U, D>(values: &ArrayView<U, D>) -> Result<Array<T, D>, PolynomialError>
where
    D: Dimension,
    U: NumCast + Copy,
    T: NumCast,
    <D as Dimension>::Pattern: NdIndex<D>,
{
    use std::mem::MaybeUninit;
    let mut output = Array::uninit(values.dim());
    for (index, value) in output.indexed_iter_mut() {
        let toval: T = NumCast::from(*values.get(index).ok_or_else(|| {
            PolynomialError::Other("Some kind of crazy lookup error".to_string())
        })?)
        .ok_or_else(|| PolynomialError::Other("Could not convert data type".to_string()))?;
        *value = MaybeUninit::new(toval);
    }

    Ok(unsafe { output.assume_init() })
}

impl ExpressionType {
    impl_foreach!(
        expand () -> PyResult<Self>,
        (expand, ?), (to_expression, ?)
    );
    impl_foreach!(
        deriv_integ (indices: &[isize]) -> PyResult<Self>,
        (deriv_integ, ?)
    );
    impl_foreach!(
        drop_params (indices: &[usize]) -> PyResult<Self>,
        (drop_params, ?)
    );
    impl_foreach!(
        dimension () -> usize,
        (dimension, )
    );
    impl_foreach!(
        shape () -> Vec<usize>,
        (shape, )
    );
    impl_binop!(add);
    impl_binop!(sub);
    impl_binop!(mul);
    impl_binop!(div);
}

impl std::fmt::Display for ExpressionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use ExpressionType::*;
        match self {
            Float(e) => write!(f, "{}, dtype = f64", e),
            Complex(e) => write!(f, "{}, dtype = c64", e),
            Int(e) => write!(f, "{}, dtype = i64", e),
        }
    }
}

/// Converts a python object to a numpy array using the numpy API.
unsafe fn array_from_any<'py>(
    object: PyObject,
    py: &'py Python<'py>,
    descr: NPY_TYPES,
) -> &'py PyAny {
    use numpy::npyffi::objects::PyArray_Descr;
    let mintype: *mut PyArray_Descr = PY_ARRAY_API.PyArray_DescrFromType((descr as u32) as i32);
    let ob_ptr: *mut pyo3::ffi::PyObject = object.as_ptr();
    let desc = PY_ARRAY_API.PyArray_DescrFromObject(ob_ptr, mintype);
    let context: *mut pyo3::ffi::PyObject = std::ptr::null_mut();
    let ar = PY_ARRAY_API.PyArray_FromAny(ob_ptr, desc, 0, 32, 0, context);
    PyAny::from_borrowed_ptr(*py, ar)
}

#[pymethods]
impl ExpressionTree {
    #[new]
    fn new<'py>(py: Python<'py>, vals: &PyAny) -> PyResult<Self> {
        let vals = unsafe { array_from_any(vals.to_object(py), &py, NPY_TYPES::NPY_LONGLONG) };
        if let Ok(vals) = vals.extract::<PyReadonlyArrayDyn<i64>>() {
            let p = crate::polynomial::Polynomial::<i64>::new(vals.as_array().to_owned());
            return Ok(ExpressionTree {
                expression: ExpressionType::Int(Expression::polynomial(&p)),
            });
        }
        let vals = unsafe { array_from_any(vals.to_object(py), &py, NPY_TYPES::NPY_DOUBLE) };
        if let Ok(vals) = vals.extract::<PyReadonlyArrayDyn<f64>>() {
            let p = crate::polynomial::Polynomial::<f64>::new(vals.as_array().to_owned());
            return Ok(ExpressionTree {
                expression: ExpressionType::Float(Expression::polynomial(&p)),
            });
        }
        let vals = unsafe { array_from_any(vals.to_object(py), &py, NPY_TYPES::NPY_CDOUBLE) };
        if let Ok(vals) = vals.extract::<PyReadonlyArrayDyn<Complex<f64>>>() {
            let p = crate::polynomial::Polynomial::<Complex<f64>>::new(vals.as_array().to_owned());
            return Ok(ExpressionTree {
                expression: ExpressionType::Complex(Expression::polynomial(&p)),
            });
        }
        Err(PyValueError::new_err("Unsupported data type returned"))
    }
    /// Evaluates the expression on an array of values.
    ///
    /// Parameters
    /// ==========
    ///
    /// vals : array_like
    ///     Shape (a, b, ..., z, n) where n matches the dimension of the expression tree
    ///
    /// Returns
    /// =======
    /// array_like
    ///     Shape (a, b, ..., z)
    ///
    fn __call__<'py>(&self, py: Python<'py>, vals: &PyAny) -> PyResult<PyObject> {
        use ExpressionType::*;
        fn eval<'py, T>(
            expression: &Expression<T>,
            values: &'py PyReadonlyArrayDyn<T>,
            py: &'py Python<'py>
        ) -> PyResult<PyObject>
            where T: Copy
            + Zero
            + Pow<usize, Output = T>
            + Mul<Output = T>
            + Add<Output = T>
            + Sub<Output = T>
            + Div<Output = T>
            + ScalarOperand
            + Send
            + Sync
            + NumCast
            + Element,
        {
            Ok(expression.eval(&values.as_array())?.into_pyarray(*py).to_object(*py))
        }
        match &self.expression {
            Complex(e) => {
                if let Ok(vals) = extract_array::<num_complex::Complex<f64>, _>(vals, &py, NPY_TYPES::NPY_CDOUBLE) {
                    return eval(e, &vals, &py)
                }
            }
            Float(e) => {
                if let Ok(vals) = extract_array::<f64, _>(vals, &py, NPY_TYPES::NPY_DOUBLE) {
                    return eval::<>(e, &vals, &py)
                }
                if let Ok(vals) = extract_array::<num_complex::Complex<f64>, _>(vals, &py, NPY_TYPES::NPY_CDOUBLE) {
                    return eval::<>(&e.astype()?, &vals, &py)
                }
            }
            Int(e) => {
                if let Ok(vals) = extract_array::<i64, _>(vals, &py, NPY_TYPES::NPY_LONGLONG) {
                    return eval::<>(e, &vals, &py)
                }
                if let Ok(vals) = extract_array::<f64, _>(vals, &py, NPY_TYPES::NPY_DOUBLE) {
                    return eval::<>(&e.astype()?, &vals, &py)
                }
                if let Ok(vals) = extract_array::<num_complex::Complex<f64>, _>(vals, &py, NPY_TYPES::NPY_CDOUBLE) {
                    return eval::<>(&e.astype()?, &vals, &py)
                }
            }
        }
        Err(PyValueError::new_err(
            "Unsupported data type for evaluation",
        ))
    }

    /// The maximum degree of any term in the expression tree
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> &'py PyArray1<usize> {
        self.expression.shape().into_pyarray(py)
    }
    /// The number of free parameters in the expression tree
    #[getter]
    fn dimension(&self) -> usize {
        self.expression.dimension()
    }
    /// Recursively evaluate the syntax tree to yield either a single polynomial
    /// or a rational with a polynomial numerator and denominator.
    ///
    /// Returns
    /// =======
    /// FloatExpression
    #[pyo3(text_signature = "()")]
    fn expand(&self) -> PyResult<Self> {
        Ok(ExpressionTree {
            expression: self.expression.expand()?,
        })
    }
    /// Evaluate the expression tree on a subset of provided terms and return a
    /// new expression tree.
    ///
    /// Parameters
    /// ==========
    /// indicies : array_like
    ///     (n,) shaped array of integers. This argument specifies which terms in the
    ///     expression are to be evaluated.
    /// values : array_like
    ///     (n,) shaped array of floats. This argument specifies the values to use in
    ///     the expression.
    ///
    /// Returns
    /// =======
    /// FloatExpression
    #[pyo3(text_signature = "(indices, values)")]
    fn partial<'py>(&self, py: Python<'py>, indices: &PyAny, values: &PyAny) -> PyResult<Self> {
        let indices: PyReadonlyArray1<i64> = extract_array(indices, &py, NPY_TYPES::NPY_LONGLONG)?;
        // Annoying way of getting an array of usize from an array of isize or error.
        let indices: Array1<usize> = cast_array(&indices.as_array().view())?;

        fn partial_loc<T>(
            expression: &Expression<T>,
            indices: Array1<usize>,
            values: PyReadonlyArray1<T>,
        ) -> Result<Expression<T>, PolynomialError>
        where
            T: Copy + Zero + Mul<Output = T> + Pow<usize, Output = T> + NumCast + Element,
        {

            expression.partial(
                indices.as_slice().ok_or_else(|| {
                    PolynomialError::Other("Could not convert indices to slice".to_string())
                })?,
                values.as_slice().map_err(|_| {
                    PolynomialError::Other("Could not convert values to slice".to_string())
                })?,
            )
        }
        // This could probably be replaced with a macro
        use ExpressionType::*;
        match &self.expression {
            Complex(e) => {
                if let Ok(vals) = extract_array(values, &py, NPY_TYPES::NPY_CDOUBLE) {
                    return Ok(ExpressionTree {
                        expression: Complex(partial_loc(e, indices, vals)?),
                    });
                };
            }
            Float(e) => {
                if let Ok(vals) = extract_array(values, &py, NPY_TYPES::NPY_DOUBLE) {
                    return Ok(ExpressionTree {
                        expression: Float(partial_loc(e, indices, vals)?),
                    });
                };
                if let Ok(vals) = extract_array::<num_complex::Complex<f64>, _>(values, &py, NPY_TYPES::NPY_CDOUBLE) {
                    return Ok(ExpressionTree {
                        expression: Complex(partial_loc(&e.astype()?, indices, vals)?),
                    });
                };
            }
            Int(e) => {
                if let Ok(vals) = extract_array(values, &py, NPY_TYPES::NPY_CDOUBLE) {
                    return Ok(ExpressionTree {
                        expression: Int(partial_loc(e, indices, vals)?),
                    });
                }
                if let Ok(vals) = extract_array::<f64, _>(values, &py, NPY_TYPES::NPY_DOUBLE) {
                    return Ok(ExpressionTree {
                        expression: Float(partial_loc(&e.astype()?, indices, vals)?),
                    });
                };
                if let Ok(vals) = extract_array::<num_complex::Complex<f64>, _>(values, &py, NPY_TYPES::NPY_CDOUBLE) {
                    return Ok(ExpressionTree {
                        expression: Complex(partial_loc(&e.astype()?, indices, vals)?),
                    });
                };
            }
        };
        Err(PyValueError::new_err(
            "Unsupported data type for evaluation",
        ))
    }
    /// Takes the derivative of the expression with respect to multiple
    /// parameters. Zero is a no-op. Negative numbers integrate.
    ///
    /// Parameters
    /// ==========
    /// indices: array_like of int
    ///     shape (n,) where n is the dimension of the expression.
    ///
    /// Returns
    /// =======
    /// FloatExpression
    #[pyo3(text_signature = "(indices)")]
    fn deriv(&self, indices: PyReadonlyArray1<i64>) -> PyResult<Self> {
        let indices = indices.as_array();
        let indices = {
            let mut indices_ = vec![];
            for index in indices {
                indices_.push(*index as isize);
            }
            indices_
        };
        Ok(ExpressionTree {
            expression: self.expression.deriv_integ(&indices)?,
        })
    }
    /// Takes the antiderivative of the expression with respect to multiple
    /// parameters. Zero is a no-op. Negative numbers differentiate.
    ///
    /// Parameters
    /// ==========
    /// indices: array_like of int
    ///     shape (n,) where n is the dimension of the expression.
    ///
    /// Returns
    /// =======
    /// FloatExpression
    #[pyo3(text_signature = "(indices)")]
    fn integ(&self, indices: PyReadonlyArray1<i64>) -> PyResult<Self> {
        let indices = indices.as_array();
        let indices = {
            let mut indices_ = vec![];
            for index in indices {
                indices_.push(-*index as isize);
            }
            indices_
        };
        Ok(ExpressionTree {
            expression: self.expression.deriv_integ(&indices)?,
        })
    }
    /// Try to drop parameters and reduce the dimensionality.
    ///
    /// Parameters
    /// ==========
    /// indices: array_like of int
    ///     shape (m,), specifies degrees of freedom to drop.
    ///
    /// Returns
    /// =======
    /// FloatExpression
    ///
    /// Raises
    /// ======
    /// ValueError
    ///     If non-empty or invalid degrees of freedom are specified.
    #[pyo3(text_signature = "(indices)")]
    fn drop_params(&self, indices: PyReadonlyArray1<i64>) -> PyResult<Self> {
        let indices = indices.as_array();
        let indices = {
            let mut indices_ = vec![];
            for index in indices {
                if *index < 0 {
                    return Err(
                        PolynomialError::Other("Negative indices not allowed".to_string()).into(),
                    );
                }
                indices_.push(*index as usize);
            }
            indices_
        };
        Ok(ExpressionTree {
            expression: self.expression.drop_params(&indices)?,
        })
    }
    /// Automatically drop empty degrees of freedom
    ///
    /// Returns
    /// =======
    /// FloatExpression
    #[pyo3(text_signature = "()")]
    fn squeeze(&self) -> PyResult<Self> {
        let to_drop = self
            .expression
            .shape()
            .iter()
            .enumerate()
            .filter_map(|(i, v)| if *v == 1 { Some(i) } else { None })
            .collect::<Vec<_>>();
        Ok(ExpressionTree {
            expression: self.expression.drop_params(to_drop.as_slice())?,
        })
    }
    /// Try to evaluate the expression and return a constant
    ///
    /// Raises
    /// ======
    ///
    /// ValueError
    ///     If some degrees of freedom still exist
    #[pyo3(text_signature = "()")]
    fn to_constant<'py>(&'py self, py: Python<'py>) -> PyResult<PyObject> {
        use ExpressionType::*;
        match &self.expression {
            Float(e) => {
                let v = e.to_constant()?;
                Ok(v.into_py(py))
            }
            Complex(e) => {
                let v = e.to_constant()?;
                Ok(v.into_py(py))
            }
            Int(e) => {
                let v = e.to_constant()?;
                Ok(v.into_py(py))
            }
        }
        // Ok(self.expression.to_constant()?)
    }
    // TODO Some kind of "Kind"

    /// Construct a polynomial of desired dimension equal to the constant 1.
    ///
    /// Returns
    /// =======
    /// FloatExpression
    #[classmethod]
    #[pyo3(text_signature = "()")]
    fn zero(_cls: &PyType, dimension: i64) -> PyResult<Self> {
        if dimension < 0 {
            return Err(PolynomialError::Other("Negative dimension is invalid".to_string()).into());
        }
        let dimension = dimension as usize;
        Ok(ExpressionTree {
            expression: ExpressionType::Float(Expression::zero(dimension)),
        })
    }
    /// Construct a polynomial of desired dimension equal to the constant 0.
    ///
    /// Returns
    /// =======
    /// FloatExpression
    #[classmethod]
    #[pyo3(text_signature = "()")]
    fn one(_cls: &PyType, dimension: i64) -> PyResult<Self> {
        if dimension < 0 {
            return Err(PolynomialError::Other("Negative dimension is invalid".to_string()).into());
        }
        let dimension = dimension as usize;
        Ok(ExpressionTree {
            expression: ExpressionType::Float(Expression::one(dimension)),
        })
    }
}

#[pyproto]
impl PyObjectProtocol for ExpressionTree {
    fn __str__(&'p self) -> PyResult<String> {
        Ok(format!("{}", self.expression))
    }
}

#[pyproto]
impl PyNumberProtocol for ExpressionTree {
    fn __add__(lhs: Self, rhs: Self) -> PyResult<Self> {
        Ok(ExpressionTree {
            expression: lhs.expression.add(&rhs.expression)?,
        })
    }
    fn __sub__(lhs: Self, rhs: Self) -> PyResult<Self> {
        Ok(ExpressionTree {
            expression: lhs.expression.sub(&rhs.expression)?,
        })
    }
    fn __truediv__(lhs: Self, rhs: &'p PyAny) -> PyResult<Self> {
        if let Ok(ExpressionTree { expression: other }) = rhs.extract() {
            return Ok(ExpressionTree {
                expression: lhs.expression.div(&other)?,
            });
        };

        use ExpressionType::*;
        match lhs.expression {
            Float(e) => {
                if let Ok(v) = rhs.extract::<i64>() {
                    return Ok(ExpressionTree {
                        expression: Float(e.scale(1.0 / v as f64)?),
                    });
                }
                if let Ok(v) = rhs.extract::<f64>() {
                    return Ok(ExpressionTree {
                        expression: Float(e.scale(1.0 / v)?),
                    });
                }
                if let Ok(v) = rhs.extract::<num_complex::Complex<f64>>() {
                    return Ok(ExpressionTree {
                        expression: Complex(e.scale(1.0 / v)?),
                    });
                }
            }
            Complex(e) => {
                if let Ok(v) = rhs.extract::<i64>() {
                    return Ok(ExpressionTree {
                        expression: Complex(e.scale(1.0 / v as f64)?),
                    });
                }
                if let Ok(v) = rhs.extract::<f64>() {
                    return Ok(ExpressionTree {
                        expression: Complex(e.scale(1.0 / v)?),
                    });
                }
                if let Ok(v) = rhs.extract::<num_complex::Complex<f64>>() {
                    return Ok(ExpressionTree {
                        expression: Complex(e.scale(1.0 / v)?),
                    });
                }
            }
            Int(e) => {
                if let Ok(v) = rhs.extract::<i64>() {
                    return Ok(ExpressionTree {
                        expression: Float(e.scale(1.0 / v as f64)?),
                    });
                }
                if let Ok(v) = rhs.extract::<f64>() {
                    return Ok(ExpressionTree {
                        expression: Float(e.scale(1.0 / v)?),
                    });
                }
                if let Ok(v) = rhs.extract::<num_complex::Complex<f64>>() {
                    return Ok(ExpressionTree {
                        expression: Complex(e.scale(1.0 / v)?),
                    });
                }
            }
        };
        Err(PyValueError::new_err("Invalid type provided for division"))
    }
    fn __mul__(lhs: Self, rhs: &'p PyAny) -> PyResult<Self> {
        if let Ok(ExpressionTree { expression: other }) = rhs.extract() {
            return Ok(ExpressionTree {
                expression: lhs.expression.mul(&other)?,
            });
        };
        use ExpressionType::*;
        match lhs.expression {
            Float(e) => {
                if let Ok(v) = rhs.extract::<i64>() {
                    return Ok(ExpressionTree {
                        expression: Float(e.scale(v)?),
                    });
                }
                if let Ok(v) = rhs.extract::<f64>() {
                    return Ok(ExpressionTree {
                        expression: Float(e.scale(v)?),
                    });
                }
                if let Ok(v) = rhs.extract::<num_complex::Complex<f64>>() {
                    return Ok(ExpressionTree {
                        expression: Complex(e.scale(v)?),
                    });
                }
            }
            Complex(e) => {
                if let Ok(v) = rhs.extract::<i64>() {
                    return Ok(ExpressionTree {
                        expression: Complex(e.scale(v)?),
                    });
                }
                if let Ok(v) = rhs.extract::<f64>() {
                    return Ok(ExpressionTree {
                        expression: Complex(e.scale(v)?),
                    });
                }
                if let Ok(v) = rhs.extract::<num_complex::Complex<f64>>() {
                    return Ok(ExpressionTree {
                        expression: Complex(e.scale(v)?),
                    });
                }
            }
            Int(e) => {
                if let Ok(v) = rhs.extract::<i64>() {
                    return Ok(ExpressionTree {
                        expression: Int(e.scale(v)?),
                    });
                }
                if let Ok(v) = rhs.extract::<f64>() {
                    return Ok(ExpressionTree {
                        expression: Float(e.scale(v)?),
                    });
                }
                if let Ok(v) = rhs.extract::<num_complex::Complex<f64>>() {
                    return Ok(ExpressionTree {
                        expression: Complex(e.scale(v)?),
                    });
                }
            }
        };
        Err(PyValueError::new_err(
            "Invalid type provided for multiplication",
        ))
    }

    fn __rmul__(&self, other: &'p PyAny) -> PyResult<Self> {
        Self::__mul__(self.clone(), other)
    }
}

/// Provides the ExpressionTree class
#[pymodule]
#[pyo3(name = "rust_poly")]
fn polynomial(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ExpressionTree>()?;
    Ok(())
}
