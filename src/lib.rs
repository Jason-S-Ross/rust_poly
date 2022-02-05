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

use numpy::{IntoPyArray, PyArray1, PyArrayDyn, PyReadonlyArray1, PyReadonlyArrayDyn};
use polynomial::PolynomialError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;
use pyo3::{PyNumberProtocol, PyObjectProtocol};
use tree::Expression;
pub mod polynomial;
pub mod tree;

/// An abstract expression tree with concrete terms. Lazily evaluates arithmetic
/// operations.
#[derive(Clone)]
#[pyclass]
#[pyo3(text_signature = "(vals)")]
struct FloatExpression {
    expression: crate::tree::Expression<f64>,
}
#[pymethods]
impl FloatExpression {
    #[new]
    fn new(vals: PyReadonlyArrayDyn<f64>) -> Self {
        let p = crate::polynomial::Polynomial::<f64>::new(vals.as_array().to_owned());
        FloatExpression {
            expression: Expression::polynomial(&p),
        }
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
    /// array_like
    ///     Shape (a, b, ..., z)
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        vals: PyReadonlyArrayDyn<'_, f64>,
    ) -> PyResult<&'py PyArrayDyn<f64>> {
        let vals = vals.as_array();
        Ok(self.expression.eval(&vals)?.into_pyarray(py))
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
        Ok(FloatExpression {
            expression: self.expression.expand()?.to_expression()?,
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
    fn partial<'py>(
        &self,
        _py: Python<'py>,
        indices: PyReadonlyArray1<i64>,
        values: PyReadonlyArray1<f64>,
    ) -> PyResult<Self> {
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
        let values = values.as_array();
        let values = values
            .as_slice()
            .ok_or_else(|| PolynomialError::Other("Failed to process value slice".to_string()))?;
        Ok(FloatExpression {
            expression: self.expression.partial(&indices, values)?,
        })
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
        Ok(FloatExpression {
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
        Ok(FloatExpression {
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
        Ok(FloatExpression {
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
        Ok(FloatExpression {
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
    fn to_constant(&self) -> PyResult<f64> {
        Ok(self.expression.to_constant()?)
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
        Ok(FloatExpression {
            expression: Expression::zero(dimension),
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
        Ok(FloatExpression {
            expression: Expression::one(dimension),
        })
    }
}

#[pyproto]
impl PyObjectProtocol for FloatExpression {
    fn __str__(&'p self) -> PyResult<String> {
        Ok(format!("{}", self.expression))
    }
}

#[pyproto]
impl PyNumberProtocol for FloatExpression {
    fn __add__(lhs: Self, rhs: Self) -> PyResult<Self> {
        Ok(FloatExpression {
            expression: lhs.expression.add(&rhs.expression)?,
        })
    }
    fn __sub__(lhs: Self, rhs: Self) -> PyResult<Self> {
        Ok(FloatExpression {
            expression: lhs.expression.sub(&rhs.expression)?,
        })
    }
    fn __truediv__(lhs: Self, rhs: &'p PyAny) -> PyResult<Self> {
        if let Ok(FloatExpression { expression: other }) = rhs.extract() {
            return Ok(FloatExpression {
                expression: lhs.expression.div(&other)?,
            });
        };
        if let Ok(v) = rhs.extract::<f64>() {
            return Ok(FloatExpression {
                expression: lhs.expression.scale(1.0 / v)?,
            });
        };
        Err(PyValueError::new_err("Invalid type provided for division"))
    }
    fn __mul__(lhs: Self, rhs: &'p PyAny) -> PyResult<Self> {
        if let Ok(FloatExpression { expression: other }) = rhs.extract() {
            return Ok(FloatExpression {
                expression: lhs.expression.mul(&other)?,
            });
        };
        if let Ok(v) = rhs.extract::<f64>() {
            return Ok(FloatExpression {
                expression: lhs.expression.scale(v)?,
            });
        };
        Err(PyValueError::new_err(
            "Invalid type provided for multiplication",
        ))
    }

    fn __rmul__(&self, other: &'p PyAny) -> PyResult<Self> {
        if let Ok(FloatExpression { expression: other }) = other.extract() {
            return Ok(FloatExpression {
                expression: self.expression.mul(&other)?,
            });
        };
        if let Ok(v) = other.extract::<f64>() {
            return Ok(FloatExpression {
                expression: self.expression.scale(v)?,
            });
        };
        Err(PyValueError::new_err(
            "Invalid type provided for right multiplication",
        ))
    }
}

/// Provides the FloatExpression class
#[pymodule]
#[pyo3(name = "rust_poly")]
fn polynomial(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<FloatExpression>()?;
    Ok(())
}
