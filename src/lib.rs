use numpy::{IntoPyArray, PyArray1, PyArrayDyn, PyReadonlyArray1, PyReadonlyArrayDyn};
// use numpy::ndarray::{Array, ArrayD, Array1, Dimension, ShapeBuilder, Axis, Dim, IxDyn};
use numpy::convert::ToPyArray;
use polynomial::PolynomialError;
use pyo3::class::number::{
    PyNumberAddProtocol, PyNumberMulProtocol, PyNumberSubProtocol, PyNumberTruedivProtocol,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyTuple, PyType};
use pyo3::{exceptions, PyNumberProtocol, PyObjectProtocol};
use tree::Expression;
pub mod polynomial;
pub mod tree;

#[derive(Clone)]
#[pyclass]
struct FloatExpandedExpression {
    expression: crate::tree::ExpandedExpression<f64>,
}

#[derive(Clone)]
#[pyclass]
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
    /// Evaluates the polynomial on an array of values
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        vals: PyReadonlyArrayDyn<'_, f64>,
    ) -> PyResult<&'py PyArrayDyn<f64>> {
        let vals = vals.as_array();
        Ok(self.expression.eval(&vals)?.into_pyarray(py))
    }

    /// The degree of polynomial
    fn shape<'py>(&self, py: Python<'py>) -> &'py PyArray1<usize> {
        self.expression.shape().into_pyarray(py)
    }
    /// Recursively expands the syntax tree to yield either a polynomial or a rational
    fn expand(&self) -> PyResult<Self> {
        Ok(FloatExpression {
            expression: self.expression.expand()?.to_expression()?,
        })
    }
    /// Evaluate the polynomial on values
    fn partial<'py>(
        &self,
        py: Python<'py>,
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
        let values = values.as_slice().ok_or(PolynomialError::Other(
            "Failed to process value slice".to_string(),
        ))?;
        Ok(FloatExpression {
            expression: self.expression.partial(&indices, &values)?,
        })
    }
    /// Derivative (negative numbers integrate)
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
    /// Integrate (negative numbers differentiate)
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
    /// Drop parameters and reduce the dimensionality
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
    /// Drops single-dimension axes
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
    /// Try to reduce to a constant
    fn to_constant(&self) -> PyResult<f64> {
        return Ok(self.expression.to_constant()?);
    }
    // TODO Some kind of "Kind"
    #[classmethod]
    fn zero(cls: &PyType, dimension: i64) -> PyResult<Self> {
        if dimension < 0 {
            return Err(PolynomialError::Other("Negative dimension is invalid".to_string()).into());
        }
        let dimension = dimension as usize;
        Ok(FloatExpression {
            expression: Expression::zero(dimension),
        })
    }
    #[classmethod]
    fn one(cls: &PyType, dimension: i64) -> PyResult<Self> {
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
        return Err(PyValueError::new_err("Invalid type provided for division"));
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
        return Err(PyValueError::new_err("Invalid type provided for division"));
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
        return Err(PyValueError::new_err("Invalid type provided for division"));
    }
}

#[pymodule]
#[pyo3(name = "rust_poly")]
fn polynomial(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<FloatExpression>()?;
    m.add_class::<FloatExpandedExpression>()?;
    Ok(())
}
