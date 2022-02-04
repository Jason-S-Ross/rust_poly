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

use crate::polynomial::{pow, Polynomial, PolynomialError};
use ahash::AHashSet;
use ndarray::{ArrayD, ArrayViewD, ScalarOperand};
use num_traits::cast::NumCast;
use num_traits::identities::{One, Zero};
use num_traits::pow::Pow;
use std::ops::{Add, Div, Mul, Sub};

pub fn pow_array<T>(array: ArrayD<T>, exp: usize) -> ArrayD<T>
where
    T: Mul<Output = T> + Copy + One + Pow<u16, Output = T>,
{
    let mut output = array;
    for val in output.iter_mut() {
        *val = pow(*val, exp);
    }
    output
}

/// A lazy polynomial expression
#[derive(Debug, Clone)]
pub enum Expression<T> {
    Polynomial(Polynomial<T>),
    Add {
        left: Box<Expression<T>>,
        right: Box<Expression<T>>,
    },
    Sub {
        left: Box<Expression<T>>,
        right: Box<Expression<T>>,
    },
    Mul {
        left: Box<Expression<T>>,
        right: Box<Expression<T>>,
    },
    Div {
        num: Box<Expression<T>>,
        denom: Box<Expression<T>>,
    },
    Scale {
        scale: T,
        expression: Box<Expression<T>>,
    },
    DerivInteg {
        expression: Box<Expression<T>>,
        wrt: Vec<isize>,
    },
    Pow {
        expression: Box<Expression<T>>,
        power: usize,
    },
}

type CreateResult<T> = Result<Expression<T>, PolynomialError>;
#[derive(Debug, Clone)]
pub enum ExpandedExpression<T> {
    Polynomial(Polynomial<T>),
    Rational {
        num: Polynomial<T>,
        denom: Polynomial<T>,
    },
}

impl<T> ExpandedExpression<T>
where
    T: Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + ScalarOperand
        + Copy
        + Zero
        + One
        + NumCast
        + Pow<u16, Output = T>
        + Send
        + Sync,
{
    pub fn to_expression(&self) -> Result<Expression<T>, PolynomialError> {
        match self {
            ExpandedExpression::Polynomial(p) => Ok(Expression::Polynomial(p.clone())),
            ExpandedExpression::Rational { num, denom } => {
                Ok(Expression::Polynomial(num.clone())
                    .div(&Expression::Polynomial(denom.clone()))?)
            }
        }
    }
}

impl<T> Expression<T>
where
    T: Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + ScalarOperand
        + Copy
        + Zero
        + One
        + NumCast
        + Pow<u16, Output = T>
        + Send
        + Sync,
{
    #[inline]
    pub fn polynomial(p: &Polynomial<T>) -> Self {
        Expression::Polynomial(p.clone())
    }
    #[inline]
    pub fn add(&self, right: &Self) -> CreateResult<T> {
        if self.dimension() != right.dimension() {
            return Err(PolynomialError::Composition);
        }
        Ok(Expression::Add {
            left: Box::new(self.clone()),
            right: Box::new(right.clone()),
        })
    }
    #[inline]
    pub fn sub(&self, right: &Self) -> CreateResult<T> {
        if self.dimension() != right.dimension() {
            return Err(PolynomialError::Composition);
        }
        Ok(Expression::Sub {
            left: Box::new(self.clone()),
            right: Box::new(right.clone()),
        })
    }
    #[inline]
    pub fn mul(&self, right: &Self) -> CreateResult<T> {
        if self.dimension() != right.dimension() {
            return Err(PolynomialError::Composition);
        }
        Ok(Expression::Mul {
            left: Box::new(self.clone()),
            right: Box::new(right.clone()),
        })
    }
    #[inline]
    pub fn div(&self, denom: &Self) -> CreateResult<T> {
        if self.dimension() != denom.dimension() {
            return Err(PolynomialError::Composition);
        }
        Ok(Expression::Div {
            num: Box::new(self.clone()),
            denom: Box::new(denom.clone()),
        })
    }
    #[inline]
    pub fn scale(&self, scale: T) -> CreateResult<T> {
        Ok(Expression::Scale {
            scale,
            expression: Box::new(self.clone()),
        })
    }
    #[inline]
    pub fn deriv_integ(&self, wrt: &[isize]) -> CreateResult<T> {
        Ok(Expression::DerivInteg {
            expression: Box::new(self.clone()),
            wrt: wrt.to_vec(),
        })
    }
    #[inline]
    pub fn pow(&self, power: usize) -> CreateResult<T> {
        Ok(Expression::Pow {
            expression: Box::new(self.clone()),
            power,
        })
    }
    #[inline]
    pub fn dimension(&self) -> usize {
        use Expression::*;
        match self {
            Polynomial(p) => p.dimension(),
            Add { left, right: _ } => left.dimension(),
            Sub { left, right: _ } => left.dimension(),
            Mul { left, right: _ } => left.dimension(),
            Div { num, denom: _ } => num.dimension(),
            Scale {
                scale: _,
                expression,
            } => expression.dimension(),
            DerivInteg { expression, wrt: _ } => expression.dimension(),
            Pow {
                expression,
                power: _,
            } => expression.dimension(),
        }
    }

    #[inline]
    pub fn drop_params(&self, to_drop: &[usize]) -> Result<Self, PolynomialError> {
        use Expression::*;
        match self {
            Polynomial(p) => Ok(Polynomial(p.drop_params(to_drop)?)),
            Add { left, right } => Ok(Add {
                left: Box::new(left.drop_params(to_drop)?),
                right: Box::new(right.drop_params(to_drop)?),
            }),
            Sub { left, right } => Ok(Sub {
                left: Box::new(left.drop_params(to_drop)?),
                right: Box::new(right.drop_params(to_drop)?),
            }),
            Mul { left, right } => Ok(Mul {
                left: Box::new(left.drop_params(to_drop)?),
                right: Box::new(right.drop_params(to_drop)?),
            }),
            Div { num, denom } => Ok(Div {
                num: Box::new(num.drop_params(to_drop)?),
                denom: Box::new(denom.drop_params(to_drop)?),
            }),
            Scale { scale, expression } => Ok(Scale {
                scale: *scale,
                expression: Box::new(expression.drop_params(to_drop)?),
            }),
            DerivInteg { expression, wrt } => Ok(DerivInteg {
                wrt: wrt.clone(),
                expression: Box::new(expression.drop_params(to_drop)?),
            }),
            Pow { expression, power } => Ok(Pow {
                expression: Box::new(expression.drop_params(to_drop)?),
                power: *power,
            }),
        }
    }
    #[inline]
    pub fn shape(&self) -> Vec<usize> {
        use Expression::*;
        match self {
            Polynomial(p) => p.shape(),
            Add { left, right } => left
                .shape()
                .iter()
                .zip(right.shape().iter())
                .map(|(x, y)| if x > y { *x } else { *y })
                .collect(),
            Sub { left, right } => left
                .shape()
                .iter()
                .zip(right.shape().iter())
                .map(|(x, y)| if x > y { *x } else { *y })
                .collect(),
            Mul { left, right } => left
                .shape()
                .iter()
                .zip(right.shape().iter())
                .map(|(x, y)| if x > y { *x } else { *y })
                .collect(),
            Div { num, denom } => num
                .shape()
                .iter()
                .zip(denom.shape().iter())
                .map(|(x, y)| if x > y { *x } else { *y })
                .collect(),
            Scale {
                scale: _,
                expression,
            } => expression.shape(),
            DerivInteg { expression, wrt: _ } => expression.shape(),
            Pow {
                expression,
                power: _,
            } => expression.shape(),
        }
    }
    #[inline]
    pub fn to_constant(&self) -> Result<T, PolynomialError> {
        use Expression::*;
        match self {
            Polynomial(p) => p.to_constant(),
            Div { num, denom } => Ok(num.to_constant()? / denom.to_constant()?),
            Scale { scale, expression } => Ok(*scale * expression.to_constant()?),
            Pow { expression, power } => Ok(pow(expression.to_constant()?, *power)),
            _ => Err(PolynomialError::Other(
                "Attempted to get constant value on non-atomic expression. Try expanding?"
                    .to_string(),
            )),
        }
    }
    /// The variables in this polynomial that have non-zero power
    #[inline]
    pub fn dofs(&self) -> AHashSet<usize> {
        match self {
            Expression::Polynomial(p) => p.dofs(),
            Expression::Add { left, right } => left.dofs().union(&right.dofs()).cloned().collect(),
            Expression::Sub { left, right } => left.dofs().union(&right.dofs()).cloned().collect(),
            Expression::Mul { left, right } => left.dofs().union(&right.dofs()).cloned().collect(),
            Expression::Div { num, denom } => num.dofs().union(&denom.dofs()).cloned().collect(),
            Expression::Scale {
                scale: _,
                expression,
            } => expression.dofs(),
            Expression::DerivInteg { expression, wrt: _ } => expression.dofs(),
            Expression::Pow {
                expression,
                power: _,
            } => expression.dofs(),
        }
    }
    #[inline]
    pub fn zero(dimension: usize) -> Self {
        Expression::Polynomial(Polynomial::zero(dimension))
    }
    #[inline]
    pub fn one(dimension: usize) -> Self {
        Expression::Polynomial(Polynomial::one(dimension))
    }
    #[inline]
    pub fn eval(&self, values: &ArrayViewD<T>) -> Result<ArrayD<T>, PolynomialError> {
        use Expression::*;
        match self {
            Polynomial(p) => p.eval(values),
            Add { left, right } => Ok(left.eval(values)? + right.eval(values)?),
            Sub { left, right } => Ok(left.eval(values)? - right.eval(values)?),
            Mul { left, right } => Ok(left.eval(values)? * right.eval(values)?),
            Div { num, denom } => Ok(num.eval(values)? / denom.eval(values)?),
            Scale { scale, expression } => {
                let res = expression.eval(values)?;
                Ok(res * *scale)
            }
            DerivInteg { expression, wrt } => expression.deriv_integ_eval(wrt.as_slice(), values),
            Pow { expression, power } => {
                let res = expression.eval(values)?;
                Ok(pow_array(res, *power))
            }
        }
    }
    #[inline]
    pub fn partial(&self, indices: &[usize], values: &[T]) -> Result<Self, PolynomialError> {
        use Expression::*;
        match self {
            Polynomial(p) => Ok(Polynomial(p.partial(indices, values)?)),
            Add { left, right } => Ok(Add {
                left: Box::new(left.partial(indices, values)?),
                right: Box::new(right.partial(indices, values)?),
            }),
            Sub { left, right } => Ok(Sub {
                left: Box::new(left.partial(indices, values)?),
                right: Box::new(right.partial(indices, values)?),
            }),
            Mul { left, right } => Ok(Mul {
                left: Box::new(left.partial(indices, values)?),
                right: Box::new(right.partial(indices, values)?),
            }),
            Div { num, denom } => Ok(Div {
                num: Box::new(num.partial(indices, values)?),
                denom: Box::new(denom.partial(indices, values)?),
            }),
            Scale { scale, expression } => Ok(Scale {
                scale: *scale,
                expression: Box::new(expression.partial(indices, values)?),
            }),
            DerivInteg { expression, wrt } => Ok(expression
                .deriv_integ_expand(wrt)?
                .partial(indices, values)?),
            Pow { expression, power } => Ok(Pow {
                expression: Box::new(expression.partial(indices, values)?),
                power: *power,
            }),
        }
    }
    #[inline]
    fn deriv_integ_eval(
        &self,
        wrt: &[isize],
        values: &ArrayViewD<T>,
    ) -> Result<ArrayD<T>, PolynomialError> {
        use Expression::*;
        match self {
            Polynomial(p) => p.deriv_integ(wrt)?.eval(values),
            Add { left, right } => {
                Ok(left.deriv_integ_eval(wrt, values)? + right.deriv_integ_eval(wrt, values)?)
            }
            Sub { left, right } => {
                Ok(left.deriv_integ_eval(wrt, values)? - right.deriv_integ_eval(wrt, values)?)
            }
            Mul { left, right } => Ok(left.deriv_integ_eval(wrt, values)? * right.eval(values)?
                + left.eval(values)? * right.deriv_integ_eval(wrt, values)?),
            Div { num: _, denom: _ } => todo!(),
            Scale { scale, expression } => Ok(expression.deriv_integ_eval(wrt, values)? * *scale),
            DerivInteg {
                expression,
                wrt: other_wrt,
            } => expression.deriv_integ_eval(
                wrt.iter()
                    .zip(other_wrt)
                    .map(|(x, y)| x + y)
                    .collect::<Vec<_>>()
                    .as_slice(),
                values,
            ),
            Pow {
                expression: _,
                power: _,
            } => todo!(),
        }
    }
    #[inline]
    fn deriv_integ_expand(&self, wrt: &[isize]) -> Result<Expression<T>, PolynomialError> {
        use Expression::*;
        match self {
            Polynomial(p) => Ok(Polynomial(p.deriv_integ(wrt)?)),
            Add { left, right } => Ok(Add {
                left: Box::new(left.deriv_integ_expand(wrt)?),
                right: Box::new(right.deriv_integ_expand(wrt)?),
            }),
            Sub { left, right } => Ok(Sub {
                left: Box::new(left.deriv_integ_expand(wrt)?),
                right: Box::new(right.deriv_integ_expand(wrt)?),
            }),
            Mul { left, right } => Ok(Add {
                left: Box::new(Mul {
                    left: Box::new(left.deriv_integ_expand(wrt)?),
                    right: right.clone(),
                }),
                right: Box::new(Mul {
                    left: Box::new(right.deriv_integ_expand(wrt)?),
                    right: left.clone(),
                }),
            }),
            Div { num, denom } => {
                let wrt_hash: AHashSet<_> = wrt
                    .iter()
                    .enumerate()
                    .filter_map(|(i, x)| if *x > 0 { Some(i) } else { None })
                    .collect();
                let denom_dofs = denom.dofs();
                if wrt_hash.intersection(&denom_dofs).count() == 0 {
                    Ok(Div {
                        num: Box::new(Sub {
                            left: Box::new(Mul {
                                left: Box::new(num.deriv_integ(wrt)?),
                                right: denom.clone(),
                            }),
                            right: Box::new(Mul {
                                left: Box::new(denom.deriv_integ(wrt)?),
                                right: num.clone(),
                            }),
                        }),
                        denom: Box::new(Pow {
                            expression: denom.clone(),
                            power: 2,
                        }),
                    })
                } else {
                    Err(PolynomialError::Other("Can't perform integral/derivative operation if active dofs are in denominator".to_string()))
                }
            }
            Scale { scale, expression } => Ok(Scale {
                scale: *scale,
                expression: Box::new(expression.deriv_integ_expand(wrt)?),
            }),
            DerivInteg {
                expression,
                wrt: inner_wrt,
            } => {
                let wrt = wrt
                    .iter()
                    .zip(inner_wrt)
                    .map(|(x, y)| x + y)
                    .collect::<Vec<_>>();
                Ok(expression.deriv_integ_expand(wrt.as_slice())?)
            }
            Pow { expression, power } => {
                let scale: T = <T as NumCast>::from(power - 1)
                    .ok_or_else(|| PolynomialError::Other("Couldn't convert power".to_string()))?;
                Ok(Mul {
                    left: Box::new(Scale {
                        scale,
                        expression: expression.clone(),
                    }),
                    right: Box::new(expression.deriv_integ_expand(wrt)?),
                })
            }
        }
    }
    /// Expands all compositions as you go
    #[inline]
    pub fn expand(&self) -> Result<ExpandedExpression<T>, PolynomialError> {
        use Expression::*;
        match self {
            Polynomial(p) => Ok(ExpandedExpression::Polynomial(p.clone())),
            Add { left, right } => match (left.expand()?, right.expand()?) {
                (ExpandedExpression::Polynomial(left), ExpandedExpression::Polynomial(right)) => {
                    Ok(ExpandedExpression::Polynomial(left.add(&right)?))
                }
                (
                    ExpandedExpression::Polynomial(p),
                    ExpandedExpression::Rational { num, denom },
                )
                | (
                    ExpandedExpression::Rational { num, denom },
                    ExpandedExpression::Polynomial(p),
                ) => Ok(ExpandedExpression::Rational {
                    num: p.mul(&denom)?.add(&num)?,
                    denom: denom.clone(),
                }),
                (
                    ExpandedExpression::Rational {
                        num: lnum,
                        denom: ldenom,
                    },
                    ExpandedExpression::Rational {
                        num: rnum,
                        denom: rdenom,
                    },
                ) => Ok(ExpandedExpression::Rational {
                    num: lnum.mul(&rdenom)?.add(&rnum.mul(&ldenom)?)?,
                    denom: ldenom.mul(&rdenom)?,
                }),
            },
            Sub { left, right } => match (left.expand()?, right.expand()?) {
                (ExpandedExpression::Polynomial(left), ExpandedExpression::Polynomial(right)) => {
                    Ok(ExpandedExpression::Polynomial(left.sub(&right)?))
                }
                (
                    ExpandedExpression::Polynomial(p),
                    ExpandedExpression::Rational { num, denom },
                ) => Ok(ExpandedExpression::Rational {
                    num: p.mul(&denom)?.sub(&num)?,
                    denom: denom.clone(),
                }),
                (
                    ExpandedExpression::Rational { num, denom },
                    ExpandedExpression::Polynomial(p),
                ) => Ok(ExpandedExpression::Rational {
                    num: num.sub(&p.mul(&denom)?)?,
                    denom: denom.clone(),
                }),
                (
                    ExpandedExpression::Rational {
                        num: lnum,
                        denom: ldenom,
                    },
                    ExpandedExpression::Rational {
                        num: rnum,
                        denom: rdenom,
                    },
                ) => Ok(ExpandedExpression::Rational {
                    num: lnum.mul(&rdenom)?.sub(&rnum.mul(&ldenom)?)?,
                    denom: ldenom.mul(&rdenom)?,
                }),
            },
            Mul { left, right } => match (left.expand()?, right.expand()?) {
                (ExpandedExpression::Polynomial(left), ExpandedExpression::Polynomial(right)) => {
                    Ok(ExpandedExpression::Polynomial(left.mul(&right)?))
                }
                (
                    ExpandedExpression::Polynomial(p),
                    ExpandedExpression::Rational { num, denom },
                )
                | (
                    ExpandedExpression::Rational { num, denom },
                    ExpandedExpression::Polynomial(p),
                ) => Ok(ExpandedExpression::Rational {
                    num: p.mul(&num)?,
                    denom,
                }),
                (
                    ExpandedExpression::Rational {
                        num: lnum,
                        denom: ldenom,
                    },
                    ExpandedExpression::Rational {
                        num: rnum,
                        denom: rdenom,
                    },
                ) => Ok(ExpandedExpression::Rational {
                    num: lnum.mul(&rnum)?,
                    denom: ldenom.mul(&rdenom)?,
                }),
            },
            Div { num, denom } => match (num.expand()?, denom.expand()?) {
                (ExpandedExpression::Polynomial(num), ExpandedExpression::Polynomial(denom)) => {
                    Ok(ExpandedExpression::Rational { num, denom })
                }
                (
                    ExpandedExpression::Polynomial(p),
                    ExpandedExpression::Rational { num, denom },
                ) => Ok(ExpandedExpression::Rational {
                    num: p.mul(&denom)?,
                    denom: num,
                }),
                (
                    ExpandedExpression::Rational { num, denom },
                    ExpandedExpression::Polynomial(p),
                ) => Ok(ExpandedExpression::Rational {
                    num,
                    denom: denom.mul(&p)?,
                }),
                (
                    ExpandedExpression::Rational {
                        num: lnum,
                        denom: ldenom,
                    },
                    ExpandedExpression::Rational {
                        num: rnum,
                        denom: rdenom,
                    },
                ) => Ok(ExpandedExpression::Rational {
                    num: lnum.mul(&rdenom)?,
                    denom: ldenom.mul(&rnum)?,
                }),
            },
            Scale { scale, expression } => match expression.expand()? {
                ExpandedExpression::Polynomial(p) => {
                    Ok(ExpandedExpression::Polynomial(p.scale(*scale)?))
                }
                ExpandedExpression::Rational { num, denom } => Ok(ExpandedExpression::Rational {
                    num: num.scale(*scale)?,
                    denom,
                }),
            },
            DerivInteg { expression, wrt } => expression.deriv_integ_expand(wrt)?.expand(),
            Pow { expression, power } => match expression.expand()? {
                ExpandedExpression::Polynomial(p) => {
                    Ok(ExpandedExpression::Polynomial(p.pow(*power)?))
                }
                ExpandedExpression::Rational { num, denom } => Ok(ExpandedExpression::Rational {
                    num: num.pow(*power)?,
                    denom: denom.pow(*power)?,
                }),
            },
        }
    }
}

impl<T> std::fmt::Display for Expression<T>
where
    T: std::fmt::Display + Zero + One,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let s = match self {
            Expression::Polynomial(p) => format!("{}", p),
            Expression::Add { left, right } => format!("({}) + ({})", left, right),
            Expression::Sub { left, right } => format!("({}) - ({})", left, right),
            Expression::Mul { left, right } => format!("({}) * ({})", left, right),
            Expression::Div { num, denom } => format!("({}) / ({})", num, denom),
            Expression::Scale { scale, expression } => format!("{} * ({})", scale, expression),
            Expression::DerivInteg { expression, wrt } => {
                format!("deriv({}, [{:?}])", expression, wrt)
            }
            Expression::Pow { expression, power } => format!("pow({}, {})", expression, power),
        };
        write!(f, "{}", s)
    }
}
