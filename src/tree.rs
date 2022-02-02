use crate::polynomial::{pow, Polynomial, PolynomialError};
use bitvec::prelude::*;
use ndarray::{
    Array, ArrayBase, ArrayD, ArrayView, Dim, Dimension, IntoDimension, IxDynImpl, NdIndex,
    OwnedRepr, ScalarOperand, SliceArg, SliceInfoElem,
};
use num_traits::cast::NumCast;
use num_traits::identities::{One, Zero};
use std::ops::{Add, Div, Mul, Sub};

pub fn pow_array<T>(array: ArrayD<T>, exp: usize) -> ArrayD<T>
where
    T: Mul<Output = T> + Copy + One,
{
    let mut output = array.clone();
    for val in output.iter_mut() {
        *val = pow(val.clone(), exp);
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

#[derive(Debug, Clone)]
pub enum ExpandedExpression<T> {
    Polynomial(Polynomial<T>),
    Rational {
        num: Polynomial<T>,
        denom: Polynomial<T>,
    },
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
        + NumCast,
{
    pub fn dimension(&self) -> usize {
        todo!()
    }
    pub fn zero(dimension: usize) -> Self {
        Expression::Polynomial(Polynomial::zero(dimension))
    }
    pub fn one(dimension: usize) -> Self {
        Expression::Polynomial(Polynomial::one(dimension))
    }
    pub fn eval<D>(
        &self,
        values: &ArrayView<T, D>,
    ) -> Result<Array<T, Dim<IxDynImpl>>, PolynomialError>
    where
        D: Dimension,
        Dim<IxDynImpl>: NdIndex<D>,
        [SliceInfoElem]: SliceArg<D>,
        // ArrayBase<OwnedRepr<T>, Dim<IxDynImpl>>: One,
    {
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
            DerivInteg { expression, wrt } => expression.deriv_integ_eval(&wrt.as_slice(), values),
            Pow { expression, power } => {
                let res = expression.eval(values)?;
                Ok(pow_array(res, *power))
            }
        }
    }
    fn deriv_integ_eval<D>(
        &self,
        wrt: &[isize],
        values: &ArrayView<T, D>,
    ) -> Result<Array<T, Dim<IxDynImpl>>, PolynomialError>
    where
        D: Dimension,
        Dim<IxDynImpl>: NdIndex<D>,
        [SliceInfoElem]: SliceArg<D>,
        // ArrayBase<OwnedRepr<T>, Dim<IxDynImpl>>: One,
    {
        use Expression::*;
        match self {
            Polynomial(p) => p.deriv_integ(&wrt)?.eval(values),
            Add { left, right } => {
                Ok(left.deriv_integ_eval(&wrt, values)? + right.deriv_integ_eval(&wrt, values)?)
            }
            Sub { left, right } => {
                Ok(left.deriv_integ_eval(&wrt, values)? - right.deriv_integ_eval(&wrt, values)?)
            }
            Mul { left, right } => Ok(left.deriv_integ_eval(&wrt, values)? * right.eval(values)?
                + left.eval(values)? * right.deriv_integ_eval(&wrt, values)?),
            Div { num, denom } => todo!(),
            Scale { scale, expression } => Ok(expression.deriv_integ_eval(&wrt, values)? * *scale),
            DerivInteg {
                expression,
                wrt: other_wrt,
            } => expression.deriv_integ_eval(
                &wrt.iter()
                    .zip(other_wrt)
                    .map(|(x, y)| x + y)
                    .collect::<Vec<_>>()
                    .as_slice(),
                values,
            ),
            Pow { expression, power } => todo!(),
        }
    }
    fn deriv_integ(&self, wrt: &[isize]) -> Result<Expression<T>, PolynomialError> {
        use Expression::*;
        match self {
            Polynomial(p) => Ok(Polynomial(p.deriv_integ(&wrt)?)),
            Add { left, right } => Ok(Add {
                left: Box::new(left.deriv_integ(&wrt)?),
                right: Box::new(right.deriv_integ(&wrt)?),
            }),
            Sub { left, right } => Ok(Sub {
                left: Box::new(left.deriv_integ(&wrt)?),
                right: Box::new(right.deriv_integ(&wrt)?),
            }),
            Mul { left, right } => Ok(Add {
                left: Box::new(Mul {
                    left: Box::new(left.deriv_integ(&wrt)?),
                    right: right.clone(),
                }),
                right: Box::new(Mul {
                    left: Box::new(right.deriv_integ(&wrt)?),
                    right: left.clone(),
                }),
            }),
            Div { num, denom } => Err(PolynomialError::Other(
                "Derivative/integral of rational not implemented".to_string(),
            )),
            // Ok(Div{
            //
            //         Sub{
            //             left: Box::new(
            //                 Mul{
            //                     left: Box::new(num.deriv_integ(&wrt)?),
            //                     right: denom.clone(),
            //                 }
            //             ),
            //             right: Box::new(
            //                 Mul{
            //                     left: Box::new(denom.deriv_integ(&wrt)?),
            //                     right: num.clone(),
            //                 }
            //             ),
            //         }
            //     ),
            //     denom: Box::new(
            //         Pow{ expression: denom.clone(), power: 2 }
            //     )
            // }),
            Scale { scale, expression } => Ok(Scale {
                scale: *scale,
                expression: Box::new(expression.deriv_integ(&wrt)?),
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
                Ok(expression.deriv_integ(&wrt.as_slice())?)
            }
            Pow { expression, power } => {
                let scale: T = <T as NumCast>::from(power - 1)
                    .ok_or(PolynomialError::Other("Couldn't convert power".to_string()))?;
                Ok(Mul {
                    left: Box::new(Scale {
                        scale,
                        expression: expression.clone(),
                    }),
                    right: Box::new(expression.deriv_integ(&wrt)?),
                })
            }
        }
    }
    /// Expands all compositions as you go
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
                    denom: num.clone(),
                }),
                (
                    ExpandedExpression::Rational { num, denom },
                    ExpandedExpression::Polynomial(p),
                ) => Ok(ExpandedExpression::Rational {
                    num: num.clone(),
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
            DerivInteg { expression, wrt } => expression.deriv_integ(&wrt)?.expand(),
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
