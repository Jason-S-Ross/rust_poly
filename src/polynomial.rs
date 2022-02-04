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

use ahash::AHashSet;
use bitvec::prelude::*;
use ndarray::parallel::prelude::*;
use ndarray::{
    Array, ArrayD, ArrayView, ArrayViewD, Dim, Dimension, IntoDimension, IxDynImpl, NdIndex,
    ScalarOperand, ShapeBuilder, SliceInfoElem,
};
use num_traits::cast::NumCast;
use num_traits::identities::{One, Zero};
use num_traits::pow::Pow;
use std::ops::{Add, Div, Mul, Sub};

use pyo3::exceptions::PyValueError;
use pyo3::PyErr;
const CHUNK_SIZE: usize = 5000;

#[derive(Debug, Clone)]
pub enum PolynomialError {
    Composition,
    Evaluation,
    Other(String),
}

impl From<PolynomialError> for PyErr {
    fn from(err: PolynomialError) -> Self {
        match err {
            PolynomialError::Composition => {
                PyValueError::new_err("Incompatible dimensions for composition".to_string())
            }
            PolynomialError::Evaluation => {
                PyValueError::new_err("Incompatibile dimensions for evaluation".to_string())
            }
            PolynomialError::Other(s) => PyValueError::new_err(s),
        }
    }
}

impl std::fmt::Display for PolynomialError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            PolynomialError::Composition => {
                write!(f, "Composed polynomials of incompatible dimension")
            }
            PolynomialError::Evaluation => {
                write!(f, "Tried to evaluate argument of incompatible dimension")
            }
            PolynomialError::Other(s) => write!(f, "Something went wrong. Error message: {}", s),
        }
    }
}

/// Permutations of N things taken k at a time, i.e., k-permutations of N
#[inline]
fn perm(k: usize, n: usize) -> usize {
    if n > k {
        return 0;
    }
    if n == 1 {
        return k;
    }
    ((k - n + 1)..(k + 1))
        .reduce(|prod, val| prod * val)
        .unwrap_or(1)
}

#[inline]
pub fn pow<T>(val: T, pow: usize) -> T
where
    T: Pow<u16, Output = T>,
{
    Pow::pow(val, pow as u16)
}

#[derive(Debug, Clone, PartialEq)]
pub struct Polynomial<S> {
    coefficients: ArrayD<S>,
}

impl<S> std::fmt::Display for Polynomial<S>
where
    S: std::fmt::Display + Zero,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        fn get_script(number: usize, sub: bool) -> String {
            let superscripts = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];
            let subscripts = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉'];
            let digit_str = format!("{}", number);
            let digits = digit_str.chars().map(|x| {
                let res: usize = String::from(x).parse().expect("Failed to parse number");
                res
            });
            let mut result = String::new();
            for digit in digits {
                result.push(if sub {
                    subscripts[digit]
                } else {
                    superscripts[digit]
                });
            }
            result
        }
        let mut result = vec![];
        for (multiindex, coef) in self.coefficients.indexed_iter() {
            if coef.is_zero() {
                continue;
            }
            let mut factor = format!("{}", coef);
            for (index, power) in multiindex
                .into_dimension()
                .as_array_view()
                .iter()
                .enumerate()
            {
                if *power == 0 {
                    continue;
                }
                let term = format!(
                    "x{}{}",
                    get_script(index + 1, true),
                    get_script(*power, false)
                );
                factor.push_str(&term);
            }
            result.push(factor);
        }
        let result = if result.is_empty() {
            String::from("0")
        } else {
            result.join(" + ")
        };
        write!(f, "{}", result)
    }
}

impl<S> Polynomial<S>
where
    S: Add<Output = S>
        + Mul<Output = S>
        + Sub<Output = S>
        + ScalarOperand
        + Copy
        + Zero
        + One
        + Pow<u16, Output = S>
        + Send
        + Sync,
{
    #[inline]
    pub fn new(coefficients: ArrayD<S>) -> Self {
        Polynomial { coefficients }
    }
    #[inline]
    pub fn shape(&self) -> Vec<usize> {
        self.coefficients.shape().to_vec()
    }
    #[inline]
    pub fn dimension(&self) -> usize {
        self.coefficients.shape().len()
    }
    #[inline]
    pub fn dofs(&self) -> AHashSet<usize> {
        self.coefficients
            .shape()
            .iter()
            .enumerate()
            .filter_map(|(i, v)| if *v > 1 { Some(i) } else { None })
            .collect()
    }
    #[inline]
    pub fn drop_params(&self, to_drop: &[usize]) -> Result<Self, PolynomialError> {
        for i in to_drop.iter() {
            if *self
                .shape()
                .get(*i)
                .ok_or_else(|| PolynomialError::Other("Could not get axis to drop".to_string()))?
                > 1
            {
                return Err(PolynomialError::Other(
                    "Attempted to drop a non-empty axis".to_string(),
                ));
            }
        }
        let new_shape = self
            .shape()
            .iter()
            .enumerate()
            .filter_map(|(i, x)| if to_drop.contains(&i) { None } else { Some(*x) })
            .collect::<Vec<_>>();
        Ok(Polynomial::new(
            self.coefficients
                .clone()
                .into_shape(new_shape.as_slice())
                .map_err(|_| {
                    PolynomialError::Other("Invalid shape in drop operation".to_string())
                })?,
        ))
    }
    #[inline]
    pub fn scale(&self, scalar: S) -> Result<Self, PolynomialError> {
        Ok(Self::new(self.coefficients.clone() * scalar))
    }
    #[inline]
    pub fn to_constant(&self) -> Result<S, PolynomialError> {
        if self.coefficients.indexed_iter().any(|(index, coef)| {
            index
                .into_dimension()
                .as_array_view()
                .iter()
                .any(|i| *i != 0)
                && !coef.is_zero()
        }) {
            return Err(PolynomialError::Other(
                "Polynomial has non-constant terms".to_string(),
            ));
        }
        Ok(*self
            .coefficients
            .iter()
            .next()
            .expect("Unwrapping a next on a single-valued iterator shouldn't pannic"))
    }
    /// Evaluates coefficients generically
    #[inline]
    fn polyval<D>(coefs: &ArrayView<S, D>, values: &[S]) -> Result<S, PolynomialError>
    where
        D: Dimension,
        Dim<IxDynImpl>: NdIndex<D>,
    {
        let res = coefs
            .indexed_iter()
            .map(|(index, coef)| {
                let term = index
                    .into_dimension()
                    .as_array_view()
                    .iter()
                    .enumerate()
                    .map(|(dim, power)| {
                        let val = *values.get(dim)?;
                        let prod = pow(val, *power);
                        Some(prod)
                    })
                    .reduce(|acc, val| match (acc, val) {
                        (None, None) => None,
                        (Some(x), None) => Some(x),
                        (None, Some(x)) => Some(x),
                        (Some(x), Some(y)) => Some(x * y),
                    })?;
                match term {
                    Some(t) => Some(*coef * t),
                    None => Some(*coef),
                }
            })
            .reduce(|acc, term| match (acc, term) {
                (None, None) => None,
                (Some(acc), None) => Some(acc),
                (None, Some(term)) => Some(term),
                (Some(acc), Some(term)) => Some(acc + term),
            });
        res.ok_or_else(|| PolynomialError::Other("Inner none on polyval".to_string()))?
            .ok_or_else(|| PolynomialError::Other("Outer none on polyval".to_string()))
    }
    #[inline]
    fn polyval_vector(
        coefs: &ArrayViewD<S>,
        values: &ArrayViewD<S>,
    ) -> Result<ArrayD<S>, PolynomialError> {
        let shape = values.shape()[..values.shape().len() - 1].into_shape();
        let size = values.shape().len();
        let res = coefs
            .indexed_iter()
            .filter_map(|(index, coef)| {
                if coef.is_zero() {
                    None
                } else {
                    let prod = index
                        .into_dimension()
                        .as_array_view()
                        .iter()
                        .enumerate()
                        .filter_map(|(dim, power)| {
                            use SliceInfoElem::*;
                            let slice = std::iter::repeat(Slice {
                                start: 0,
                                end: None,
                                step: 1,
                            })
                            .take(size - 1)
                            .chain([Index(dim as isize)])
                            .collect::<Vec<_>>();
                            if *power == 0 {
                                None
                            } else if *power == 1 {
                                Some(values.slice(slice.as_slice()).to_owned())
                            } else {
                                let vals = values.slice(slice.as_slice());
                                let res = vals.mapv(|x| pow(x, *power));
                                Some(res)
                            }
                        })
                        .reduce(|acc, val| acc * val);
                    match prod {
                        Some(p) => Some(p * *coef),
                        None => Some(ArrayD::from_elem(shape.clone(), *coef)),
                    }
                }
            })
            .reduce(|acc, term| acc + term)
            .unwrap_or_else(|| ArrayD::zeros(shape));
        Ok(res)
    }

    /// Composes one polynomial with another
    #[inline]
    pub fn compose_scalar(&self, values: &[&Self]) -> Result<Self, PolynomialError> {
        let dimension = values
            .iter()
            .next()
            .ok_or(PolynomialError::Composition)?
            .shape()
            .len();
        if values.iter().any(|x| x.shape().len() != dimension) {
            return Err(PolynomialError::Composition);
        }
        let res = self
            .coefficients
            .indexed_iter()
            .map(|(index, coef)| {
                let term = index
                    .into_dimension()
                    .as_array_view()
                    .iter()
                    .enumerate()
                    .map(|(dim, power)| {
                        let val = values.get(dim)?;
                        val.pow(*power).ok()
                    })
                    .reduce(|acc, val| match (acc, val) {
                        (None, None) => None,
                        (Some(x), None) => Some(x),
                        (None, Some(x)) => Some(x),
                        (Some(x), Some(y)) => x.mul(&y).ok(),
                    })?;
                match term {
                    Some(t) => t.scale(*coef).ok(),
                    None => Self::one(dimension).scale(*coef).ok(),
                }
            })
            .reduce(|acc, term| match (acc, term) {
                (None, None) => None,
                (Some(acc), None) => Some(acc),
                (None, Some(term)) => Some(term),
                (Some(acc), Some(term)) => acc.add(&term).ok(),
            });
        res.ok_or_else(|| PolynomialError::Other("Inner none on compose scalar".to_string()))?
            .ok_or_else(|| PolynomialError::Other("Outer none on compose scalar".to_string()))
    }
    /// Composes one polynomial with another
    #[inline]
    pub fn compose<D>(&self, values: &ArrayViewD<&Self>) -> Result<ArrayD<Self>, PolynomialError> {
        let dimension = values
            .iter()
            .next()
            .ok_or(PolynomialError::Composition)?
            .coefficients
            .shape()
            .len();
        if values
            .iter()
            .any(|x| x.coefficients.shape().len() != dimension)
        {
            return Err(PolynomialError::Composition);
        }
        let val_shape: Vec<_> = values.shape().to_vec();
        let outputshape = Self::get_shape_eval(&self.shape(), &val_shape)?;
        let mut output = Array::<Self, _>::from_elem(outputshape.as_slice(), Self::zero(dimension));
        for (result_index, result_val) in output.indexed_iter_mut() {
            let slice_positions: Vec<_> = result_index
                .as_array_view()
                .iter()
                .map(|x| SliceInfoElem::Index(*x as isize))
                .chain([SliceInfoElem::Slice {
                    start: 0,
                    end: None,
                    step: 1,
                }])
                .collect();
            let some_vals = values.slice(slice_positions.as_slice());
            let slc = some_vals.as_slice().ok_or(PolynomialError::Composition)?;
            let res = self.compose_scalar(slc);
            *result_val = res?;
        }
        Ok(output)
    }
    /// Evaluate a polynomial on an array of scalars.
    #[inline]
    pub fn eval_scalar(&self, values: &[S]) -> Result<S, PolynomialError> {
        Polynomial::<S>::polyval(&self.coefficients.view(), values)
    }
    #[inline]
    pub fn eval<'a>(&'a self, values: &ArrayViewD<S>) -> Result<ArrayD<S>, PolynomialError> {
        let result_shape = IxDyn(
            values
                .shape()
                .iter()
                .take(values.shape().len() - 1)
                .cloned()
                .collect::<Vec<_>>()
                .as_slice(),
        );
        if values.shape().len() == 1 {
            return Self::polyval_vector(&self.coefficients.view(), values);
        }
        let new_len = values
            .shape()
            .iter()
            .take(values.shape().len() - 1)
            .cloned()
            .reduce(|acc, x| acc * x)
            .ok_or_else(|| PolynomialError::Other("Failed to get new len".to_string()))?;

        let new_shape = [
            new_len,
            *values
                .shape()
                .iter()
                .last()
                .ok_or_else(|| PolynomialError::Other("Failed to get new shape".to_string()))?,
        ];
        use ndarray::{concatenate, Axis, IxDyn};
        let mut out = vec![];
        let reshaped_vals = values.to_shape(IxDyn(&new_shape)).map_err(|_| {
            PolynomialError::Other("Failed to reshape values before eval".to_string())
        })?;
        reshaped_vals
            .axis_chunks_iter(Axis(0), CHUNK_SIZE)
            .into_par_iter()
            .map(|chunk| {
                Self::polyval_vector(&self.coefficients.view(), &chunk)
                    .expect("Tried to eval within a closure")
            })
            .collect_into_vec(&mut out);
        let views = out.iter().map(|x| x.view()).collect::<Vec<_>>();
        Ok(concatenate(Axis(0), views.as_slice())
            .map_err(|_| {
                PolynomialError::Other("Failed to concatenate values after eval".to_string())
            })?
            .to_shape(result_shape)
            .map_err(|_| {
                PolynomialError::Other("Failed to reshape values after evaluation".to_string())
            })?
            .into_owned())
    }
    /// Evaluates a vector on the provided buffer
    #[inline]
    pub fn eval_no_alloc<'a>(
        &self,
        values: &'a ArrayViewD<S>,
        output: &'a mut ArrayD<S>,
    ) -> Result<(), PolynomialError> {
        for (result_index, result_val) in output.indexed_iter_mut() {
            let slice_positions: Vec<_> = result_index
                .as_array_view()
                .iter()
                .map(|x| SliceInfoElem::Index(*x as isize))
                .chain([SliceInfoElem::Slice {
                    start: 0,
                    end: None,
                    step: 1,
                }])
                .collect();
            let some_vals = values.slice(slice_positions.as_slice());
            let res = self.eval_scalar(some_vals.as_slice().unwrap());
            *result_val = res?;
        }
        Ok(())
    }
    /// Gets the shape of the result of partial evaluation
    #[inline]
    pub fn get_shape_partial(
        coef_shape: &[usize],
        target_indices: &[usize],
    ) -> Result<Vec<usize>, PolynomialError> {
        if target_indices.iter().any(|x| x >= &coef_shape.len()) {
            return Err(PolynomialError::Evaluation);
        }
        let mut unique: AHashSet<usize> = AHashSet::with_capacity(target_indices.len());
        if !target_indices.iter().all(|x| unique.insert(*x)) {
            return Err(PolynomialError::Evaluation);
        }
        Ok(coef_shape
            .iter()
            .enumerate()
            .map(|(i, x)| if unique.contains(&i) { 1 } else { *x })
            .collect())
    }
    /// Gets the shape of the result of evaluating a polynomial
    #[inline]
    pub fn get_shape_eval(
        coef_shape: &[usize],
        val_shape: &[usize],
    ) -> Result<Vec<usize>, PolynomialError> {
        if val_shape[val_shape.len() - 1] != coef_shape.len() {
            return Err(PolynomialError::Evaluation);
        }
        Ok(val_shape[..val_shape.len() - 1].to_vec())
    }

    /// Gets the shape of the result of a multiplication operation
    #[inline]
    pub fn get_shape_mul(
        left_shape: &[usize],
        right_shape: &[usize],
    ) -> Result<Vec<usize>, PolynomialError> {
        if left_shape.len() != right_shape.len() {
            return Err(PolynomialError::Composition);
        }
        let new_shape: Vec<_> = left_shape
            .iter()
            .zip(right_shape.iter())
            .map(|(left, right)| left + right - 1)
            .collect();
        Ok(new_shape)
    }
    #[inline]
    pub fn get_shape_pow(shape: &[usize], exponent: usize) -> Result<Vec<usize>, PolynomialError> {
        let one = || vec![1; shape.len()];
        if exponent == 0 {
            return Ok(one());
        }
        if exponent == 1 {
            return Ok(shape.to_vec());
        }
        let mut result = one();
        let mut last_pow = shape.to_vec();
        let powarray = [exponent];
        let bit_view = powarray.view_bits::<Lsb0>();
        match bit_view.last_one() {
            Some(bitfinal) => {
                for (bit, _) in bit_view.iter().zip(0..bitfinal + 1) {
                    let b = bit.to_owned();
                    if *b {
                        result =
                            Polynomial::<S>::get_shape_mul(result.as_slice(), last_pow.as_slice())?;
                    }
                    last_pow =
                        Polynomial::<S>::get_shape_mul(last_pow.as_slice(), last_pow.as_slice())?;
                }
                Ok(result)
            }
            None => Ok(shape.to_owned()),
        }
    }
    /// Gets the shape of the result of an addition operation
    #[inline]
    pub fn get_shape_add(
        left_shape: &[usize],
        right_shape: &[usize],
    ) -> Result<Vec<usize>, PolynomialError> {
        if left_shape.len() != right_shape.len() {
            return Err(PolynomialError::Composition);
        }
        let new_shape: Vec<_> = left_shape
            .iter()
            .zip(right_shape.iter())
            .map(|(left, right)| if left > right { *left } else { *right })
            .collect();
        Ok(new_shape)
    }
    /// Gets the shape of the result of an integral operation
    #[inline]
    pub fn get_shape_integ(shape: &[usize], arg: &[usize]) -> Result<Vec<usize>, PolynomialError> {
        if shape.len() != arg.len() {
            return Err(PolynomialError::Composition);
        }
        let new_shape: Vec<_> = shape
            .iter()
            .zip(arg.iter())
            .map(|(left, right)| *left + *right)
            .collect();
        Ok(new_shape)
    }
    /// Gets the shape of the result of a generalized integral/derivative
    /// Negative values in arg indicate integration
    #[inline]
    pub fn get_shape_deriv_integ(
        shape: &[usize],
        arg: &[isize],
    ) -> Result<Vec<usize>, PolynomialError> {
        if shape.len() != arg.len() {
            return Err(PolynomialError::Composition);
        }
        let new_shape: Vec<_> = shape
            .iter()
            .zip(arg.iter())
            .map(|(left, right)| {
                let v = *left as isize - *right;
                if v < 0 {
                    0
                } else {
                    v as usize
                }
            })
            .collect();
        Ok(new_shape)
    }
    /// Gets the shape of the result of a derivative operation
    #[inline]
    pub fn get_shape_deriv(shape: &[usize], arg: &[usize]) -> Result<Vec<usize>, PolynomialError> {
        if shape.len() != arg.len() {
            return Err(PolynomialError::Composition);
        }
        let new_shape: Vec<_> = shape
            .iter()
            .zip(arg.iter())
            .map(|(left, right)| left.saturating_sub(*right))
            .collect();
        Ok(new_shape)
    }
    /// Panics on out-of-bounds
    #[inline]
    pub fn deriv(&self, arg: &[usize]) -> Result<Self, PolynomialError>
    where
        S: NumCast,
    {
        let arg: Vec<_> = arg.iter().map(|x| *x as isize).collect();
        self.deriv_integ(arg.as_slice())
    }
    /// Panics on out-of-bounds
    #[inline]
    pub fn deriv_integ_no_alloc(
        &self,
        arg: &[isize],
        output: &mut ArrayD<S>,
    ) -> Result<(), PolynomialError>
    where
        S: NumCast,
    {
        let index_size = self.shape().len();
        for (ref index, _val) in self.coefficients.indexed_iter() {
            let factor = index
                .clone()
                .into_dimension()
                .as_array_view()
                .iter()
                .zip(arg.iter())
                .map(|(index, arg)| match *arg {
                    1.. => perm(*index, *arg as usize) as f64,
                    0 => 1.0,
                    _ => 1.0 / (perm(*index + (*arg).abs() as usize, (*arg).abs() as usize) as f64),
                })
                .reduce(|prod, val| prod * val)
                .unwrap_or(1.0); // TODO Check if 1 is the right case here
            let new_index: Vec<_> = index
                .as_array_view()
                .iter()
                .zip(arg.iter())
                .map_while(|(left, right)| {
                    // Use map_while and checked_sub so that if the index is negative,
                    // we ignore it
                    let val = *left as isize - *right;
                    if val < 0 {
                        None
                    } else {
                        Some(val as usize)
                    }
                })
                .collect();
            if new_index.len() != index_size {
                // True if any of the indices underflowed
                continue;
            }
            output[new_index.as_slice()] = self.coefficients[index]
                * <S as NumCast>::from(factor).ok_or_else(|| {
                    PolynomialError::Other("Couldn't cast factor as scalar".to_string())
                })?;
        }
        Ok(())
    }
    #[inline]
    pub fn deriv_integ(&self, arg: &[isize]) -> Result<Self, PolynomialError>
    where
        S: NumCast,
    {
        let new_shape = Self::get_shape_deriv_integ(&self.shape(), arg)?;
        let mut result = ArrayD::<S>::zeros(new_shape);
        self.deriv_integ_no_alloc(arg, &mut result).map_err(|_| {
            PolynomialError::Other(
                "Something went wrong trying to integrate after allocation".to_string(),
            )
        })?;
        Ok(Self::new(result))
    }
    /// Partial evaluation on a subset of axes.
    #[inline]
    pub fn partial(
        &self,
        target_indices: &[usize],
        target_vals: &[S],
    ) -> Result<Self, PolynomialError> {
        let shape = Polynomial::<S>::get_shape_partial(&self.shape(), target_indices)?;
        if target_indices.len() != target_vals.len() {
            return Err(PolynomialError::Evaluation);
        }
        let target_vals = {
            let mut vals: Vec<_> = target_vals.iter().enumerate().collect();
            vals.sort_by_key(|(i, _)| target_indices[*i]);
            vals.iter()
                .map(|(_, x)| x)
                .cloned()
                .cloned()
                .collect::<Vec<_>>()
        };
        let mut coefs = ArrayD::<S>::zeros(shape);
        let indices: AHashSet<_> = target_indices.iter().collect();
        for (to_index, to_val) in coefs.indexed_iter_mut() {
            let slice: Vec<_> = to_index
                .as_array_view()
                .iter()
                .enumerate()
                .map(|(i, index)| {
                    if indices.contains(&i) {
                        SliceInfoElem::Slice {
                            start: 0,
                            end: None,
                            step: 1,
                        }
                    } else {
                        SliceInfoElem::Index(*index as isize)
                    }
                })
                .collect();
            let subcoefs = self.coefficients.slice(slice.as_slice());
            *to_val = Polynomial::<S>::polyval(&subcoefs, target_vals.as_slice())?;
        }
        Ok(Polynomial::new(coefs))
    }
    #[inline]
    pub fn integ(&self, arg: &[usize]) -> Result<Self, PolynomialError>
    where
        S: NumCast + std::fmt::Debug + Div<Output = S>,
    {
        let arg: Vec<_> = arg.iter().map(|x| -(*x as isize)).collect();
        self.deriv_integ(arg.as_slice())
    }

    /// Adds two polynomials without allocating space for either. Panics on
    /// out-of-bounds
    #[inline]
    pub fn add_no_alloc(&self, other: &Self, output: &mut ArrayD<S>) {
        for (ref index, val) in self.coefficients.indexed_iter() {
            output[index] = output[index] + *val;
        }
        for (ref index, val) in other.coefficients.indexed_iter() {
            output[index] = output[index] + *val;
        }
    }
    #[inline]
    pub fn sub_no_alloc(&self, other: &Self, output: &mut ArrayD<S>) {
        for (ref index, val) in self.coefficients.indexed_iter() {
            output[index] = output[index] + *val;
        }
        for (ref index, val) in other.coefficients.indexed_iter() {
            output[index] = output[index] - *val;
        }
    }
    /// Multiplies two polynomials without allocating space for either. Panics
    /// on out-of-bounds
    #[inline]
    pub fn mul_no_alloc(&self, other: &Self, output: &mut ArrayD<S>) {
        for (ref left_index, left_val) in self.coefficients.indexed_iter() {
            for (ref right_index, right_val) in other.coefficients.indexed_iter() {
                let output_index: Vec<_> = left_index
                    .clone()
                    .into_dimension()
                    .as_array_view()
                    .iter()
                    .zip(right_index.clone().into_dimension().as_array_view().iter())
                    .map(|(left, right)| *left + *right)
                    .collect();
                output[output_index.as_slice()] =
                    output[output_index.as_slice()] + *left_val * *right_val;
            }
        }
    }
    #[inline]
    pub fn pow(&self, exp: usize) -> Result<Self, PolynomialError> {
        let dimension = self.shape().len();
        if exp == 0 {
            return Ok(Self::one(dimension));
        }
        if exp == 1 {
            return Ok(self.clone());
        }
        let mut result = Self::one(dimension);
        let mut last_pow = self.clone();
        let powarray = [exp];
        let bit_view = powarray.view_bits::<Lsb0>();
        match bit_view.last_one() {
            Some(bitfinal) => {
                for (bit, _) in bit_view.iter().zip(0..bitfinal + 1) {
                    let b = bit.to_owned();
                    if *b {
                        result = result.mul(&last_pow)?;
                    }
                    last_pow = last_pow.mul(&last_pow)?;
                }
                Ok(result)
            }
            None => Err(PolynomialError::Other(
                "Somehow there was no last bit in pow".to_string(),
            )),
        }
    }
    #[inline]
    pub fn one(dimension: usize) -> Self {
        let coefficients = ArrayD::<S>::ones(vec![1; dimension].as_slice());
        Self { coefficients }
    }
    #[inline]
    pub fn zero(dimension: usize) -> Self {
        let coefficients = ArrayD::<S>::zeros(vec![1; dimension].as_slice());
        Self { coefficients }
    }
    #[inline]
    pub fn add(&self, rhs: &Self) -> Result<Self, PolynomialError> {
        let new_shape = Polynomial::<S>::get_shape_add(&self.shape(), &rhs.shape())?;
        let mut result = ArrayD::<S>::zeros(new_shape);
        self.add_no_alloc(rhs, &mut result);
        Ok(Polynomial::new(result))
    }
    #[inline]
    pub fn sub(&self, rhs: &Self) -> Result<Self, PolynomialError> {
        let new_shape = Polynomial::<S>::get_shape_add(&self.shape(), &rhs.shape())?;
        let mut result = ArrayD::<S>::zeros(new_shape);
        self.sub_no_alloc(rhs, &mut result);
        Ok(Polynomial::new(result))
    }
    #[inline]
    pub fn mul(&self, rhs: &Self) -> Result<Self, PolynomialError> {
        let new_shape = Polynomial::<S>::get_shape_mul(&self.shape(), &rhs.shape())?;
        let mut result = ArrayD::<S>::zeros(new_shape);
        self.mul_no_alloc(rhs, &mut result);
        Ok(Polynomial::new(result))
    }
}

#[cfg(test)]

mod test {
    use super::*;

    use ndarray::{array, Axis, IxDyn};
    #[test]
    fn it_works() {
        assert_eq!(1, 1);
    }
    #[test]
    fn make_eval_poly_1d() {
        let mut array = ArrayD::<f64>::zeros(IxDyn(&[3]));
        array[[0]] = 1.0;
        array[[1]] = 2.0;
        array[[2]] = 3.0;
        let foo = Polynomial::new(array);
        let mut bar = Array::<f64, _>::zeros(IxDyn(&[1]));
        bar[[0]] = 2.0;
        println!("foo.eval(&bar.view()): {:#?}", foo.eval(&bar.view()));
        assert_eq!(foo.eval(&bar.view()).unwrap()[[]], 17.0);
    }
    #[test]
    fn square_poly_1d() {
        let mut array = ArrayD::zeros(IxDyn(&[3]));
        array[[0]] = 1.0;
        array[[1]] = 2.0;
        array[[2]] = 3.0;
        let foo = Polynomial::<f64>::new(array);
        let actual = foo.clone().mul(&foo);
        let expected = {
            let mut coefs = ArrayD::zeros(IxDyn(&[5]));
            coefs[[0]] = 1.0;
            coefs[[1]] = 4.0;
            coefs[[2]] = 10.0;
            coefs[[3]] = 12.0;
            coefs[[4]] = 9.0;
            Polynomial::<f64>::new(coefs)
        };
        assert_eq!(expected, actual.unwrap());
    }
    #[test]
    fn mul_poly_1d() {
        let foo = {
            let mut array = ArrayD::zeros(IxDyn(&[3]));
            array[[0]] = 1.0;
            array[[1]] = 2.0;
            array[[2]] = 3.0;
            Polynomial::<f64>::new(array)
        };
        let bar = {
            let mut array = ArrayD::zeros(IxDyn(&[3]));
            array[[0]] = 2.0;
            array[[1]] = 3.0;
            array[[2]] = 4.0;
            Polynomial::<f64>::new(array)
        };
        let actual = foo.mul(&bar);
        let expected = {
            let mut coefs = ArrayD::zeros(IxDyn(&[5]));
            coefs[[0]] = 2.0;
            coefs[[1]] = 7.0;
            coefs[[2]] = 16.0;
            coefs[[3]] = 17.0;
            coefs[[4]] = 12.0;
            Polynomial::<f64>::new(coefs)
        };
        assert_eq!(expected, actual.unwrap());
    }
    #[test]
    fn add_poly_1d() {
        let foo = {
            let mut array = ArrayD::zeros(IxDyn(&[3]));
            array[[0]] = 1.0;
            array[[1]] = 2.0;
            array[[2]] = 3.0;
            Polynomial::<f64>::new(array)
        };
        let bar = {
            let mut array = ArrayD::zeros(IxDyn(&[3]));
            array[[0]] = 2.0;
            array[[1]] = 3.0;
            array[[2]] = 4.0;
            Polynomial::<f64>::new(array)
        };
        let actual = foo.add(&bar);
        let expected = {
            let mut coefs = ArrayD::zeros(IxDyn(&[3]));
            coefs[[0]] = 3.0;
            coefs[[1]] = 5.0;
            coefs[[2]] = 7.0;
            Polynomial::<f64>::new(coefs)
        };
        assert_eq!(expected, actual.unwrap());
    }
    #[test]
    fn mul_poly_commute_1d() {
        let foo = {
            let mut array = ArrayD::zeros(IxDyn(&[3]));
            array[[0]] = 1.0;
            array[[1]] = 2.0;
            array[[2]] = 3.0;
            Polynomial::<f64>::new(array)
        };
        let bar = {
            let mut array = ArrayD::zeros(IxDyn(&[3]));
            array[[0]] = 2.0;
            array[[1]] = 3.0;
            array[[2]] = 4.0;
            Polynomial::<f64>::new(array)
        };

        assert_eq!(foo.mul(&bar).unwrap(), bar.mul(&foo).unwrap());
    }
    #[test]
    fn make_eval_poly_2d() {
        let mut array = ArrayD::zeros(IxDyn(&[2, 2]));
        array[[0, 0]] = 1.0;
        array[[0, 1]] = 2.0;
        array[[1, 0]] = 3.0;
        array[[1, 1]] = 4.0;
        let foo = Polynomial::new(array);
        let mut bar = ArrayD::zeros(IxDyn(&[2]));
        bar[[0]] = 3.;
        bar[[1]] = 7.;
        assert_eq!(foo.eval(&bar.view()).unwrap()[[]], 108.0);
    }
    #[test]
    fn make_eval_poly_2d_vector() {
        let mut array = ArrayD::zeros(IxDyn(&[2, 2]));
        array[[0, 0]] = 1.0;
        array[[0, 1]] = 2.0;
        array[[1, 0]] = 3.0;
        array[[1, 1]] = 4.0;
        let foo = Polynomial::new(array);
        let mut bar = ArrayD::zeros(IxDyn(&[2, 2]));
        bar[[0, 0]] = 3.;
        bar[[0, 1]] = 7.;
        bar[[1, 0]] = 3.;
        bar[[1, 1]] = 7.;
        assert_eq!(foo.eval(&bar.view()).unwrap()[[0]], 108.0);
        assert_eq!(foo.eval(&bar.view()).unwrap()[[1]], 108.0);
    }
    #[test]
    fn make_eval_scalar_2d() {
        let mut array = ArrayD::zeros(IxDyn(&[2, 2]));
        array[[0, 0]] = 1.0;
        array[[0, 1]] = 2.0;
        array[[1, 0]] = 3.0;
        array[[1, 1]] = 4.0;
        let foo = Polynomial::new(array);
        assert_eq!(foo.eval_scalar(&[3., 7.]).unwrap(), 108.0);
    }
    #[test]
    fn eval_big_array() {
        let values = (0..2000000).map(|x| x as f64).collect::<Vec<_>>();
        let values = ArrayD::from_shape_vec(IxDyn(&[1000000, 2]), values).unwrap();
        let mut array = ArrayD::zeros(IxDyn(&[2, 2]));
        array[[0, 0]] = 1.0;
        array[[0, 1]] = 2.0;
        array[[1, 0]] = 3.0;
        array[[1, 1]] = 4.0;
        let foo = Polynomial::<f64>::new(array);
        foo.eval(&values.view())
            .expect("This shouldn't be an error!");
        //assert_eq!(foo.eval_scalar(&[3., 7.]).unwrap(), 108.0);
    }
    #[test]
    fn square_poly_2d() {
        let term = {
            let mut coefs = ArrayD::zeros(IxDyn(&[2, 2]));
            coefs[[0, 0]] = 1.0;
            coefs[[0, 1]] = 2.0;
            coefs[[1, 0]] = 3.0;
            coefs[[1, 1]] = 4.0;
            Polynomial::<f64>::new(coefs)
        };
        let actual = (term.mul(&term)).unwrap();
        let expected = {
            let mut coefs = ArrayD::zeros(IxDyn(&[3, 3]));
            coefs[[0, 0]] = 1.0;
            coefs[[0, 1]] = 4.0;
            coefs[[1, 0]] = 6.0;
            coefs[[2, 0]] = 9.0;
            coefs[[1, 1]] = 20.0;
            coefs[[2, 1]] = 24.0;
            coefs[[0, 2]] = 4.0;
            coefs[[1, 2]] = 16.0;
            coefs[[2, 2]] = 16.0;
            Polynomial::<f64>::new(coefs)
        };
        assert_eq!(expected, actual);
    }
    #[test]
    fn partial_poly_2d() {
        let term = {
            let mut coefs = ArrayD::zeros(IxDyn(&[2, 2]));
            coefs[[0, 0]] = 1.0;
            coefs[[0, 1]] = 2.0;
            coefs[[1, 0]] = 3.0;
            coefs[[1, 1]] = 4.0;
            Polynomial::<f64>::new(coefs)
        };
        let actual = term.partial(&[0], &[1.0]).unwrap();
        let expected = {
            let mut coefs = ArrayD::zeros(IxDyn(&[1, 2]));
            coefs[[0, 0]] = 4.0;
            coefs[[0, 1]] = 6.0;
            Polynomial::<f64>::new(coefs)
        };
        assert_eq!(expected, actual);
    }
    #[test]
    fn partial_poly_2d_2() {
        let term = {
            let mut coefs = ArrayD::zeros(IxDyn(&[2, 2]));
            coefs[[0, 0]] = 1.0;
            coefs[[0, 1]] = 2.0;
            coefs[[1, 0]] = 3.0;
            coefs[[1, 1]] = 4.0;
            Polynomial::<f64>::new(coefs)
        };
        let actual = term.partial(&[1], &[1.0]).unwrap();
        let expected = {
            let mut coefs = ArrayD::zeros(IxDyn(&[2, 1]));
            coefs[[0, 0]] = 3.0;
            coefs[[1, 0]] = 7.0;
            Polynomial::<f64>::new(coefs)
        };
        assert_eq!(expected, actual);
    }
    #[test]
    fn partial_poly_2d_3() {
        let term = {
            let mut coefs = ArrayD::zeros(IxDyn(&[2, 2]));
            coefs[[0, 0]] = 1.0;
            coefs[[0, 1]] = 2.0;
            coefs[[1, 0]] = 3.0;
            coefs[[1, 1]] = 4.0;
            Polynomial::<f64>::new(coefs)
        };
        let actual = term.partial(&[0, 1], &[1.0, 3.0]).unwrap();
        let expected = {
            let mut coefs = ArrayD::zeros(IxDyn(&[1, 1]));
            coefs[[0, 0]] = 22.0;
            Polynomial::<f64>::new(coefs)
        };
        assert_eq!(expected, actual);
    }
    #[test]
    fn partial_poly_2d_4() {
        let term = {
            let mut coefs = ArrayD::zeros(IxDyn(&[2, 2]));
            coefs[[0, 0]] = 1.0;
            coefs[[0, 1]] = 2.0;
            coefs[[1, 0]] = 3.0;
            coefs[[1, 1]] = 4.0;
            Polynomial::<f64>::new(coefs)
        };
        let actual = term.partial(&[1, 0], &[3.0, 1.0]).unwrap();
        let expected = {
            let mut coefs = ArrayD::zeros(IxDyn(&[1, 1]));
            coefs[[0, 0]] = 22.0;
            Polynomial::<f64>::new(coefs)
        };
        assert_eq!(expected, actual);
    }

    #[test]
    fn make_eval_poly_2d_2() {
        let mut array = ArrayD::zeros(IxDyn(&[2, 2]));
        array[[0, 0]] = 1.0;
        array[[0, 1]] = 2.0;
        array[[1, 0]] = 3.0;
        array[[1, 1]] = 4.0;
        let foo = Polynomial::new(array);
        let mut bar = Array::zeros(IxDyn(&[2, 2]));
        bar[[0, 0]] = 3.;
        bar[[1, 0]] = 7.;
        bar[[0, 1]] = 4.;
        bar[[1, 1]] = 7.;
        assert_eq!(foo.eval(&bar.view()).unwrap()[[0]], 66.0);
        assert_eq!(foo.eval(&bar.view()).unwrap()[[1]], 232.0);
    }
    #[ignore]
    #[test]
    fn test_reduce() {
        let foo = vec![Some(3)];
        let _res = foo.iter().map(|v| *v).reduce(|acc, val| match (acc, val) {
            (None, None) => None,
            (Some(x), None) => Some(x),
            (None, Some(x)) => Some(x),
            (Some(x), Some(y)) => Some(x + y),
        });
    }
    #[test]
    fn test_array0() {
        use ndarray::arr0;
        let foo = arr0(5.0);
        let _dim = foo.raw_dim();
    }
    #[test]
    fn test_array2() {
        let bar = array![[1.0, 2.0], [3.0, 4.0]];
        let dim = bar.raw_dim();
        let (_, bazdim) = dim
            .as_array_view()
            .split_at(Axis(0), dim.as_array_view().len() - 1);
        let _baz = Array::<f64, _>::zeros(bazdim.to_slice().unwrap());
    }
    #[test]
    fn test_perms() {
        assert_eq!(perm(4, 4), 24);
        assert_eq!(perm(4, 3), 24);
        assert_eq!(perm(4, 2), 12);
        assert_eq!(perm(4, 1), 4);
        assert_eq!(perm(4, 0), 1);
        assert_eq!(perm(4, 5), 0);
    }
    #[test]
    fn test_deriv_1() {
        let term = {
            let mut coefs = ArrayD::zeros(IxDyn(&[2, 3]));
            coefs[[0, 0]] = 1.0;
            coefs[[0, 1]] = 2.0;
            coefs[[0, 2]] = 3.0;
            coefs[[1, 0]] = 4.0;
            coefs[[1, 1]] = 5.0;
            coefs[[1, 2]] = 6.0;
            Polynomial::<f64>::new(coefs)
        };
        let actual = term.deriv(&[0, 1]);
        let expected = {
            let mut coefs = ArrayD::zeros(IxDyn(&[2, 2]));
            coefs[[0, 0]] = 2.0;
            coefs[[0, 1]] = 6.0;
            coefs[[1, 0]] = 5.0;
            coefs[[1, 1]] = 12.0;
            Polynomial::<f64>::new(coefs)
        };
        assert_eq!(expected, actual.unwrap());
    }
    #[test]
    fn test_deriv_2() {
        let term = {
            let mut coefs = ArrayD::zeros(IxDyn(&[2, 3]));
            coefs[[0, 0]] = 1.0;
            coefs[[0, 1]] = 2.0;
            coefs[[0, 2]] = 3.0;
            coefs[[1, 0]] = 4.0;
            coefs[[1, 1]] = 5.0;
            coefs[[1, 2]] = 6.0;
            Polynomial::<f64>::new(coefs)
        };
        let actual = term.deriv(&[1, 1]);
        let expected = {
            let mut coefs = ArrayD::zeros(IxDyn(&[1, 2]));
            coefs[[0, 0]] = 5.0;
            coefs[[0, 1]] = 12.0;
            Polynomial::<f64>::new(coefs)
        };
        assert_eq!(expected, actual.unwrap());
    }
    #[test]
    fn test_integ_2() {
        let term = {
            let mut coefs = ArrayD::zeros(IxDyn(&[2, 3]));
            coefs[[0, 0]] = 1.0;
            coefs[[0, 1]] = 2.0;
            coefs[[0, 2]] = 3.0;
            coefs[[1, 0]] = 4.0;
            coefs[[1, 1]] = 5.0;
            coefs[[1, 2]] = 6.0;
            Polynomial::<f64>::new(coefs)
        };
        let actual = term.integ(&[0, 1]);
        let expected = {
            let mut coefs = ArrayD::zeros(IxDyn(&[2, 4]));
            coefs[[0, 0]] = 0.0;
            coefs[[0, 1]] = 1.0;
            coefs[[0, 2]] = 1.0;
            coefs[[0, 3]] = 1.0;
            coefs[[1, 0]] = 0.0;
            coefs[[1, 1]] = 4.0;
            coefs[[1, 2]] = 2.5;
            coefs[[1, 3]] = 2.0;
            Polynomial::<f64>::new(coefs)
        };
        assert_eq!(expected, actual.unwrap());
    }
    #[test]
    fn test_pow() {
        assert_eq!(pow(3, 0), 1);
        assert_eq!(pow(3, 1), 3);
        assert_eq!(pow(3, 2), 9);
        assert_eq!(pow(3, 3), 27);
    }
    #[test]
    fn test_integ_poly_pow() {
        let term = {
            let mut coefs = ArrayD::zeros(IxDyn(&[2, 3]));
            coefs[[0, 0]] = 1.0;
            coefs[[0, 1]] = 2.0;
            coefs[[0, 2]] = 3.0;
            coefs[[1, 0]] = 4.0;
            coefs[[1, 1]] = 5.0;
            coefs[[1, 2]] = 6.0;
            Polynomial::<f64>::new(coefs)
        };
        let actual = term.pow(2).unwrap();
        let expected = (term.mul(&term)).unwrap();
        assert_eq!(expected, actual);
        let actual = term.pow(3).unwrap();
        let expected = (term.mul(&expected)).unwrap();
        assert_eq!(expected, actual);
        let actual = term.pow(4).unwrap();
        let expected = (term.mul(&expected)).unwrap();
        assert_eq!(expected, actual);
        let actual = term.pow(5).unwrap();
        let expected = (term.mul(&expected)).unwrap();
        assert_eq!(expected, actual);
    }
    #[test]
    fn test_compose() {
        let composer = {
            let mut coefs = ArrayD::zeros(IxDyn(&[2, 3]));
            coefs[[0, 0]] = 1.0;
            coefs[[0, 1]] = 2.0;
            coefs[[0, 2]] = 3.0;
            coefs[[1, 0]] = 4.0;
            coefs[[1, 1]] = 5.0;
            coefs[[1, 2]] = 6.0;
            Polynomial::<f64>::new(coefs)
        };
        let foo = {
            let mut array = ArrayD::zeros(IxDyn(&[3]));
            array[[0]] = 1.0;
            array[[1]] = 2.0;
            array[[2]] = 3.0;
            Polynomial::<f64>::new(array)
        };
        let bar = {
            let mut array = ArrayD::zeros(IxDyn(&[3]));
            array[[0]] = 2.0;
            array[[1]] = 3.0;
            array[[2]] = 4.0;
            Polynomial::<f64>::new(array)
        };
        let expected = {
            let mut array = ArrayD::zeros(IxDyn(&[7]));
            array[[0]] = 55.0;
            array[[1]] = 205.0;
            array[[2]] = 541.0;
            array[[3]] = 817.0;
            array[[4]] = 942.0;
            array[[5]] = 624.0;
            array[[6]] = 288.0;
            Polynomial::<f64>::new(array)
        };
        assert_eq!(composer.compose_scalar(&[&foo, &bar]).unwrap(), expected);
    }
    #[test]
    fn test_view() {
        use ndarray::s;
        let mut array = ArrayD::<f64>::zeros(IxDyn(&[7]));
        array[[0]] = 55.0;
        array[[1]] = 205.0;
        array[[2]] = 541.0;
        array[[3]] = 817.0;
        array[[4]] = 942.0;
        array[[5]] = 624.0;
        array[[6]] = 288.0;
        let foo = array.clone();
        let mut slice = array.slice_mut(s![..]);
        slice *= &foo;
    }
    #[test]
    fn test_reduce_2() {
        let a: &[usize] = &[];
        assert_eq!(a.iter().cloned().reduce(|acc, x| (acc + x)), None);
        let a: &[usize] = &[1];
        assert_eq!(a.iter().cloned().reduce(|acc, x| (acc + x)), Some(1));
        let a: &[usize] = &[1, 2];
        assert_eq!(a.iter().cloned().reduce(|acc, x| (acc + x)), Some(3));
    }
}
