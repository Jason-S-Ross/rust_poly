// use numpy::{PyArrayDyn, IntoPyArray};
// use numpy::ndarray::{Array, ArrayD, Array1, Dimension, ShapeBuilder, Axis, Dim, IxDyn};
// use numpy::convert::ToPyArray;
// use pyo3::prelude::*;
// use pyo3::types::PyTuple;
// use pyo3::exceptions;
use ndarray::{array, Array, ArrayD, ArrayView, Dim, IxDynImpl, ShapeBuilder, Dimension, IntoDimension, IxDyn, Axis, NdIndex};
use num_traits::identities::{Zero, One};
use num_traits::cast::NumCast;
use std::ops::{Add, Mul, Div};
use std::iter;
use std::convert::{TryFrom, TryInto};
use std::error::Error;

/// Permutations of N things taken k at a time, i.e., k-permutations of N
fn perm(K: usize, n: usize) -> usize {
    if n > K {
        return 0
    }
    if n == 1 {
        return K
    }
    ((K - n + 1)..(K + 1)).reduce(|prod, val| prod * val).unwrap_or(1)
}

#[derive(Debug, Clone, PartialEq)]
struct Polynomial<S>
{
    coefficients: ArrayD<S>,
}

impl<S> Polynomial<S>
where
    S: Add<Output = S> + Mul<Output = S> + Copy + Zero + One,
{
    fn new(coefficients: ArrayD<S>) -> Self {
        Polynomial {
            coefficients
        }
    }
    fn shape(&self) -> Vec<usize> {
        self.coefficients.shape().iter().cloned().collect()
    }
    fn eval<'a, D>(&'a self, values: &ArrayView<S, D>)
                   -> Result<Array<S, Dim<IxDynImpl>>, ()>
    where
        D: Dimension,
        Dim<IxDynImpl>: NdIndex<D>,
    {
        let val_shape: Vec<_> = values.shape().iter().cloned().collect();
        let outputshape = Polynomial::<S>::get_shape_eval(&self.shape(), &val_shape);
        let mut output = Array::<S, _>::ones(outputshape?);
        self.eval_no_alloc(values, &mut output)?;
        Ok(output)
    }
    fn eval_no_alloc<D>(
        &self,
        values: &ArrayView<S, D>,
        output: &mut Array<S, Dim<IxDynImpl>>
    ) -> Result<(), ()>
    where
        D: Dimension,
        Dim<IxDynImpl>: NdIndex<D>,
    {
        for (result_index, result_val) in output.indexed_iter_mut() {
            let this_slice = result_index.as_array_view().to_slice().unwrap_or(&[]);
            let res = self.coefficients
                    .indexed_iter()
                    .map(|(index, coef)| {
                        let term = index.into_dimension()
                            .as_array_view()
                            .iter()
                            .enumerate()
                            .map(|(dim, power)| {
                                let slice = IxDyn([this_slice, &[dim]].concat().as_slice());
                                let val = *values.get(slice)?;
                                let prod = iter::repeat(val)
                                    .take(*power)
                                    .reduce(|p, q| p * q)?;
                                Some(prod)
                            })
                            .reduce(|acc, val| {
                                match (acc, val) {
                                    (None, None) => None,
                                    (Some(x), None) => Some(x),
                                    (None, Some(x)) => Some(x),
                                    (Some(x), Some(y)) => Some(x * y)
                                }
                            })?;
                        match term {
                            Some(t) => Some(*coef * t),
                            None => Some(*coef)
                        }
                    })
                .reduce(|acc, term| {
                    match (acc, term) {
                        (None, None) => None,
                        (Some(acc), None) => Some(acc),
                        (None, Some(term)) => Some(term),
                        (Some(acc), Some(term)) => Some(acc + term),
                    }
                });
            *result_val = res.ok_or(())?.ok_or(())?;
        }
        Ok(())
    }
    /// Gets the shape of the result of evaluating a polynomial
    fn get_shape_eval(coef_shape: &[usize], val_shape: &[usize]) -> Result<Vec<usize>, ()> {
        if val_shape[val_shape.len()-1] != coef_shape.len() {
            return Err(())
        }
        Ok(val_shape[..val_shape.len()-1].to_vec())
    }

    /// Gets the shape of the result of a multiplication operation
    fn get_shape_mul(left_shape: &[usize], right_shape: &[usize]) -> Result<Vec<usize>, ()> {
        if left_shape.len() != right_shape.len() {
            return Err(())
        }
        let new_shape: Vec<_> = left_shape
            .iter()
            .zip(right_shape.iter()).map(|(left, right)| {
                left + right - 1
            }).collect();
       Ok(new_shape)
    }
    /// Gets the shape of the result of an addition operation
    fn get_shape_add(left_shape: &[usize], right_shape: &[usize]) -> Result<Vec<usize>, ()> {
        if left_shape.len() != right_shape.len() {
            return Err(())
        }
        let new_shape: Vec<_> = left_shape
            .iter()
            .zip(right_shape.iter()).map(|(left, right)| {
                if left > right {*left} else {*right}
            }).collect();
       Ok(new_shape)
    }
    /// Gets the shape of the result of an integral operation
    fn get_shape_integ(shape: &[usize], arg: &[usize]) -> Result<Vec<usize>, ()> {
        if shape.len() != arg.len() {
            return Err(())
        }
        let new_shape: Vec<_> = shape
            .iter()
            .zip(arg.iter()).map(|(left, right)| *left + *right)
            .collect();
        Ok(new_shape)
    }
    /// Gets the shape of the result of a derivative operation
    fn get_shape_deriv(shape: &[usize], arg: &[usize]) -> Result<Vec<usize>, ()> {
        if shape.len() != arg.len() {
            return Err(())
        }
        let new_shape: Vec<_> = shape
            .iter()
            .zip(arg.iter()).map(|(left, right)| left.saturating_sub(*right))
            .collect();
        Ok(new_shape)
    }
    /// Panics on out-of-bounds
    fn deriv_no_alloc(&self, arg: &[usize], output: &mut ArrayD<S>) -> Result<(), ()>
        where S: NumCast + std::fmt::Debug
    {
        let index_size = self.shape().len();
        for (ref index, _val) in self.coefficients.indexed_iter() {
            let factor = index.clone()
                 .into_dimension()
                 .as_array_view()
                 .iter()
                 .zip(arg.iter())
                 .map(|(index, arg)| {
                     perm(*index, *arg)
                 })
                .reduce(|prod, val| prod * val)
                .unwrap_or(1); // TODO Check if 1 is the right case here
            let new_index: Vec<_> = index
                .as_array_view()
                .iter()
                .zip(arg.iter())
                .map_while(|(left, right)| {
                    // Use map_while and checked_sub so that if the index is negative,
                    // we ignore it
                    left.checked_sub(*right)
                })
                .collect();
            if new_index.len() != index_size {
                // True if any of the indices underflowed
                continue
            }
            output[new_index.as_slice()] = self.coefficients[index] * <S as NumCast>::from(factor).ok_or(())?;
        }
        Ok(())
    }
    fn deriv(&self, arg: &[usize]) -> Result<Self, ()>
    where
        S: NumCast+ std::fmt::Debug
    {
        let new_shape = Polynomial::<S>::get_shape_deriv(&self.shape(), &arg)?;
        let mut result = ArrayD::<S>::zeros(new_shape);
        self.deriv_no_alloc(arg, &mut result).or(Err(()))?;
        Ok(Polynomial::new(result))
    }
    /// Panics on out-of-bounds
    fn integ_no_alloc(&self, arg: &[usize], output: &mut ArrayD<S>) -> Result<(), ()>
        where S: NumCast + Div<Output = S> + std::fmt::Debug
    {
        let index_size = self.shape().len();
        for (ref index, _val) in self.coefficients.indexed_iter() {
            let factor = index.clone()
                 .into_dimension()
                 .as_array_view()
                 .iter()
                 .zip(arg.iter())
                 .map(|(index, arg)| {
                     perm(*index + *arg, *arg)
                 })
                .reduce(|prod, val| prod * val)
                .unwrap_or(1); // TODO Check if 1 is the right case here
            let new_index: Vec<_> = index
                .as_array_view()
                .iter()
                .zip(arg.iter())
                .map(|(left, right)| {
                    *left + *right
                })
                .collect();
            output[new_index.as_slice()] = self.coefficients[index] / <S as NumCast>::from(factor).ok_or(())?;
        }
        Ok(())
    }
    fn integ(&self, arg: &[usize]) -> Result<Self, ()>
    where
        S: NumCast + std::fmt::Debug + Div<Output = S>
    {
        let new_shape = Polynomial::<S>::get_shape_integ(&self.shape(), &arg)?;
        let mut result = ArrayD::<S>::zeros(new_shape);
        self.integ_no_alloc(arg, &mut result).or(Err(()))?;
        Ok(Polynomial::new(result))
    }

    /// Adds two polynomials without allocating space for either. Panics on
    /// out-of-bounds
    fn add_no_alloc(&self, other: &Self, output: &mut ArrayD<S>)  {
        for (ref index, val) in self.coefficients.indexed_iter() {
            output[index] = output[index] + *val;
        }
        for (ref index, val) in other.coefficients.indexed_iter() {
            output[index] = output[index] + *val;
        }
    }
    /// Multiplies two polynomials without allocating space for either. Panics
    /// on out-of-bounds
    fn mul_no_alloc(&self, other: &Self, output: &mut ArrayD<S>) {
        for (ref left_index, left_val) in self.coefficients.indexed_iter() {
            for (ref right_index, right_val) in other.coefficients.indexed_iter() {
                let output_index: Vec<_> = left_index
                    .clone()
                    .into_dimension()
                    .as_array_view()
                    .iter()
                    .zip(
                        right_index
                            .clone()
                            .into_dimension()
                            .as_array_view()
                            .iter())
                    .map(|(left, right)| *left + *right)
                    .collect();
                output[output_index.as_slice()] = output[output_index.as_slice()] + *left_val * *right_val;
            }
        };
    }
}


impl<S> Mul for &Polynomial<S>
where
    S: Clone + Zero + One + Mul<Output = S> + Copy + Add,
{
    type Output = Result<Polynomial<S>, ()>;

    fn mul(self, rhs: Self) -> Self::Output {
        let new_shape = Polynomial::<S>::get_shape_mul(&self.shape(), &rhs.shape())?;
        let mut result = ArrayD::<S>::zeros(new_shape);
        self.mul_no_alloc(&rhs, &mut result);
        Ok(Polynomial::new(result))
    }
}

impl<S> Add for &Polynomial<S>
where
    S: Clone + Zero + One + Copy + Add,
{
    type Output = Result<Polynomial<S>, ()>;

    fn add(self, rhs: Self) -> Self::Output {
        let new_shape = Polynomial::<S>::get_shape_add(&self.shape(), &rhs.shape())?;
        let mut result = ArrayD::<S>::zeros(new_shape);
        self.add_no_alloc(&rhs, &mut result);
        Ok(Polynomial::new(result))
    }
}



#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn it_works() {
        assert_eq!(1, 1);
    }
    #[test]
    fn make_eval_poly_1d() {
        let mut array = ArrayD::zeros(IxDyn(&[3]));
        array[[0]] = 1.0;
        array[[1]] = 2.0;
        array[[2]] = 3.0;
        let foo = Polynomial::new(array);
        let mut bar = Array::zeros(IxDyn(&[1]));
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
        let actual = &foo * &foo;
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
        let actual = &foo * &bar;
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
        let actual = &foo + &bar;
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
        assert_eq!(&foo * &bar, &bar * &foo);
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
    fn square_poly_2d() {
        let term = {
            let mut coefs = ArrayD::zeros(IxDyn(&[2, 2]));
            coefs[[0, 0]] = 1.0;
            coefs[[0, 1]] = 2.0;
            coefs[[1, 0]] = 3.0;
            coefs[[1, 1]] = 4.0;
            Polynomial::<f64>::new(coefs)
        };
        let actual = (&term * &term).unwrap();
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
        let res = foo.iter().map(|v| *v).reduce(|acc, val| {
            println!("acc: {:#?}, val: {:#?}", acc, val);
            match (acc, val) {
                (None, None) => None,
                (Some(x), None) => Some(x),
                (None, Some(x)) => Some(x),
                (Some(x), Some(y)) => Some(x + y),
            }
            //Some((acc? + val?))
        });
        println!("res: {:?}", res);
    }
    #[test]
    fn test_array0() {
        use ndarray::arr0;
        let foo = arr0(5.0);
        let dim = foo.raw_dim();
        println!("test dim: {:#}", dim.as_array_view());
        println!("foo: {:#}", foo);
    }
    #[test]
    fn test_array2() {
        let bar = array![[1.0, 2.0],[3.0, 4.0]];
        println!("bar: {:?}", bar);
        let dim = bar.raw_dim();
        let (_, bazdim) = dim.as_array_view().split_at(Axis(0), dim.as_array_view().len() - 1);
        println!("bar dim: {:#?}", bazdim);
        let baz = Array::<f64, _>::zeros(bazdim.to_slice().unwrap());
        println!("baz: {:#}", baz);
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
}
