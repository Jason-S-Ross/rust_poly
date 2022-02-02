use crate::polynomial::Polynomial;

/// A lazy polynomial expression
pub enum Expression<T> {
    Polynomial(Polynomial<T>),
    Add {
        left: Box<Expression<T>>,
        right: Box<Expression<T>>,
    },
    Mul {
        left: Box<Expression<T>>,
        right: Box<Expression<T>>,
    },
    Div {
        num: Box<Expression<T>>,
        denom: Box<Expression<T>>
    },
    Scale {
        scale: T,
        expression: Box<Expression<T>>,
    },
    Neg (Box<Expression<T>>),
    Deriv {
        expression: Box<Expression<T>>,
        wrt: Vec<T>,
    },
    Integ {
        expression: Box<Expression<T>>,
        wrt: Vec<T>,
    }
}
