use std::{
    borrow::Cow,
    ops::{Deref, Index, IndexMut},
};

pub const BUCKET_SIZE: usize = 32;
pub type Function = [f64; BUCKET_SIZE];

#[derive(Clone, Debug)]
pub struct ErrorFunction<'a, const N: usize> {
    index: usize,
    function: Cow<'a, [f64; N]>,
    pub bits: Cow<'a, [Vec<usize>; N]>,
}

impl<'a, const N: usize> Deref for ErrorFunction<'a, N> {
    type Target = [f64; N];

    fn deref(&self) -> &Self::Target {
        &self.function
    }
}

impl<'a, const N: usize> Index<usize> for ErrorFunction<'a, N> {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.function[index.min(N - 1)]
    }
}
impl<'a, const N: usize> Index<isize> for ErrorFunction<'a, N> {
    type Output = f64;

    fn index(&self, index: isize) -> &Self::Output {
        &self.function[(index.max(0) as usize).min(N - 1)]
    }
}

impl<'a, const N: usize> IndexMut<usize> for ErrorFunction<'a, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.function.to_mut()[index.min(N - 1)]
    }
}

impl<'a, const N: usize> ErrorFunction<'a, N> {
    pub fn new(function: &[f64]) -> ErrorFunction<N> {
        let mut fn_iter = function.iter();
        ErrorFunction {
            index: 0,
            function: Cow::Owned(core::array::from_fn(|_| *fn_iter.next().unwrap())),
            bits: Cow::Owned(core::array::from_fn(|x| vec![x])),
        }
    }
    pub fn empty() -> ErrorFunction<'static, N> {
        ErrorFunction {
            index: 0,
            function: Cow::Owned([-1.0; N]),
            bits: Cow::Owned(core::array::from_fn(|x| vec![x])),
        }
    }
    pub fn len(&self) -> usize {
        self.function.len()
    }
    pub fn is_empty(&self) -> bool {
        self.function.is_empty()
    }

    pub fn push<const A: usize, const B: usize>(
        &mut self,
        first: &ErrorFunction<'a, A>,
        second: &ErrorFunction<'a, B>,
    ) {
        // use dynamic programming to merge the error functions
        let mut min = f64::MAX;
        let mut first_bits = self.index;
        for i in 0..=self.index {
            //println!("{} {}", i, self.index - i);
            let error = first[i] + second[self.index - i];
            if error < min {
                min = error;
                first_bits = i;
            }
        }
        let index = self.index;
        self[index] = min;
        self.bits.to_mut()[index] = first.bits.as_ref()[first_bits.min(A - 1)].clone();
        self.bits.to_mut()[index].push(self.index - first_bits);
        self.index += 1;
    }
}

pub fn merge_error_functions<'a, const N: usize, const M: usize, const O: usize>(
    first: &ErrorFunction<'a, N>,
    second: &ErrorFunction<'a, M>,
) -> ErrorFunction<'a, O> {
    let mut combined = ErrorFunction::empty();
    // use dynamic programming to merge the error functions
    for _ in 0..O {
        combined.push(first, second);
    }
    combined
}
