use std::{
    borrow::Cow,
    ops::{Deref, Index, IndexMut},
};

pub const BUCKET_SIZE: usize = 16;
pub type Function = [f64; BUCKET_SIZE];

#[derive(Clone, Debug)]
pub struct ErrorFunction<'a> {
    index: usize,
    function: Cow<'a, [f64; BUCKET_SIZE]>,
    bits: Cow<'a, [Vec<usize>; BUCKET_SIZE]>,
}

impl<'a> Deref for ErrorFunction<'a> {
    type Target = Function;

    fn deref(&self) -> &Self::Target {
        &self.function
    }
}

impl<'a> Index<usize> for ErrorFunction<'a> {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.function[index]
    }
}
impl<'a> Index<isize> for ErrorFunction<'a> {
    type Output = f64;

    fn index(&self, index: isize) -> &Self::Output {
        if index < 0 {
            &0.
        } else {
            &self.function[index as usize]
        }
    }
}

impl<'a> IndexMut<usize> for ErrorFunction<'a> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.function.to_mut()[index]
    }
}

impl<'a> ErrorFunction<'a> {
    pub fn new(function: &[f64]) -> ErrorFunction {
        let mut fn_iter = function.iter();
        ErrorFunction {
            index: 0,
            function: Cow::Owned(core::array::from_fn(|_| *fn_iter.next().unwrap())),
            bits: Default::default(),
        }
    }
    pub fn empty() -> ErrorFunction<'static> {
        ErrorFunction {
            index: 0,
            function: Cow::Owned([-1.0; BUCKET_SIZE]),
            bits: Default::default(),
        }
    }
    pub fn len(&self) -> usize {
        self.function.len()
    }
    pub fn is_empty(&self) -> bool {
        self.function.is_empty()
    }

    pub fn push(&mut self, first: &Self, second: &Self) {
        // use dynamic programming to merge the error functions
        let mut min = f64::MAX;
        self.bits.to_mut()[self.index] = first.bits.as_ref()[self.index].clone();
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
        self.bits.to_mut()[index].push(first_bits);
        self.index += 1;
    }
}

pub fn merge_error_functions<'a>(
    first: &ErrorFunction<'a>,
    second: &ErrorFunction<'a>,
) -> ErrorFunction<'a> {
    let mut combined = ErrorFunction::empty();
    // use dynamic programming to merge the error functions
    for _ in 0..first.len() {
        combined.push(first, second);
    }
    combined
}
