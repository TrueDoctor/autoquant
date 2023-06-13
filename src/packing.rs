use std::{
    borrow::Cow,
    ops::{Deref, Index, IndexMut},
};

pub const BUCKET_SIZE: usize = 8;
pub type Function = [f32; BUCKET_SIZE];

#[derive(Clone, Debug)]
pub struct ErrorFunction<'a> {
    index: usize,
    function: Cow<'a, [f32; BUCKET_SIZE]>,
}

impl<'a> Deref for ErrorFunction<'a> {
    type Target = Function;

    fn deref(&self) -> &Self::Target {
        &self.function
    }
}

impl<'a> Index<usize> for ErrorFunction<'a> {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.function[index]
    }
}
impl<'a> Index<isize> for ErrorFunction<'a> {
    type Output = f32;

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
    pub fn new(function: &Function) -> ErrorFunction {
        ErrorFunction {
            index: 0,
            function: Cow::Borrowed(function),
        }
    }
    pub fn empty() -> ErrorFunction<'static> {
        ErrorFunction {
            index: 0,
            function: Cow::Owned([-1.0; BUCKET_SIZE]),
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
        let mut min = f32::MAX;
        for i in 0..=self.index {
            //println!("{} {}", i, self.index - i);
            let error = first[i] + second[self.index - i];
            if error < min {
                min = error;
            }
        }
        let index = self.index;
        self[index] = min;
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
