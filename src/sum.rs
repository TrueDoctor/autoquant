// credit: https://github.com/rust-num/num/issues/309
pub struct Sum {
    partials: Vec<f64>,
}

impl Sum {
    pub fn new() -> Sum {
        Sum { partials: vec![] }
    }

    fn add(&mut self, mut x: f64) {
        let mut j = 0;
        // This inner loop applies `hi`/`lo` summation to each
        // partial so that the list of partial sums remains exact.
        for i in 0..self.partials.len() {
            let mut y: f64 = self.partials[i];
            if x.abs() < y.abs() {
                core::mem::swap(&mut x, &mut y);
            }
            // Rounded `x+y` is stored in `hi` with round-off stored in
            // `lo`. Together `hi+lo` are exactly equal to `x+y`.
            let hi = x + y;
            let lo = y - (hi - x);
            if lo != 0.0 {
                self.partials[j] = lo;
                j += 1;
            }
            x = hi;
        }
        if j >= self.partials.len() {
            self.partials.push(x);
        } else {
            self.partials[j] = x;
            self.partials.truncate(j + 1);
        }
    }

    pub fn sum(&self) -> f64 {
        self.partials.iter().fold(0., |p, q| p + *q)
    }
}

impl Default for Sum {
    fn default() -> Self {
        Self::new()
    }
}

impl std::iter::FromIterator<f64> for Sum {
    fn from_iter<T>(iter: T) -> Sum
    where
        T: IntoIterator<Item = f64>,
    {
        let mut e = Sum::new();
        for i in iter {
            e.add(i);
        }
        e
    }
}
