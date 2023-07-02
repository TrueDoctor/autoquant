#[cfg(feature = "generation")]
use rand::distributions::Distribution;
#[cfg(feature = "generation")]
use statrs::distribution::Normal;

use rayon::prelude::*;

pub mod sum;

pub mod packing;

pub fn integrate_distribution(mut distribution: Vec<f64>) -> Vec<(f64, f64)> {
    distribution.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let mut result = Vec::with_capacity(distribution.len());
    let mut sum = 0.0;
    for value in distribution {
        sum += 1.;
        result.push((value, sum));
    }
    result
}

pub fn drop_duplicates(distribution: &mut Vec<(f64, f64)>) {
    distribution.reverse();
    distribution.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    distribution.dedup_by(|a, b| a.0 == b.0);
    distribution.reverse();
}

pub fn normalize_distribution(distribution: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let max_y = distribution.last().unwrap().1;
    let max_x = distribution.last().unwrap().0;
    distribution
        .iter()
        .map(|&(x, y)| (x / max_x, y / max_y))
        .collect()
}

#[cfg(feature = "generation")]
pub fn generate_normal_distribution(
    mean: f64,
    standard_deviation: f64,
    number_of_samples: usize,
) -> Vec<f64> {
    let normal = Normal::new(mean, standard_deviation).unwrap();
    let mut rng = rand::thread_rng();
    let mut result = Vec::with_capacity(number_of_samples);
    for _ in 0..number_of_samples {
        result.push(normal.sample(&mut rng));
    }
    result
}

pub fn encode(value: f64, fit: &dyn FitFn, samples: u64) -> u64 {
    let mut mapped_value = fit.function(value);
    if !mapped_value.is_finite() {
        mapped_value = 0.;
    }
    assert!(!mapped_value.is_nan(), "mapped value is NaN");
    (mapped_value * samples as f64).clamp(0., samples as f64) as u64
}
pub fn decode(value: u64, fit: &dyn FitFn, samples: u64) -> f64 {
    let mut x = value as f64 / samples as f64;
    if !x.is_finite() {
        x = 0.;
    }
    x = fit.inverse(x);
    if !x.is_finite() {
        x = 0.;
    }
    x
}

pub fn distribution_error<T: FitFn>(data: &[(f64, f64)], fun: T, quantization: u64) -> f64 {
    let mut last_y = 0.;
    let iter = data.iter().map(|&(x, y)| {
        let encoded = encode(x, &fun, quantization);
        let decoded = decode(encoded, &fun, quantization);

        let weight = |x: f64| (-(x).abs().clamp(0., 1.)).exp();
        let error = ((decoded - x) * 100.).powi(2) * (y - last_y); // * weight(x));
                                                                   //dbg!(x, (y - last_y) * weight(x));
        last_y = y;
        //dbg!(error, x, weight(x));

        assert!(
            error.is_finite(),
            "encontered non-finite error {} for encoded value {}, decoded value {}, x: {}",
            error,
            encoded,
            decoded,
            encoded as f64 / quantization as f64,
        );
        error
    });

    crate::sum::Sum::from_iter(iter).sum().sqrt()
}

pub fn inverse_of_distribution(distribution: &[(f64, f64)], y: f64) -> f64 {
    // TODO use lerp
    distribution
        .iter()
        .find(|&(_, y_)| *y_ > y)
        .map(|(x, _)| *x)
        .unwrap()
}

pub fn inverse_of_fn(f: impl Fn(f64) -> f64, x: f64) -> f64 {
    let mut a = -100.0;
    let mut b = 100.0;
    let mut c = 1.0;
    let mut y = f(c);
    let mut counter = 0;
    while (y - x).abs() > 1e-6 {
        counter += 1;
        if (y > x) == (f(c + 1e-6) > f(c)) {
            b = c;
        } else {
            a = c;
        }
        c = (a + b) / 2.;
        y = f(c);
        if counter > 1000 {
            panic!("Could not find inverse of function");
        }
    }
    c
}
type Dist = Vec<(f64, f64)>;

pub trait FitFn {
    fn function(&self, x: f64) -> f64;
    fn inverse(&self, x: f64) -> f64;
    fn name(&self) -> &str;
}

impl<T: FitFn> FitFn for &T {
    fn function(&self, x: f64) -> f64 {
        (*self).function(x)
    }
    fn inverse(&self, x: f64) -> f64 {
        (*self).inverse(x)
    }
    fn name(&self) -> &str {
        (*self).name()
    }
}
impl FitFn for &dyn FitFn {
    fn function(&self, x: f64) -> f64 {
        (*self).function(x)
    }
    fn inverse(&self, x: f64) -> f64 {
        (*self).inverse(x)
    }
    fn name(&self) -> &str {
        (*self).name()
    }
}

pub trait CreateFitFn: FitFn + Sized {
    fn new(_params: Vec<f64>) -> Self {
        unimplemented!()
    }
}

pub struct SimpleFitFn<F: Fn(f64) -> f64, I: Fn(f64) -> f64> {
    pub function: F,
    pub inverse: I,
    pub name: &'static str,
}

#[cfg(feature = "fitting")]
pub mod models;

impl<F: Fn(f64) -> f64, I: Fn(f64) -> f64> FitFn for SimpleFitFn<F, I> {
    fn function(&self, x: f64) -> f64 {
        (self.function)(x)
    }
    fn inverse(&self, x: f64) -> f64 {
        (self.inverse)(x)
    }
    fn name(&self) -> &str {
        self.name
    }
}

#[cfg(feature = "fitting")]
pub fn calculate_error_functions(train: &Dist, test: &Dist) -> (Vec<String>, Vec<Vec<f64>>) {
    let names = vec![
        "linear".to_string(),
        "log".to_string(),
        "pow".to_string(),
        "exp".to_string(),
    ];
    let mut errors = (0..names.len()).map(|_| Vec::new()).collect::<Vec<_>>();

    for i in 0..names.len() {
        errors[i] = calculate_error_function(train, i, test);
    }
    (names, errors)
}

pub fn calculate_error_function(train: &[(f64, f64)], i: usize, test: &[(f64, f64)]) -> Vec<f64> {
    let bits: Vec<_> = (0..12).collect();
    let errors: Vec<_> = bits
        .par_iter()
        .map(|bits| {
            let levels = 1 << bits;
            let fn_ = fit_function(train.to_vec(), levels, i);
            let error = distribution_error(test, fn_.as_ref(), levels);
            error
        })
        .collect();
    errors
}
#[cfg(feature = "fitting")]
pub fn fit_functions(dist: Dist, levels: u64) -> Vec<Box<dyn FitFn>> {
    vec![
        fit_function(dist.clone(), levels, 0),
        fit_function(dist.clone(), levels, 1),
        fit_function(dist.clone(), levels, 2),
        fit_function(dist, levels, 3),
    ]
}
#[cfg(feature = "fitting")]
pub fn fit_function(dist: Dist, levels: u64, index: usize) -> Box<dyn FitFn> {
    match index {
        0 => Box::new(models::OptimizedLin::new(dist, levels)),
        1 => Box::new(models::OptimizedLog::new(dist, levels)),
        2 => Box::new(models::OptimizedPow::new(dist, levels)),
        3 => Box::new(models::OptimizedExp::new(dist, levels)),
        _ => panic!("Unknown fit function"),
    }
}

pub fn linearize_distribution(distribution: &[(f64, f64)], fit: &impl FitFn) -> Vec<(f64, f64)> {
    let mapped_distribution: Vec<_> = distribution
        .iter()
        .map(|&(x, y)| (x, fit.inverse(y)))
        .collect();
    mapped_distribution
}

#[cfg(feature = "plotting")]
pub mod plot;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse() {
        let inverse = inverse_of_fn(|x| 1. / x, 0.5);
        assert!(dbg!((inverse) - 2.).abs() < 1e-5);
    }
}
