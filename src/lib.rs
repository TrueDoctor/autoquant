use nalgebra::Scalar;
use num::traits::Float;
#[cfg(feature = "generation")]
use rand::distributions::Distribution;
#[cfg(feature = "generation")]
use statrs::distribution::Normal;

pub mod sum;

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

pub fn encode(value: f64, fit: &dyn FitFn, samples: u32) -> u64 {
    let mapped_value = fit.function(value);
    let normalized = mapped_value.clamp(0., 1.);
    (normalized * samples as f64) as u64
}
pub fn decode(value: u64, fit: &dyn FitFn, samples: u32) -> f64 {
    fit.inverse(value as f64 / samples as f64)
}

pub fn calculate_sampled_error(distribution: &[f64], fit: &dyn FitFn, samples: u32) -> f64 {
    let mut sumerror = 0.0;
    for sample in distribution {
        let encoded = encode(*sample, fit, samples);
        let decoded = decode(encoded, fit, samples);
        let error = (decoded / sample).abs();
        sumerror += error;
    }
    (sumerror - distribution.len() as f64).abs()
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
pub trait CreateFitFn: FitFn + Sized {
    fn new(params: Vec<f64>) -> Self {
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
#[cfg(feature = "fitting")]
use models::*;

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
pub fn fit_functions(dist: Dist) -> Vec<Box<dyn FitFn>> {
    let max = dist.last().unwrap().0;
    vec![
        /*SimpleFitFn {
            function: |x| x,
            inverse: |x| x,
            name: "identity",
        },*/ /*
           SimpleFitFn {
               function: |x| 1. / x,
               inverse: |x| 1. / x,
               name: "inverse",
           },*/
        Box::new(SimpleFitFn {
            function: move |x| x / max,
            inverse: move |x| x * max,
            name: "identity",
        }),
        Box::new(models::VarPro::<PowerTwo>::new(dist.clone())),
        //Box::new(models::VarPro::<Log>::new(dist)),
        Box::new(models::OptimizedLog::new(dist, 100)),
        /*
        SimpleFitFn {
            function: |x| x.sqrt(),
            inverse: |x| x.powi(2),
            name: "sqrt",
        },
        SimpleFitFn {
            function: |x| x.powi(3),
            inverse: |x| x.powf(1. / 3.),
            name: "power 3",
        },
        SimpleFitFn {
            function: |x| x.ln(),
            inverse: |x| x.exp(),
            name: "ln",
        },
        SimpleFitFn {
            function: |x| x.exp(),
            inverse: |x| x.ln(),
            name: "exp",
        },*/
    ]
}

pub fn linearize_distribution(distribution: &[(f64, f64)], fit: &impl FitFn) -> Vec<(f64, f64)> {
    let mapped_distribution: Vec<_> = distribution
        .iter()
        .map(|&(x, y)| (x, fit.inverse(y)))
        .collect();
    //normalize_distribution(&mapped_distribution)
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
