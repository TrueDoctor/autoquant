use linregress::{FormulaRegressionBuilder, RegressionDataBuilder, RegressionModel};
use rand::distributions::Distribution;
use statrs::distribution::Normal;

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

pub fn normalize_distribution(distribution: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let max = distribution.last().unwrap().1;
    distribution.iter().map(|&(x, y)| (x, y / max)).collect()
}

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

pub fn encode(value: f64, fit: &impl FitFn, model: &RegressionModel, samples: u32) -> u64 {
    let mapped_value = fit.function(model.predict([("X", vec![value])]).unwrap()[0]);
    let normalized = mapped_value.clamp(0., 1.);
    (normalized * samples as f64) as u64
}
pub fn decode(value: u64, fit: &impl FitFn, model: &RegressionModel, samples: u32) -> f64 {
    let mapped_value = fit.inverse(value as f64 / samples as f64);
    let offset = model.parameters()[0];
    let muliplier = model.parameters()[1];
    ((mapped_value - offset) / muliplier) as f64
}

pub fn calculate_sampled_error(
    distribution: &[f64],
    fit: &impl FitFn,
    model: &RegressionModel,
    samples: u32,
) -> f64 {
    let mut sumerror = 0.0;
    let mut outliers = 0;
    for sample in distribution {
        let encoded = encode(*sample, fit, model, samples);
        let decoded = decode(encoded, fit, model, samples);
        let error = (decoded - sample).powi(2);
        // TODO: Fix
        if error.is_finite() {
            sumerror += error;
        } else {
            outliers += 1;
        }
    }
    sumerror / (distribution.len() - outliers) as f64
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

pub trait FitFn {
    fn function(&self, x: f64) -> f64;
    fn inverse(&self, x: f64) -> f64;
    fn name(&self) -> &str;
}

pub struct SimpleFitFn {
    function: fn(f64) -> f64,
    inverse: fn(f64) -> f64,
    name: &'static str,
}

impl FitFn for SimpleFitFn {
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

pub static FIT_FUNCTIONS: [SimpleFitFn; 7] = [
    SimpleFitFn {
        function: |x| x,
        inverse: |x| x,
        name: "identity",
    },
    SimpleFitFn {
        function: |x| 1. / x,
        inverse: |x| 1. / x,
        name: "inverse",
    },
    SimpleFitFn {
        function: |x| x.powi(2),
        inverse: |x| x.sqrt(),
        name: "power 2",
    },
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
        function: |x| x.log10(),
        inverse: |x| x.powf(10.),
        name: "log10",
    },
    SimpleFitFn {
        function: |x| x.powf(10.),
        inverse: |x| x.log10(),
        name: "exp",
    },
];

pub fn linearize_distribution(distribution: &[(f64, f64)], fit: &impl FitFn) -> Vec<(f64, f64)> {
    let mapped_distribution: Vec<_> = distribution
        .iter()
        .map(|&(x, y)| (x, fit.inverse(y)))
        .collect();
    //normalize_distribution(&mapped_distribution)
    mapped_distribution
}

pub fn fit_distribution(
    distribution: &[(f64, f64)],
    fit: &impl FitFn,
) -> anyhow::Result<RegressionModel> {
    let linearized_distribution = linearize_distribution(distribution, fit);
    let x = linearized_distribution
        .iter()
        .map(|&(x, _)| x)
        .collect::<Vec<_>>();
    let y = linearized_distribution
        .iter()
        .map(|&(_, y)| y)
        .collect::<Vec<_>>();
    let data = vec![("Y", y), ("X", x)];
    let data = RegressionDataBuilder::new().build_from(data)?;
    let model = FormulaRegressionBuilder::new()
        .data(&data)
        .formula("Y ~ X")
        .fit()?;
    let error = calculate_sampled_error(
        &distribution.iter().map(|&(x, _)| x).collect::<Vec<_>>(),
        fit,
        &model,
        1000,
    );
    println!(
        "{} \tr²={:?} \t sampled_error: {}",
        fit.name(),
        model.rsquared(),
        error
    );
    Ok(model)
}

pub fn fit_distributions(
    distribution: &[(f64, f64)],
    fits: &[impl FitFn],
) -> anyhow::Result<Vec<RegressionModel>> {
    fits.iter()
        .map(|fit| fit_distribution(distribution, fit))
        .collect()
}

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
