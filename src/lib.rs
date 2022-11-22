use linregress::{assert_almost_eq, FormulaRegressionBuilder, RegressionDataBuilder};
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

pub fn calculate_sampled_error(
    samples: usize,
    distribution: &[(f64, f64)],
    fit: impl Fn(f64) -> f64 + Copy,
) -> f64 {
    (0..samples)
        .into_iter()
        .map(|x| x as f64 / samples as f64)
        .map(|x| {
            let y = inverse_of_distribution(distribution, x);
            let y_fit = inverse_of_fn(fit, x);
            (y - y_fit).powi(2)
        })
        .sum::<f64>();
    distribution
        .iter()
        .map(|&(x, y)| (y - fit(x)).powi(2))
        .sum()
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
    while (y - x).abs() > 1e-6 {
        if (y > x) == (f(c + 1e-6) > f(c)) {
            b = c;
        } else {
            a = c;
        }
        c = (a + b) / 2.;
        y = f(c);
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

pub static FIT_FUNCTIONS: [SimpleFitFn; 4] = [
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
        function: |x| x.powi(3),
        inverse: |x| x.powf(1. / 3.),
        name: "power 3",
    },
];

pub fn linearize_distribution(distribution: &[(f64, f64)], fit: &impl FitFn) -> Vec<(f64, f64)> {
    let mapped_distribution: Vec<_> = distribution
        .iter()
        .map(|&(x, y)| (x, fit.inverse(y)))
        .collect();
    normalize_distribution(&mapped_distribution)
}

pub fn fit_distribution(
    distribution: &[(f64, f64)],
    fit: &impl FitFn,
) -> anyhow::Result<(f32, f32)> {
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
    let parameters: Vec<_> = model.iter_parameter_pairs().collect();
    let pvalues: Vec<_> = model.iter_p_value_pairs().collect();
    let standard_errors: Vec<_> = model.iter_se_pairs().collect();
    println!(
        "{}\nparameters: {:?}\npvalues{:?}\nerrors{:?}",
        fit.name(),
        parameters,
        pvalues,
        standard_errors
    );
    Ok((0., 0.))
}

pub fn fit_distributions(
    distribution: &[(f64, f64)],
    fits: &[impl FitFn],
) -> anyhow::Result<Vec<(f32, f32)>> {
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
