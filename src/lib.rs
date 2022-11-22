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
