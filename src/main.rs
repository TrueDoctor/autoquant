use autoquant::plot::plot_histogram;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut data = autoquant::generate_normal_distribution(0.0, 1.1, 1000);
    data.iter_mut().for_each(|x| *x = x.abs());
    let data = autoquant::integrate_distribution(data);
    let data = autoquant::normalize_distribution(data.as_slice());
    let fit = autoquant::fit_distributions(data.as_slice(), &autoquant::FIT_FUNCTIONS);

    plot_histogram(&data)
}
