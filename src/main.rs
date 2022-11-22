use autoquant::plot::plot_histogram;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = autoquant::generate_normal_distribution(0.0, 1.0, 1000);
    let data = autoquant::integrate_distribution(data);
    let data = autoquant::normalize_distribution(data.as_slice());
    let fit = autoquant::fit_distributions(data.as_slice(), &autoquant::FIT_FUNCTIONS);

    plot_histogram(&data)
}
