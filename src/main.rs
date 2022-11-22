use autoquant::plot::plot_histogram;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = autoquant::generate_normal_distribution(3.0, 1.1, 1000);
    //data.iter_mut().for_each(|x| *x = x.abs());
    let data = autoquant::integrate_distribution(data);
    let data = autoquant::normalize_distribution(data.as_slice());
    let models = autoquant::fit_distributions(data.as_slice(), &autoquant::FIT_FUNCTIONS);
    let fits = autoquant::FIT_FUNCTIONS
        .iter()
        .map(|fit| fit as &dyn autoquant::FitFn)
        .zip(models.as_ref().unwrap().iter())
        .collect::<Vec<_>>();

    plot_histogram(&data, &fits)
}
