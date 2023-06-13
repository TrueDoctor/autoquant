use autoquant::plot::{plot_channels, plot_errors};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let first = autoquant::packing::ErrorFunction::new(&[1.0, 0.8, 0.6, 0.4, 0.4, 0.1, 0.1, 0.1]);
    let second = autoquant::packing::ErrorFunction::new(&[1.0, 0.9, 0.8, 0.8, 0.3, 0.2, 0.1, 0.0]);
    let third = autoquant::packing::ErrorFunction::new(&[1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.0, 0.0]);

    let merged = autoquant::packing::merge_error_functions(&first, &second);
    println!("Merged: {:?}", merged);

    let merged = autoquant::packing::merge_error_functions(&merged, &third);
    println!("Merged: {:?}", merged);
    let diagram = Diagram::ErrorDistribution;

    #[cfg(feature = "plotting")]
    {
        use autoquant::plot::plot_histogram;
        use std::env;

        let args: Vec<_> = env::args().collect();
        if args.len() != 2 {
            println!("Usage: {} <file>", args[0]);
            std::process::exit(2);
        }
        let file = &args[1];
        let image = rawloader::decode_file(file).unwrap();

        dbg!(image.cfa);
        let rawloader::RawImageData::Integer(data) = image.data else {
        panic!("Don't know how to process non-integer raw files");
    };

        let len = data.len();
        let dist = create_distribution(data.clone(), 100000, 3);
        let full_dist = create_distribution(data.clone(), len, 3);

        match diagram {
            Diagram::Cdf => {
                let functions = autoquant::fit_functions(dist.clone(), 50);

                //let models =
                //    autoquant::fit_distributions(data.as_slice(), autoquant::fit_functions().as_slice());
                for fit in functions.iter() {
                    let error = autoquant::distribution_error(&full_dist, fit.as_ref(), 4);
                    println!("{} Error: {}", fit.name(), error);
                }
                let fits = functions
                    .iter()
                    .map(|fit| fit.as_ref() as &dyn autoquant::FitFn)
                    .collect::<Vec<_>>();

                return plot_histogram(&dist, &fits);
            }
            Diagram::ErrorDistribution => {
                let errors = autoquant::calculate_error_functions(&dist, &full_dist);
                println!("Errors: {:#?}", errors);
                return plot_errors(&errors.1, &errors.0);
            }
            Diagram::PlotChannels => {
                let red = create_distribution(data.clone(), data.len(), 0);
                let green = create_distribution(data.clone(), data.len(), 1);
                let blue = create_distribution(data.clone(), data.len(), 2);
                return plot_channels(&[&red, &green, &blue]);
            }
            Diagram::CombinedErrorFunction => {
                let red = create_distribution(data.clone(), data.len(), 0);
                let green = create_distribution(data.clone(), data.len(), 1);
                let blue = create_distribution(data.clone(), data.len(), 2);
                return plot_channels(&[&red, &green, &blue]);
            }
        }
    }
}

fn create_distribution(data: Vec<u16>, samples: usize, channel: usize) -> Vec<(f64, f64)> {
    let data: Vec<f64> = data
        .chunks(4 * (data.len() / (4 * samples.min(data.len() / 4))))
        .map(|x| x[channel] as f64)
        .collect();
    let max = *data
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    let data: Vec<f64> = data.iter().map(|x| x / max).collect();
    dbg!(max);
    //let data = autoquant::generate_normal_distribution(3.0, 1.1, 1000);
    //data.iter_mut().for_each(|x| *x = x.abs());
    let mut dist = autoquant::integrate_distribution(data);
    autoquant::drop_duplicates(&mut dist);
    let dist = autoquant::normalize_distribution(dist.as_slice());
    dist
}

enum Diagram {
    Cdf,
    ErrorDistribution,
    CombinedErrorFunction,
    PlotChannels,
}
