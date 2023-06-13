use autoquant::plot::plot_errors;

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
        use std::fs::File;
        use std::io::{prelude::*, BufWriter};

        let args: Vec<_> = env::args().collect();
        if args.len() != 2 {
            println!("Usage: {} <file>", args[0]);
            std::process::exit(2);
        }
        let file = &args[1];
        let image = rawloader::decode_file(file).unwrap();

        let rawloader::RawImageData::Integer(data) = image.data else {
        panic!("Don't know how to process non-integer raw files");
    };

        let data: Vec<f64> = data
            .iter()
            .map(|x| *x as f64)
            .take(data.len() / 10000)
            .collect();
        let max = *data
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        let data: Vec<f64> = data.iter().map(|x| x / max).collect();
        dbg!(max);
        //let data = autoquant::generate_normal_distribution(3.0, 1.1, 1000);
        //data.iter_mut().for_each(|x| *x = x.abs());
        let mut dist = autoquant::integrate_distribution(data.clone());
        autoquant::drop_duplicates(&mut dist);
        let dist = autoquant::normalize_distribution(dist.as_slice());
        //println!("dist: {:?}", dist);
        //
        match diagram {
            Diagram::Cdf => {
                let functions = autoquant::fit_functions(dist.clone());

                //let models =
                //    autoquant::fit_distributions(data.as_slice(), autoquant::fit_functions().as_slice());
                for fit in functions.iter() {
                    let error = autoquant::calculate_sampled_error(&data, fit.as_ref(), 4);
                    println!("{} Error: {}", fit.name(), error);
                }
                let fits = functions
                    .iter()
                    .map(|fit| fit.as_ref() as &dyn autoquant::FitFn)
                    .collect::<Vec<_>>();

                return plot_histogram(&dist, &fits);
            }
            Diagram::ErrorDistribution => {
                let errors = autoquant::calculate_error_functions(dist, &data);
                println!("Errors: {:#?}", errors);
                return plot_errors(&errors.1, &errors.0);
            }
        }
    }
}

enum Diagram {
    Cdf,
    ErrorDistribution,
}
