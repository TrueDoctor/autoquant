fn main() -> Result<(), Box<dyn std::error::Error>> {
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
        //let data: Vec<f64> = data.iter().map(|x| x / max).collect();
        dbg!(max);
        //let data = autoquant::generate_normal_distribution(3.0, 1.1, 1000);
        //data.iter_mut().for_each(|x| *x = x.abs());
        let mut dist = autoquant::integrate_distribution(data.clone());
        autoquant::drop_duplicates(&mut dist);
        let dist = autoquant::normalize_distribution(dist.as_slice());
        //println!("dist: {:?}", dist);
        let functions = autoquant::fit_functions(dist.clone());
        //let models =
        //    autoquant::fit_distributions(data.as_slice(), autoquant::fit_functions().as_slice());
        for fit in functions.iter() {
            let error = autoquant::calculate_sampled_error(&data, fit.as_ref(), 100);
            println!("{} Error: {}", fit.name(), error);
        }
        let fits = functions
            .iter()
            .map(|fit| fit.as_ref() as &dyn autoquant::FitFn)
            .collect::<Vec<_>>();

        return plot_histogram(&dist, &fits);
        //plot_histogram(&data, &[])
    }
    Ok(())
}
