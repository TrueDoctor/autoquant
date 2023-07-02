use std::{error::Error, thread::JoinHandle};

use autoquant::{
    packing::ErrorFunction,
    plot::{plot_channels, plot_errors, plot_errors_with_bits},
};
use rawloader::RawImage;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    /*let first = autoquant::packing::ErrorFunction::new(&[1.0, 0.8, 0.6, 0.4, 0.4, 0.1, 0.1, 0.1]);
    let second = autoquant::packing::ErrorFunction::new(&[1.0, 0.9, 0.8, 0.8, 0.3, 0.2, 0.1, 0.0]);
    let third = autoquant::packing::ErrorFunction::new(&[1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.0, 0.0]);

    let merged = autoquant::packing::merge_error_functions(&first, &second);
    println!("Merged: {:?}", merged);

    let merged = autoquant::packing::merge_error_functions(&merged, &third);
    println!("Merged: {:?}", merged);
    */

    let mut handles = Vec::new();
    handles.push(generate_plot(Diagram::Cdf, 0));
    handles.push(generate_plot(Diagram::Cdf, 1));
    handles.push(generate_plot(Diagram::Cdf, 3));

    handles.push(generate_plot(Diagram::PlotChannels, 0));
    handles.push(generate_plot(Diagram::ErrorDistribution, 0));
    handles.push(generate_plot(Diagram::ErrorDistribution, 1));
    handles.push(generate_plot(Diagram::ErrorDistribution, 3));
    handles.push(generate_plot(Diagram::CombinedErrorFunction, 0));
    for handle in handles {
        handle.join();
    }

    Ok(())
}

fn generate_plot(diagram: Diagram, color_index: usize) -> JoinHandle<()> {
    let foo = move || {
        let color = ["red", "green", "green", "blue"][color_index];
        use autoquant::plot::plot_histogram;
        use std::env;
        let args: Vec<_> = env::args().collect();
        if args.len() != 2 {
            println!("Usage: {} <file>", args[0]);
            std::process::exit(2);
        }
        let file = &args[1];
        let image = rawloader::decode_file(file).unwrap();
        //dbg!(&image.data);
        let len = image.width * image.height / 4;
        let dist = create_distribution(&image, len, color_index);
        let full_dist = create_distribution(&image, len, color_index);
        match diagram {
            Diagram::Cdf => {
                let functions = autoquant::fit_functions(dist.clone(), 40);

                //let models =
                //    autoquant::fit_distributions(data.as_slice(), autoquant::fit_functions().as_slice());
                for fit in functions.iter().take(4) {
                    let error = autoquant::distribution_error(&full_dist, fit.as_ref(), 40);
                    println!("{} Error: {}", fit.name(), error);
                }
                let fits = functions
                    .iter()
                    .take(4)
                    .map(|fit| fit.as_ref() as &dyn autoquant::FitFn)
                    .collect::<Vec<_>>();

                return plot_histogram(&dist, &fits, color);
            }
            Diagram::ErrorDistribution => {
                let errors = autoquant::calculate_error_functions(&dist, &full_dist);
                println!("Errors: {:#?}", errors);
                return plot_errors(&errors.1, &errors.0, color);
            }
            Diagram::PlotChannels => {
                let red = create_distribution(&image, 100, 0);
                let green = create_distribution(&image, 100, 1);
                let blue = create_distribution(&image, 100, 3);
                return plot_channels(&[&red, &green, &blue]);
            }
            Diagram::CombinedErrorFunction => {
                let red = create_distribution(&image, len, 0);
                let green = create_distribution(&image, len, 1);
                let blue = create_distribution(&image, len, 3);
                let fit_red = autoquant::calculate_error_function(&red, 0, &red);
                let fit_green = autoquant::calculate_error_function(&green, 0, &green);
                let fit_blue = autoquant::calculate_error_function(&blue, 0, &blue);
                let red_error: ErrorFunction<10> =
                    autoquant::packing::ErrorFunction::new(fit_red.as_slice());
                let green_error: ErrorFunction<10> =
                    autoquant::packing::ErrorFunction::new(fit_green.as_slice());
                let blue_error: ErrorFunction<10> =
                    autoquant::packing::ErrorFunction::new(fit_blue.as_slice());
                let merged: ErrorFunction<24> =
                    autoquant::packing::merge_error_functions(&red_error, &green_error);
                println!("Merged: {:?}", merged);
                let merged: ErrorFunction<32> =
                    autoquant::packing::merge_error_functions(&merged, &blue_error);
                println!("Merged: {:?}", merged);
                let red_bits = merged.bits.iter().map(|x| x[0]).collect::<Vec<_>>();
                let green_bits = merged.bits.iter().map(|x| x[1]).collect::<Vec<_>>();
                let blue_bits = merged.bits.iter().map(|x| x[2]).collect::<Vec<_>>();
                println!("Red bits: {:?}", red_bits);
                println!("Green bits: {:?}", green_bits);
                println!("Blue bits: {:?}", blue_bits);
                return plot_errors_with_bits(
                    &[fit_red, fit_green, fit_blue][..],
                    &[
                        red_bits.as_slice(),
                        green_bits.as_slice(),
                        blue_bits.as_slice(),
                    ][..],
                    &["red".to_string(), "green".to_string(), "blue".to_string()][..],
                );
            }
        }
    };
    std::thread::spawn(move || foo().unwrap())
}

fn create_distribution(image: &RawImage, samples: usize, channel: usize) -> Vec<(f64, f64)> {
    let rawloader::RawImageData::Integer(ref data) = image.data else {
    panic!("Don't know how to process non-integer raw files");
};
    let xoffset = channel % 2;
    let yoffset = channel / 2;
    let mut output = Vec::with_capacity(data.len() / 4);
    for y in 0..(image.height / 2) {
        for x in 0..(image.width / 2) {
            let index = x * 2 + xoffset + (y * 2 + yoffset) * image.width;
            output.push(data[index])
        }
    }
    let data: Vec<f64> = output
        .chunks(data.len() / (samples.min(data.len())))
        .map(|x| x[0] as f64)
        .collect();
    let max = *data
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    let data: Vec<f64> = data.iter().map(|x| x / max).collect();
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
