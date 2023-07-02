use plotters::{prelude::*, style::full_palette::PINK};

use crate::FitFn;
const OUT_FILE_NAME: &str = "histogram.svg";

pub fn plot_histogram(
    data: &[(f64, f64)],
    fits: &[&dyn FitFn],
    color: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let name = format!("out/cdf_{}.svg", color);
    let root = SVGBackend::new(name.as_str(), (800, 600)).into_drawing_area();

    root.fill(&WHITE)?;
    let minx = data.first().expect("No data").0;
    let maxx = data.last().expect("No data").0;
    let miny = data.first().expect("No data").1;
    let maxy = data.last().expect("No data").1;

    let caption = format!("CDF approximation for the {} channel", color);
    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(35)
        .y_label_area_size(80)
        .margin(5)
        .caption(caption.as_str(), ("sans-serif", 40.0))
        .build_cartesian_2d(minx..maxx, miny..maxy)?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .bold_line_style(WHITE.mix(0.3))
        .y_desc("Encoding")
        .x_desc("Input")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    chart
        .draw_series(LineSeries::new(data.iter().map(|(x, y)| (*x, *y)), RED))?
        .label("Target")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    for (i, fit) in fits.iter().enumerate() {
        let color = Palette99::pick(i + 3);
        let fit_data = data
            .iter()
            .map(|&(x, _)| (x, fit.function(x)))
            .collect::<Vec<_>>();
        chart
            .draw_series(LineSeries::new(fit_data.iter().cloned(), &color))?
            .label(fit.name())
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &color));
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    /*chart.draw_series(
        Histogram::vertical(&chart)
            .style(RED.mix(0.5).filled())
            .data(data.iter().map(|x: &f64| ((*x * 10.) as u32, 1))),
    )?;*/

    // To avoid the IO failure being ignored silently, we manually call the present function
    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    println!("Result has been saved to {}", name);

    Ok(())
}

pub fn plot_channels(data: &[&[(f64, f64)]]) -> Result<(), Box<dyn std::error::Error>> {
    let root = SVGBackend::new("out/channels.svg", (800, 600)).into_drawing_area();

    root.fill(&WHITE)?;
    let minx = data[0].first().expect("No data").0;
    let maxx = data[0].last().expect("No data").0;
    let miny = data[0].first().expect("No data").1;
    let maxy = data[0].last().expect("No data").1;

    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(35)
        .y_label_area_size(80)
        .margin(5)
        .caption("CDF for the different color channels", ("sans-serif", 40.0))
        .build_cartesian_2d(minx..maxx, miny..maxy)?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .bold_line_style(WHITE.mix(0.3))
        .y_desc("Encoding")
        .x_desc("Input")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    for (i, data) in data.iter().enumerate() {
        let color = [RED, GREEN, BLUE, YELLOW, MAGENTA][i];
        let caption = ["Red", "Green", "Blue", "3", "4", "5"][i];
        chart
            .draw_series(LineSeries::new(data.iter().map(|(x, y)| (*x, *y)), &color))?
            .label(caption)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], color));
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    // To avoid the IO failure being ignored silently, we manually call the present function
    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    println!("Result has been saved to {}", OUT_FILE_NAME);

    Ok(())
}

pub fn plot_errors(
    data: &[Vec<f64>],
    names: &[String],
    color: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let name = format!("out/errors_{}.svg", color);
    let root = SVGBackend::new(name.as_str(), (800, 600)).into_drawing_area();

    root.fill(&WHITE)?;
    let minx = 0.;
    let maxx = data[0].len() as f64;
    let miny = 0.;
    let maxy = data
        .iter()
        .map(|x| x.iter().cloned().fold(0., f64::max))
        .fold(0., f64::max);

    let caption = format!("Quantization error functions for the {} channel", color);
    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(35)
        .y_label_area_size(80)
        .margin(5)
        .caption(caption.as_str(), ("sans-serif", 40.0))
        .build_cartesian_2d(minx..maxx, (miny..maxy))?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .bold_line_style(WHITE.mix(0.3))
        .y_desc("Relative Error")
        .x_desc("Number of bits")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    for (i, data) in data.iter().enumerate() {
        let color = Palette99::pick(i + 3);
        chart
            .draw_series(LineSeries::new(
                data.iter().cloned().enumerate().map(|(x, y)| (x as f64, y)),
                &color,
            ))?
            .label(&names[i])
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &color));
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    // To avoid the IO failure being ignored silently, we manually call the present function
    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    println!("Result has been saved to {}", name);

    Ok(())
}

pub fn plot_errors_with_bits(
    data: &[Vec<f64>],
    bits: &[&[usize]],
    names: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    let root = SVGBackend::new("out/combined_error.svg", (1200, 800)).into_drawing_area();

    root.fill(&WHITE)?;
    let maxx = data[0].len() as f64;
    let maxy = data
        .iter()
        .map(|x| x.iter().cloned().fold(0., f64::max))
        .fold(0., f64::max);

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .caption("Error functions and bit allocation", ("sans-serif", 40.0))
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .set_label_area_size(LabelAreaPosition::Right, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .set_label_area_size(LabelAreaPosition::Top, 40)
        .build_cartesian_2d(0f64..maxx, 0f64..maxy)?
        .set_secondary_coord(0f32..24f32, 0..16u32);

    /*let mut chart = ChartBuilder::on(&root)
            .x_label_area_size(35)
            .y_label_area_size(80)
            .margin(5)
            .caption("Error functions and bit allocation", ("sans-serif", 40.0))
            .build_cartesian_2d(0..(maxx as u32), 0f64..maxy)?
            .set_secondary_coord(0..32usize, 0..32usize);
    */
    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .x_labels(12)
        //.bold_line_style(WHITE.mix(0.3))
        .max_light_lines(4)
        .y_desc("Relative Error")
        .x_desc("Number of bits")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    chart
        .configure_secondary_axes()
        .x_desc("Container size in bit")
        .x_labels(12)
        .y_desc("Number of bits allocated per channel")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;
    for (i, bits) in bits.iter().enumerate() {
        let color = [RED, GREEN, BLUE][i];
        chart
            .draw_secondary_series(bits.iter().cloned().enumerate().map(|(x, y)| {
                let y = y as u32;
                let x = x as u32;
                let color = color.mix(0.4);
                Rectangle::new(
                    [
                        (x as f32 + 0.125 + (1. / 5.) * i as f32, 0),
                        (x as f32 + 0.125 + (1. / 5.) * (i as f32 + 1.), y),
                    ],
                    color.filled(),
                )
            }))?
            .legend(move |(x, y)| Rectangle::new([(x - 5, y - 5), (x + 5, y + 5)], color.filled()))
            .label(format!("{} bits", names[i]));
    }
    for (i, data) in data.iter().enumerate() {
        let color = [RED, GREEN, BLUE][i];
        chart
            .draw_series(
                LineSeries::new(
                    data.iter().cloned().enumerate().map(|(x, y)| (x as f64, y)),
                    color.filled(),
                )
                .point_size(3),
            )?
            .label(&names[i])
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    // To avoid the IO failure being ignored silently, we manually call the present function
    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    println!("Result has been saved to {}", OUT_FILE_NAME);

    Ok(())
}
