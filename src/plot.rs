use plotters::prelude::*;

use crate::FitFn;
const OUT_FILE_NAME: &str = "histogram.svg";

pub fn plot_histogram(
    data: &[(f64, f64)],
    fits: &[&dyn FitFn],
) -> Result<(), Box<dyn std::error::Error>> {
    let root = SVGBackend::new(OUT_FILE_NAME, (800, 600)).into_drawing_area();

    root.fill(&WHITE)?;
    let minx = data.first().expect("No data").0;
    let maxx = data.last().expect("No data").0;
    let miny = data.first().expect("No data").1;
    let maxy = data.last().expect("No data").1;

    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(35)
        .y_label_area_size(80)
        .margin(5)
        .caption("CDF approximation", ("sans-serif", 40.0))
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
    println!("Result has been saved to {}", OUT_FILE_NAME);

    Ok(())
}

pub fn plot_channels(data: &[&[(f64, f64)]]) -> Result<(), Box<dyn std::error::Error>> {
    let root = SVGBackend::new(OUT_FILE_NAME, (800, 600)).into_drawing_area();

    root.fill(&WHITE)?;
    let minx = data[0].first().expect("No data").0;
    let maxx = data[0].last().expect("No data").0;
    let miny = data[0].first().expect("No data").1;
    let maxy = data[0].last().expect("No data").1;

    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(35)
        .y_label_area_size(80)
        .margin(5)
        .caption("CDF approximation", ("sans-serif", 40.0))
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
        let color = [RED, GREEN, BLUE][i];
        let caption = ["Red", "Green", "Blue"][i];
        chart
            .draw_series(LineSeries::new(data.iter().map(|(x, y)| (*x, *y)), &color))?
            .label(caption)
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

pub fn plot_errors(data: &[Vec<f64>], names: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let root = SVGBackend::new(OUT_FILE_NAME, (800, 600)).into_drawing_area();

    root.fill(&WHITE)?;
    let minx = 0.;
    let maxx = data[0].len() as f64;
    let miny = 0.;
    let maxy = data
        .iter()
        .map(|x| x.iter().cloned().fold(0., f64::max))
        .fold(0., f64::max);

    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(35)
        .y_label_area_size(80)
        .margin(5)
        .caption("Error functions", ("sans-serif", 40.0))
        .build_cartesian_2d(minx..maxx, (miny..maxy).log_scale())?;

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
    println!("Result has been saved to {}", OUT_FILE_NAME);

    Ok(())
}

pub fn plot_errors_with_bits(
    data: &[Vec<f64>],
    bits: &[&[usize]],
    names: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    let root = SVGBackend::new(OUT_FILE_NAME, (1200, 800)).into_drawing_area();

    root.fill(&WHITE)?;
    let minx = 0.;
    let maxx = data[0].len() as f64;
    let miny = 0.;
    let maxy = data
        .iter()
        .map(|x| x.iter().cloned().fold(0., f64::max))
        .fold(0., f64::max);

    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(35)
        .y_label_area_size(80)
        .margin(5)
        .caption("Error functions and bit allocation", ("sans-serif", 40.0))
        .build_cartesian_2d(minx..maxx, (miny..maxy).log_scale())?
        .set_secondary_coord(0..32usize, 0..32usize);

    chart
        .configure_mesh()
        .disable_x_mesh()
        .bold_line_style(WHITE.mix(0.3))
        .y_desc("Relative Error")
        .x_desc("Number of bits")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    chart
        .configure_secondary_axes()
        .y_desc("Number of bits allocated per channel")
        .draw()?;

    for (i, (data, bits)) in data.iter().zip(bits).enumerate() {
        let color = [RED, GREEN, BLUE][i];
        chart
            .draw_series(LineSeries::new(
                data.iter().cloned().enumerate().map(|(x, y)| (x as f64, y)),
                &color,
            ))?
            .label(&names[i])
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &color));
        let color = [RED, GREEN, BLUE][i];
        chart
            .draw_secondary_series(LineSeries::new(
                bits.iter().cloned().enumerate().map(|(x, y)| (x, y)),
                &color,
            ))?
            .label(format!("{} bits", names[i]));
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
