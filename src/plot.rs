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
