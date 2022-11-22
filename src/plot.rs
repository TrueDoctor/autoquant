use linregress::RegressionModel;
use plotters::prelude::*;

use crate::FitFn;
const OUT_FILE_NAME: &str = "histogram.svg";

pub fn plot_histogram(
    data: &[(f64, f64)],
    fits: &[(&dyn FitFn, &RegressionModel)],
) -> Result<(), Box<dyn std::error::Error>> {
    let root = SVGBackend::new(OUT_FILE_NAME, (1920, 1080)).into_drawing_area();

    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(35)
        .y_label_area_size(80)
        .margin(5)
        .caption("Integrated normal distribution", ("sans-serif", 50.0))
        .build_cartesian_2d(-5f64..5f64, 0f64..1f64)?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .bold_line_style(WHITE.mix(0.3))
        .y_desc("Encoding")
        .x_desc("Input")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    chart
        .draw_series(LineSeries::new(data.iter().cloned(), RED))?
        .label("CDF")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    for (i, (fit, model)) in fits.iter().enumerate() {
        let color = Palette99::pick(i + 1);
        let fit_data = data
            .iter()
            .map(|&(x, _)| (x, fit.function(model.predict([("X", vec![x])]).unwrap()[0])))
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
