use crate::CreateFitFn;

use super::FitFn;

use argmin::core::observers::{ObserverMode, SlogLogger};
use argmin::solver::neldermead::NelderMead;
use varpro::solvers::levmar::LevMarProblemBuilder;
use varpro::solvers::levmar::LevMarSolver;

use nalgebra::DVector;

use super::Dist;

use varpro::prelude::*;

pub struct PowerTwo(Dist, u64);
pub struct Log(Dist, u64);

pub struct VarPro<T> {
    pub(crate) solved_problem: varpro::solvers::levmar::LevMarProblem<'static, f64>,
    _phantom: std::marker::PhantomData<T>,
}

impl VarPro<PowerTwo> {
    pub fn new(dist: Dist) -> Self {
        let model = SeparableModelBuilder::<f64>::new(&["a"])
            .function(&["a"], |x: &DVector<f64>, a: f64| {
                x.map(|x| (x - a).powi(2))
            })
            .partial_deriv("a", |x: &DVector<f64>, a: f64| x.map(|x| 2. * (a - x)))
            // add the constant as a vector of ones as an invariant function
            .invariant_function(|x| DVector::from_element(x.len(), 1.))
            .build()
            .unwrap();

        let model = Box::leak(Box::new(model));
        let x = DVector::from_iterator(dist.len(), dist.iter().map(|&(x, _)| x));
        let y = DVector::from_iterator(dist.len(), dist.iter().map(|&(_, y)| y));
        let problem = LevMarProblemBuilder::new()
            .model(model)
            .x(x)
            .y(y)
            .initial_guess(&[1.])
            .build()
            .expect("Building valid problem should not panic");
        // 4. Solve using the fitting problem
        let (solved_problem, report) = LevMarSolver::new().minimize(problem);
        assert!(report.termination.was_successful());

        Self {
            solved_problem,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl FitFn for VarPro<PowerTwo> {
    fn function(&self, x: f64) -> f64 {
        let a = self.solved_problem.params()[0];
        let b = self.solved_problem.linear_coefficients().unwrap()[0];
        let c = self.solved_problem.linear_coefficients().unwrap()[1];
        (x - a).powi(2) * b + c
    }
    fn inverse(&self, x: f64) -> f64 {
        let a = self.solved_problem.params()[0];
        let b = self.solved_problem.linear_coefficients().unwrap()[0];
        let c = self.solved_problem.linear_coefficients().unwrap()[1];
        a - (-1. * b * (c - x)).sqrt() / b
    }
    fn name(&self) -> &str {
        "pow_2"
    }
}

impl VarPro<Log> {
    pub(crate) fn new(dist: Dist) -> Self {
        let model = SeparableModelBuilder::<f64>::new(&["a"])
            .function(&["a"], |x: &DVector<f64>, a: f64| x.map(|x| (x + a).ln()))
            .partial_deriv("a", |x: &DVector<f64>, a: f64| x.map(|x| -1. / (x + a)))
            // add the constant as a vector of ones as an invariant function
            .invariant_function(|x| DVector::from_element(x.len(), 1.))
            .build()
            .unwrap();

        let model = Box::leak(Box::new(model));
        let x = DVector::from_iterator(dist.len(), dist.iter().map(|&(x, _)| x));
        let y = DVector::from_iterator(dist.len(), dist.iter().map(|&(_, y)| y));
        let problem = LevMarProblemBuilder::new()
            .model(model)
            .x(x)
            .y(y)
            .initial_guess(&[0.])
            //.epsilon(0.00000001)
            .build()
            .expect("Building valid problem should not panic");
        // 4. Solve using the fitting problem
        let (solved_problem, report) = LevMarSolver::new().minimize(problem);
        assert!(report.termination.was_successful());

        Self {
            solved_problem,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl FitFn for VarPro<Log> {
    fn function(&self, x: f64) -> f64 {
        let a = self.solved_problem.params()[0];
        let b = self.solved_problem.linear_coefficients().unwrap()[0];
        let c = self.solved_problem.linear_coefficients().unwrap()[1];
        (x + a).ln() * b + c
    }
    fn inverse(&self, x: f64) -> f64 {
        let a = self.solved_problem.params()[0];
        let b = self.solved_problem.linear_coefficients().unwrap()[0];
        let c = self.solved_problem.linear_coefficients().unwrap()[1];
        -(-c / b).exp() * ((a) * (c / b).exp() - (x / b).exp())
    }
    fn name(&self) -> &str {
        "log"
    }
}

use argmin::core::{CostFunction, Error};
struct Fit<T: CreateFitFn>(Dist, u64, std::marker::PhantomData<T>);
impl<T: CreateFitFn> Fit<T> {
    pub fn new(dist: Dist, quantization: u64) -> Self {
        Self(dist, quantization, std::marker::PhantomData)
    }
}

impl<T: CreateFitFn> CostFunction for Fit<T> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        //println!("{:?}", param);
        let iter = self.0.windows(2).map(|window| {
            let [(_, y), (x, yn)]  = window else { unreachable!() } ;
            let log = T::new(param.clone());

            let y_hat = log.function(*x);
            assert!(!y_hat.is_nan(), "{:?}", param);
            //println!("x: {}, y: {}, y_hat: {}", x, y, y_hat);
            let clamped = y_hat.clamp(0., 1.);
            let quantized = (clamped * self.1 as f64) as u64;
            let clamped = quantized as f64 / self.1 as f64;
            let inverse = log.inverse(clamped);
            let error = (inverse / x).abs();
            let error = ((error * (yn - y)) - (yn - y)).abs();
            //dbg!(error);
            assert!(error.is_finite(), "{:?}", param);
            error
        });
        let sum = crate::sum::Sum::from_iter(iter).sum();
        //println!("sum: {}", sum);
        Ok(sum)
    }
}

#[derive(Debug)]
pub struct OptimizedLog(Vec<f64>);

impl OptimizedLog {
    pub fn new(dist: Dist, quantization: u64) -> Self {
        let params = vec![
            vec![0.01, 0.3, 0.5, 4.],
            vec![0.009, 0.3, 0.5, 4.],
            vec![0.01, 0.4, 0.5, 4.],
            vec![0.01, 0.3, 0.6, 4.],
            vec![0.01, 0.3, 0.5, 5.],
        ];
        let nm: NelderMead<Vec<f64>, f64> = NelderMead::new(params);

        //log::debug!("dist: {:?}", dist);
        //println!("dist: {:?}", dist);
        let fit: Fit<OptimizedLog> = Fit::new(dist, quantization);
        let executor = argmin::core::Executor::new(fit, nm)
            .configure(|state| state.max_iters(500))
            .add_observer(SlogLogger::term(), ObserverMode::Every(100));
        let res = executor.run().unwrap();
        let params = res.state().best_param.clone().unwrap();
        println!("Result: {:?}", res.state().best_cost);
        println!("params: {:?}", params);
        log::debug!("params: {:?}", params);
        Self(params)
    }

    pub fn parameters(&self) -> &[f64] {
        &self.0
    }
}

impl FitFn for OptimizedLog {
    fn function(&self, x: f64) -> f64 {
        let a = self.0[0];
        let b = self.0[1];
        let c = self.0[2];
        let d = self.0[3];
        ((x + a) * d).abs().ln() * b + c
    }
    fn inverse(&self, x: f64) -> f64 {
        let a = self.0[0];
        let b = self.0[1];
        let c = self.0[2];
        let d = self.0[3];
        -(-c / b).exp() * (a * d * (c / b).exp() - (x / b).exp()) / d
    }
    fn name(&self) -> &str {
        "log"
    }
}

impl CreateFitFn for OptimizedLog {
    fn new(params: Vec<f64>) -> Self {
        Self(params)
    }
}
