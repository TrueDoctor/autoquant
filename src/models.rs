use super::FitFn;

use varpro::solvers::levmar::LevMarProblemBuilder;
use varpro::solvers::levmar::LevMarSolver;

use nalgebra::DVector;

use super::Dist;

use varpro::prelude::*;

pub struct PowerTwo;
pub struct Log;

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
            .partial_deriv("a", |x: &DVector<f64>, a: f64| x.map(|x| (a - x)))
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
        b + (-1. * a * (c - x)).sqrt() / a
    }
    fn name(&self) -> &str {
        "pow_2"
    }
}

impl VarPro<Log> {
    pub(crate) fn new(dist: Dist) -> Self {
        let model = SeparableModelBuilder::<f64>::new(&["a"])
            .function(&["a"], |x: &DVector<f64>, a: f64| x.map(|x| (x - a).ln()))
            .partial_deriv("a", |x: &DVector<f64>, a: f64| x.map(|x| 1. / (x - a)))
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
            .epsilon(0.00000001)
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
        (x - a).ln() * b + c
    }
    fn inverse(&self, x: f64) -> f64 {
        let a = self.solved_problem.params()[0];
        let b = self.solved_problem.linear_coefficients().unwrap()[0];
        let c = self.solved_problem.linear_coefficients().unwrap()[1];
        (-c / b).exp() * ((a) * (c / b).exp() + (x / b).exp())
    }
    fn name(&self) -> &str {
        "log"
    }
}
