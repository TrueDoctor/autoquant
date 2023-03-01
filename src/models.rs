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
        println!("{:?}", param);
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
        println!("sum: {}", sum);
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
            .configure(|state| state.max_iters(1000))
            .add_observer(SlogLogger::term(), ObserverMode::Always);
        let res = executor.run().unwrap();
        let params = res.state().best_param.clone().unwrap();
        println!("Result: {:?}", res.state().best_cost);
        println!("params: {:?}", params);
        log::debug!("params: {:?}", params);
        Self(params)
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

mod executor {
    use argmin::core::{
        DeserializeOwnedAlias, Error, OptimizationResult, Problem, SerializeAlias, Solver, State,
        TerminationReason,
    };
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    /// Solves an optimization problem with a solver
    pub struct Executor<O, S, I> {
        /// Solver
        solver: S,
        /// Problem
        problem: Problem<O>,
        /// State
        state: Option<I>,
    }

    impl<O, S, I> Executor<O, S, I>
    where
        S: Solver<O, I>,
        I: State + SerializeAlias + DeserializeOwnedAlias,
    {
        /// Constructs an `Executor` from a user defined problem and a solver.
        ///
        /// # Example
        ///
        /// ```
        /// # use argmin::core::Executor;
        /// # use argmin::core::test_utils::{TestSolver, TestProblem};
        /// #
        /// # type Rosenbrock = TestProblem;
        /// # type Newton = TestSolver;
        /// #
        /// // Construct an instance of the desired solver
        /// let solver = Newton::new();
        ///
        /// // `Rosenbrock` implements `CostFunction` and `Gradient` as required by the
        /// // `SteepestDescent` solver
        /// let problem = Rosenbrock {};
        ///
        /// // Create instance of `Executor` with `problem` and `solver`
        /// let executor = Executor::new(problem, solver);
        /// ```
        pub fn new(problem: O, solver: S) -> Self {
            let state = Some(I::new());
            Executor {
                solver,
                problem: Problem::new(problem),
                state,
            }
        }

        /// This method gives mutable access to the internal state of the solver. This allows for
        /// initializing the state before running the `Executor`. The options for initialization depend
        /// on the type of state used by the chosen solver. Common types of state are
        /// [`IterState`](`crate::core::IterState`),
        /// [`PopulationState`](`crate::core::PopulationState`), and
        /// [`LinearProgramState`](`crate::core::LinearProgramState`). Please see the documentation of
        /// the desired solver for information about which state is used.
        ///
        /// # Example
        ///
        /// ```
        /// # use argmin::core::Executor;
        /// # use argmin::core::test_utils::{TestSolver, TestProblem};
        /// #
        /// #  let solver = TestSolver::new();
        /// #  let problem = TestProblem::new();
        /// #  let init_param = vec![1.0f64, 0.0];
        /// #
        /// // Create instance of `Executor` with `problem` and `solver`
        /// let executor = Executor::new(problem, solver)
        ///     // Configure and initialize internal state.
        ///     .configure(|state| state.param(init_param).max_iters(10));
        /// ```
        #[must_use]
        pub fn configure<F: FnOnce(I) -> I>(mut self, init: F) -> Self {
            let state = self.state.take().unwrap();
            let state = init(state);
            self.state = Some(state);
            self
        }

        /// Runs the executor by applying the solver to the optimization problem.
        ///
        /// # Example
        ///
        /// ```
        /// # use argmin::core::{Error, Executor};
        /// # use argmin::core::test_utils::{TestSolver, TestProblem};
        /// #
        /// # fn main() -> Result<(), Error> {
        /// # let solver = TestSolver::new();
        /// # let problem = TestProblem::new();
        /// #
        /// # let init_param = vec![1.0f64, 0.0];
        /// #
        /// // Create instance of `Executor` with `problem` and `solver`
        /// let result = Executor::new(problem, solver)
        ///     // Configure and initialize internal state.
        ///     .configure(|state| state.param(init_param).max_iters(100))
        /// #   .configure(|state| state.max_iters(1))
        ///     // Execute solver
        ///     .run()?;
        /// # Ok(())
        /// # }
        /// ```
        pub fn run(mut self) -> Result<OptimizationResult<O, S, I>, Error> {
            let state = self.state.take().unwrap();

            let running = Arc::new(AtomicBool::new(true));

            // Only call `init` of `solver` if the current iteration number is 0. This avoids that
            // `init` is called when starting from a checkpoint (because `init` could change the state
            // of the `solver`, which would overwrite the state restored from the checkpoint).
            let mut state = if state.get_iter() == 0 {
                let (mut state, kv) = self.solver.init(&mut self.problem, state)?;
                state.update();

                state.func_counts(&self.problem);
                state
            } else {
                state
            };

            loop {
                // check first if it has already terminated
                // This should probably be solved better.
                // First, check if it isn't already terminated. If it isn't, evaluate the
                // stopping criteria. If `self.terminate()` is called without the checking
                // whether it has terminated already, then it may overwrite a termination set
                // within `next_iter()`!
                state = if !state.terminated() {
                    let term = self.solver.terminate_internal(&state);
                    state.terminate_with(term)
                } else {
                    state
                };
                // Now check once more if the algorithm has terminated. If yes, then break.
                if state.terminated() {
                    break;
                }

                let (state_t, _kv) = self.solver.next_iter(&mut self.problem, state)?;
                state = state_t;

                state.func_counts(&self.problem);

                state.update();

                // increment iteration number
                state.increment_iter();

                // Check if termination occurred inside next_iter()
                if state.terminated() {
                    break;
                }
            }

            // in case it stopped prematurely and `termination_reason` is still `NotTerminated`,
            // someone must have pulled the handbrake
            if state.get_iter() < state.get_max_iters() && !state.terminated() {
                state = state.terminate_with(TerminationReason::Aborted);
            }
            Ok(OptimizationResult::new(self.problem, self.solver, state))
        }
    }
}
