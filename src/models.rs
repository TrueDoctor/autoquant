use crate::{decode, encode, CreateFitFn};

use super::FitFn;

use argmin::core::observers::{ObserverMode, SlogLogger};
use argmin::solver::neldermead::NelderMead;

use super::Dist;

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
            let [(x, y), (_, yn)]  = window else { unreachable!() } ;
            let log = T::new(param.clone());

            let encoded = encode(*x, &log, self.1);
            let decoded = decode(encoded, &log, self.1);

            let error = (decoded / x.max(f64::EPSILON)).abs();
            let error = ((error * (yn - y)) - (yn - y)).abs();
            assert!(
                error.is_finite(),
                "encontered non-finite error {} for encoded value {}, decoded value {}, x: {} and parameters {:?}",
                error,
                encoded,
                decoded,
                encoded as f64 / self.1 as f64,
                param
            );
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
            .configure(|state| state.max_iters(5000))
            .add_observer(SlogLogger::term(), ObserverMode::Every(200));
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
        let b = self.0[1].max(0.01);
        let c = self.0[2];
        let d = self.0[3];
        ((x + a) * d).max(0.).ln().max(0.) * b + c
    }
    fn inverse(&self, x: f64) -> f64 {
        let a = self.0[0];
        let b = self.0[1].max(0.01);
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

#[derive(Debug)]
pub struct OptimizedPow(Vec<f64>);

impl OptimizedPow {
    pub fn new(dist: Dist, quantization: u64) -> Self {
        let params = vec![
            vec![0.0, 1.0, 0.0, 1., 1.0],
            vec![0.1, 1.0, 0.0, 1., 1.0],
            vec![0.0, 1.1, 0.0, 1., 1.0],
            vec![0.0, 1.0, 0.1, 1., 1.0],
            vec![0.0, 1.0, 0.0, 1.1, 1.0],
            vec![0.0, 1.0, 0.0, 1., 0.5],
        ];
        let nm: NelderMead<Vec<f64>, f64> = NelderMead::new(params);

        //log::debug!("dist: {:?}", dist);
        //println!("dist: {:?}", dist);
        let fit: Fit<OptimizedPow> = Fit::new(dist, quantization);
        let executor = argmin::core::Executor::new(fit, nm)
            .configure(|state| state.max_iters(5000))
            .add_observer(SlogLogger::term(), ObserverMode::Every(200));
        let res = executor.run().unwrap();
        let params = res.state().best_param.clone().unwrap();
        println!("Result: {:?}", res.state().best_cost);
        println!("params: {:?}", params);
        log::debug!("params: {:?}", params);
        Self(params)
    }
}

impl FitFn for OptimizedPow {
    fn function(&self, x: f64) -> f64 {
        let a = self.0[0];
        let b = self.0[1];
        let c = self.0[2];
        let d = self.0[3];
        let e = self.0[4];
        ((x - a) * d).max(0.).powf(e) * b + c
    }
    fn inverse(&self, x: f64) -> f64 {
        let a = self.0[0];
        let b = self.0[1];
        let c = self.0[2];
        let d = self.0[3];
        let e = self.0[4];
        (((x - c) / b).abs().powf(1. / e) + a * d) / d
    }
    fn name(&self) -> &str {
        "powf"
    }
}

impl CreateFitFn for OptimizedPow {
    fn new(params: Vec<f64>) -> Self {
        Self(params)
    }
}

#[derive(Debug)]
pub struct OptimizedLin(Vec<f64>);

impl OptimizedLin {
    pub fn new(dist: Dist, quantization: u64) -> Self {
        let params = vec![vec![1.0, 0.0], vec![1.1, 0.0], vec![1.0, 0.1]];
        let nm: NelderMead<Vec<f64>, f64> = NelderMead::new(params);

        //log::debug!("dist: {:?}", dist);
        //println!("dist: {:?}", dist);
        let fit: Fit<OptimizedLin> = Fit::new(dist, quantization);
        let executor = argmin::core::Executor::new(fit, nm)
            .configure(|state| state.max_iters(500))
            .add_observer(SlogLogger::term(), ObserverMode::Every(200));
        let res = executor.run().unwrap();
        let params = res.state().best_param.clone().unwrap();
        println!("Result: {:?}", res.state().best_cost);
        println!("params: {:?}", params);
        log::debug!("params: {:?}", params);
        Self(params)
    }
}

impl FitFn for OptimizedLin {
    fn function(&self, x: f64) -> f64 {
        let a = self.0[0];
        let b = self.0[1];
        a * x + b
    }
    fn inverse(&self, x: f64) -> f64 {
        let a = self.0[0];
        let b = self.0[1];
        (x - b) / a
    }
    fn name(&self) -> &str {
        "linear"
    }
}

impl CreateFitFn for OptimizedLin {
    fn new(params: Vec<f64>) -> Self {
        Self(params)
    }
}
