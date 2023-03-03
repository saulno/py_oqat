use pyo3::prelude::*;

#[derive(FromPyObject)]
pub struct ACOConfig {
    pub algorithm: String,
    pub cycles: usize,
    pub ants: usize,
    pub alpha: f64,
    pub rho: f64,
    pub tau_max: f64,
    pub tau_min: f64,
}
