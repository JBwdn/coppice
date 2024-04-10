#![allow(dead_code)]
#![allow(unused_imports)]

use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
mod criteria;
mod forest;
mod io;
mod metrics;
mod splitting;
mod tree;

#[pyfunction]
fn train_tree_np(x_py: &PyArray2<f32>, y_py: &PyArray1<u32>, max_depth: u32) -> () {
    let x = io::numpy_to_x(x_py);
    let y = io::numpy_to_y(y_py);

    let train_limit = (x.len() as f64 * 0.9) as usize;
    let x_train = x[..train_limit].to_vec();
    let y_train = y[..train_limit].to_vec();
    let x_test = x[train_limit..].to_vec();
    let y_test = y[train_limit..].to_vec();

    let tree = tree::grow_tree(&x_train, &y_train, max_depth);
    println!("Tree depth: {:?}", tree::depth(&tree));
    let y_hat = tree::predict(&x_test, &tree);
    metrics::classification_report(&y_hat, &y_test);
}

#[pyfunction]
fn train_forest_np(x_py: &PyArray2<f32>, y_py: &PyArray1<u32>, max_depth: u32, n_trees: u32) -> () {
    let x = io::numpy_to_x(x_py);
    let y = io::numpy_to_y(y_py);

    let train_limit = (x.len() as f64 * 0.9) as usize;
    let x_train = x[..train_limit].to_vec();
    let y_train = y[..train_limit].to_vec();
    let x_test = x[train_limit..].to_vec();
    let y_test = y[train_limit..].to_vec();

    let f = forest::grow_forest(&x_train, &y_train, max_depth, n_trees);
    let y_hat = f.predict(&x_test);
    metrics::classification_report(&y_hat, &y_test);
}

#[pymodule]
fn _coppice(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train_tree_np, m)?)?;
    m.add_function(wrap_pyfunction!(train_forest_np, m)?)?;
    Ok(())
}
