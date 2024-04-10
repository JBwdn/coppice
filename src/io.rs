use csv::Reader;
use numpy::ndarray::Dim;
use numpy::PyArray;

pub fn load_data_x(path: &str) -> Vec<Vec<f32>> {
    println!("Loading x from {:}...", path);

    let mut x = Vec::new();
    let mut rdr = Reader::from_path(path).unwrap();

    for result in rdr.records() {
        let record = result.unwrap();
        let mut row = Vec::new();
        for val in record.iter() {
            row.push(val.parse::<f32>().unwrap());
        }
        x.push(row);
    }
    println!("Shape of x matrix ({}, {})", x.len(), x[0].len());
    x
}

pub fn load_data_y(path: &str) -> Vec<u32> {
    println!("Loading y from {:}...", path);
    let mut y = Vec::new();
    let mut rdr = Reader::from_path(path).unwrap();

    for result in rdr.records() {
        let record = result.unwrap();
        for val in record.iter() {
            y.push(val.parse::<u32>().unwrap());
        }
    }
    println!("Shape of y ({},)", y.len());
    y
}

pub fn numpy_to_x(py_x: &PyArray<f32, Dim<[usize; 2]>>) -> Vec<Vec<f32>> {
    let mut x = Vec::new();
    unsafe {
        let array_x = py_x.as_array();
        for i in 0..py_x.shape()[0] {
            let mut row = Vec::new();
            for j in 0..py_x.shape()[1] {
                row.push(array_x[[i, j]] as f32);
            }
            x.push(row);
        }
    }
    println!("Shape of x matrix ({}, {})", x.len(), x[0].len());
    x
}

pub fn numpy_to_y(py_y: &PyArray<u32, Dim<[usize; 1]>>) -> Vec<u32> {
    let mut y = Vec::new();
    unsafe {
        let array_y = py_y.as_array();
        for i in 0..py_y.shape()[0] {
            y.push(array_y[[i]] as u32);
        }
    }
    println!("Shape of y ({},)", y.len());
    y
}
