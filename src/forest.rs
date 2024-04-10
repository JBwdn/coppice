use crate::tree::{grow_tree, predict, Node};
use rand::distributions::{Distribution, Uniform};

pub struct Forest {
    pub trees: Vec<Node>,
}

fn mode(v: &Vec<u32>) -> u32 {
    let mut counts = std::collections::HashMap::new();
    for &item in v {
        let count = counts.entry(item).or_insert(0);
        *count += 1;
    }
    let mut max = 0;
    let mut mode = 0;
    for (k, v) in counts {
        if v > max {
            max = v;
            mode = k;
        }
    }
    mode
}

impl Forest {
    pub fn new() -> Forest {
        Forest { trees: Vec::new() }
    }

    pub fn add_tree(&mut self, tree: Node) {
        self.trees.push(tree);
    }

    pub fn predict(&self, x: &Vec<Vec<f32>>) -> Vec<u32> {
        let mut results = Vec::new();
        for (i, tree) in self.trees.iter().enumerate() {
            let y_hat = predict(x, tree);
            if i == 0 {
                for y in y_hat {
                    results.push(vec![y]);
                }
            } else {
                for (j, y) in y_hat.iter().enumerate() {
                    results[j].push(*y);
                }
            }
        }
        results.iter().map(|row| mode(row)).collect()
    }
}

fn bootstrap_sample(x: &Vec<Vec<f32>>, y: &Vec<u32>) -> (Vec<Vec<f32>>, Vec<u32>) {
    let uniform = Uniform::from(0..x.len());
    let mut rng = rand::thread_rng();
    let mut idxs = Vec::new();
    for _ in 0..(x.len() as f32) as usize {
        idxs.push(uniform.sample(&mut rng));
    }

    let mut x_sample = Vec::new();
    let mut y_sample = Vec::new();

    for i in idxs {
        x_sample.push(x[i].clone());
        y_sample.push(y[i]);
    }
    (x_sample, y_sample)
}

pub fn grow_forest(x: &Vec<Vec<f32>>, y: &Vec<u32>, max_depth: u32, n: u32) -> Forest {
    let mut forest = Forest::new();
    let mut handles = vec![];
    for _ in 0..n {
        let x = x.clone();
        let y = y.clone();
        let handle = std::thread::spawn(move || {
            let (x_boostrapped, y_boostrapped) = bootstrap_sample(&x, &y);
            grow_tree(&x_boostrapped, &y_boostrapped, max_depth)
        });
        handles.push(handle);
    }
    for handle in handles {
        let tree = handle.join().unwrap();
        forest.add_tree(tree);
    }
    forest
}
