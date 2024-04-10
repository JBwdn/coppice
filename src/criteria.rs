use std::collections::HashMap;

fn entropy(y: Vec<u32>) -> f32 {
    let mut hist = HashMap::new();
    for i in &y {
        let count = hist.entry(i).or_insert(0);
        *count += 1;
    }
    let mut e = 0.0;
    let n = y.len() as f32;
    for (_, v) in hist {
        let p = v as f32 / n;
        e += p * p.log2();
    }
    -e
}

pub fn information_gain_criterion(x_col: Vec<f32>, y: Vec<u32>, threshold: f32) -> f32 {
    let mut y_left = Vec::new();
    let mut y_right = Vec::new();

    for (x, y) in x_col.iter().zip(y.iter()) {
        if *x < threshold {
            y_left.push(*y);
        } else {
            y_right.push(*y);
        }
    }

    let n = y.len() as f32;
    let n_left = y_left.len() as f32;
    let n_right = y_right.len() as f32;
    let e = entropy(y);
    let e_left = entropy(y_left);
    let e_right = entropy(y_right);
    e - (n_left / n) * e_left - (n_right / n) * e_right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy() {
        assert_eq!(entropy(vec![1, 1, 1, 1, 1, 1, 1, 1]), 0.0);
        assert_eq!(entropy(vec![1, 1, 1, 1, 1, 1, 1, 0]), 0.5435644431995964);
        assert_eq!(entropy(vec![1, 1, 1, 1, 1, 1, 1, 2]), 0.5435644431995964);
    }

    #[test]
    fn test_information_gain_criterion() {
        assert_eq!(
            information_gain_criterion(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 1, 1], 2.5),
            0.0
        );
        assert_eq!(
            information_gain_criterion(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 1, 1], 3.5),
            0.0
        );
    }
}
