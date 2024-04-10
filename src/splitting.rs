pub fn split(
    x: &Vec<Vec<f32>>,
    y: &Vec<u32>,
    feature_i: usize,
    thresh: f32,
) -> (Vec<Vec<f32>>, Vec<u32>, Vec<Vec<f32>>, Vec<u32>) {
    let mut left_x = Vec::new();
    let mut left_y = Vec::new();
    let mut right_x = Vec::new();
    let mut right_y = Vec::new();

    for i in 0..x.len() {
        if x[i][feature_i] < thresh {
            left_x.push(x[i].clone());
            left_y.push(y[i]);
        } else {
            right_x.push(x[i].clone());
            right_y.push(y[i]);
        }
    }
    (left_x, left_y, right_x, right_y)
}

pub fn find_best_split(
    x: &Vec<Vec<f32>>,
    y: &Vec<u32>,
    criterion: fn(Vec<f32>, Vec<u32>, f32) -> f32,
) -> (f32, u32) {
    let mut best_score = 0.0;
    let mut best_thresh = 0.0;
    let mut best_i = 0 as u32;

    let mut handles = vec![];
    for feature_i in 0..x[0].len() {
        let x_col = x.iter().map(|row| row[feature_i]).collect::<Vec<f32>>();
        let all_thresh = x_col.iter().cloned().collect::<Vec<f32>>();

        let x_col = x_col.clone();
        let y = y.clone();
        let criterion = criterion;

        let handle = std::thread::spawn(move || {
            let mut best_score = 0.0;
            let mut best_thresh = 0.0;
            for thresh in all_thresh {
                let score = criterion(x_col.clone(), y.clone(), thresh);
                if score > best_score {
                    best_score = score;
                    best_thresh = thresh;
                }
            }
            (best_score, best_thresh, feature_i)
        });
        handles.push(handle);
    }
    for handle in handles {
        let (score, thresh, feature_i) = handle.join().unwrap();
        if score > best_score {
            best_score = score;
            best_thresh = thresh;
            best_i = feature_i as u32;
        }
    }
    (best_thresh, best_i)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::criteria::information_gain_criterion;

    #[test]
    fn test_split() {
        let x = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let y = vec![1, 2, 3];
        let (left_x, left_y, right_x, right_y) = split(&x, &y, 0, 2.5);
        assert_eq!(left_x, vec![vec![1.0, 2.0]]);
        assert_eq!(left_y, vec![1]);
        assert_eq!(right_x, vec![vec![3.0, 4.0], vec![5.0, 6.0]]);
        assert_eq!(right_y, vec![2, 3]);
    }

    #[test]
    fn test_find_best_split() {
        let x = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let y = vec![1, 2, 3];
        let (best_thresh, best_i) = find_best_split(x, y, information_gain_criterion);
        assert_eq!(best_thresh, 3.0);
        assert_eq!(best_i, 0);
    }
}
