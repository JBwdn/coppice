use crate::criteria::information_gain_criterion;
use crate::node::Node;
use crate::splitting::find_best_split;
use crate::splitting::split;

pub fn grow_tree(x: &Vec<Vec<f32>>, y: &Vec<u32>, max_depth: u32) -> Node {
    grow_tree_(x, y, max_depth, 0)
}

fn grow_tree_(x: &Vec<Vec<f32>>, y: &Vec<u32>, max_depth: u32, depth: u32) -> Node {
    let n_labels = y.iter().collect::<std::collections::HashSet<&u32>>().len();
    let min_samples_split = 2;
    let most_common_label = y
        .iter()
        .max_by_key(|&x| y.iter().filter(|&y| y == x).count())
        .unwrap();

    if n_labels == 1 || x.len() <= min_samples_split || depth == max_depth {
        return Node {
            feature_index: None,
            threshold: None,
            left: None,
            right: None,
            is_leaf: true,
            class_label: Some(*most_common_label),
        };
    }

    let (threshold, feature_idx) = find_best_split(x, y, information_gain_criterion);
    let (left_x, left_y, right_x, right_y) = split(x, y, feature_idx as usize, threshold);
    let left = grow_tree_(&left_x, &left_y, max_depth, depth + 1);
    let right = grow_tree_(&right_x, &right_y, max_depth, depth + 1);

    Node {
        feature_index: Some(feature_idx),
        threshold: Some(threshold),
        left: Some(Box::new(left)),
        right: Some(Box::new(right)),
        is_leaf: false,
        class_label: None,
    }
}
