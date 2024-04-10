use crate::criteria::information_gain_criterion;
use crate::splitting::find_best_split;
use crate::splitting::split;
use std::fmt;

pub struct Node {
    pub feature_index: Option<u32>,
    pub threshold: Option<f32>,
    pub left: Option<Box<Node>>,
    pub right: Option<Box<Node>>,
    pub is_leaf: bool,
    pub class_label: Option<u32>,
}
impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut s = String::new();
        s.push_str("{");
        if self.is_leaf {
            s.push_str(&format!(
                "LeafNode {{class_label: {:?}}}",
                self.class_label.unwrap()
            ));
        } else {
            s.push_str(&format!("BranchNode {{"));
            s.push_str(&format!(
                "feature_index: {:?}, ",
                self.feature_index.unwrap()
            ));
            s.push_str(&format!("threshold: {:?}, ", self.threshold.unwrap()));
            s.push_str(&format!("left: {}, ", self.left.as_ref().unwrap()));
            s.push_str(&format!("right: {}", self.right.as_ref().unwrap()));
        }
        s.push_str("},");
        write!(f, "{}", s)
    }
}

fn predict_(x: &Vec<f32>, node: &Node) -> u32 {
    if node.is_leaf {
        return node.class_label.unwrap();
    }
    let feature_index = node.feature_index.unwrap() as usize;
    let threshold = node.threshold.unwrap();
    if x[feature_index] < threshold {
        return predict_(x, node.left.as_ref().unwrap());
    }
    predict_(x, node.right.as_ref().unwrap())
}

pub fn predict(x: &Vec<Vec<f32>>, node: &Node) -> Vec<u32> {
    let mut y = Vec::new();
    for row in x {
        y.push(predict_(row, node));
    }
    y
}

pub fn depth(node: &Node) -> u32 {
    if node.is_leaf {
        return 0;
    }
    let left_depth = depth(node.left.as_ref().unwrap());
    let right_depth = depth(node.right.as_ref().unwrap());
    1 + std::cmp::max(left_depth, right_depth)
}

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
