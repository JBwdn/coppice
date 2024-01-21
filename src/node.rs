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
        if self.is_leaf {
            s.push_str(&format!("LeafNode {{class_label: {:?}}}", self.class_label));
        } else {
            s.push_str(&format!("BranchNode {{"));
            s.push_str(&format!("feature_index: {:?}, ", self.feature_index));
            s.push_str(&format!("threshold: {:?}\n", self.threshold));
            s.push_str(&format!("\tleft: {}\n", self.left.as_ref().unwrap()));
            s.push_str(&format!("\tright: {}\n", self.right.as_ref().unwrap()));
            s.push_str("}");
        }
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
    return predict_(x, node.right.as_ref().unwrap());
}

pub fn predict(x: &Vec<Vec<f32>>, node: &Node) -> Vec<u32> {
    let mut y = Vec::new();
    for row in x {
        y.push(predict_(row, node));
    }
    return y;
}

pub fn depth(node: &Node) -> u32 {
    if node.is_leaf {
        return 0;
    }
    let left_depth = depth(node.left.as_ref().unwrap());
    let right_depth = depth(node.right.as_ref().unwrap());
    return 1 + std::cmp::max(left_depth, right_depth);
}
