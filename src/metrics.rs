fn accuracy(y_hat: &Vec<u32>, y: &Vec<u32>) -> f32 {
    let mut correct = 0;
    for (y_hat, y) in y_hat.iter().zip(y) {
        if y_hat == y {
            correct += 1;
        }
    }
    return correct as f32 / y.len() as f32;
}

fn confusion_matrix(y_hat: &Vec<u32>, y: &Vec<u32>) -> (u32, u32, u32, u32) {
    let mut tp = 0;
    let mut tn = 0;
    let mut fp = 0;
    let mut fn_ = 0;
    for (y_hat, y) in y_hat.iter().zip(y) {
        if *y_hat == 1 && *y == 1 {
            tp += 1;
        } else if *y_hat == 0 && *y == 0 {
            tn += 1;
        } else if *y_hat == 1 && *y == 0 {
            fp += 1;
        } else if *y_hat == 0 && *y == 1 {
            fn_ += 1;
        }
    }
    return (tp, tn, fp, fn_);
}

fn precision(y_hat: &Vec<u32>, y: &Vec<u32>) -> f32 {
    let (tp, _, fp, _) = confusion_matrix(y_hat, y);
    return tp as f32 / (tp + fp) as f32;
}

fn recall(y_hat: &Vec<u32>, y: &Vec<u32>) -> f32 {
    let (tp, _, _, fn_) = confusion_matrix(y_hat, y);
    return tp as f32 / (tp + fn_) as f32;
}

fn f1_score(y_hat: &Vec<u32>, y: &Vec<u32>) -> f32 {
    let p = precision(y_hat, y);
    let r = recall(y_hat, y);
    return 2.0 * p * r / (p + r);
}

pub fn classification_report(y_hat: &Vec<u32>, y: &Vec<u32>) {
    println!("Classification Report:");
    println!("\tAccuracy:   {}", accuracy(y_hat, y));
    println!("\tPrecision:  {}", precision(y_hat, y));
    println!("\tRecall:     {}", recall(y_hat, y));
    println!("\tF1 Score:   {}", f1_score(y_hat, y));
}
