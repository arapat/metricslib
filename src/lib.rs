mod metric;
mod utils;

use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use metric::get_adaboost_loss;
use metric::get_auprc;
use metric::get_auroc;
use metric::get_error_rate;


pub enum EvalFunc {
    AdaBoostLoss,
    ErrorRate,
    AUPRC,
    AUROC,
}


pub fn validate(
    testing_filename: &String,
    prediction_filename: &String,
    eval_funcs: Vec<EvalFunc>,
    positive_str: &String,
) -> Vec<f32> {
    let f_test = File::open(testing_filename.clone())
                      .expect(&format!("Cannot open the file `{}`.", testing_filename));
    let mut test_reader = BufReader::new(f_test);
    let f_pred = File::open(prediction_filename.clone())
                      .expect(&format!("Cannot open the file `{}`.", prediction_filename));
    let mut pred_reader = BufReader::new(f_pred);

    let sorted_scores_labels = {
        let mut scores_labels: Vec<(f32, f32)> = vec![];
        let (mut score, mut label) = (String::new(), String::new());
        while test_reader.read_line(&mut label).is_ok() {
            let binary_label = {
                if label == *positive_str {
                    1.0
                } else {
                    -1.0
                }
            };
            pred_reader.read_line(&mut score).unwrap();
            scores_labels.push((score.parse().unwrap(), binary_label));
        }
        scores_labels.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().reverse());
        scores_labels
    };

    let scores: Vec<f32> =
        eval_funcs.iter()
                  .map(|func| {
                      match func {
                          EvalFunc::AdaBoostLoss => get_adaboost_loss(&sorted_scores_labels),
                          EvalFunc::ErrorRate    => get_error_rate(&sorted_scores_labels),
                          EvalFunc::AUPRC        => get_auprc(&sorted_scores_labels),
                          EvalFunc::AUROC        => get_auroc(&sorted_scores_labels),
                      }
                  }).collect();
    scores
}