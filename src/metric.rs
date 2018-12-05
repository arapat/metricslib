
use utils::is_zero;

pub fn get_adaboost_loss(scores_labels: &Vec<(f32, f32)>) -> f32 {
    let loss: f32 = scores_labels.iter()
                                 .map(|&(score, label)| (-score * label).exp())
                                 .sum();
    loss / (scores_labels.len() as f32)
}


pub fn get_error_rate(scores_labels: &Vec<(f32, f32)>) -> f32 {
    let error: usize = scores_labels.iter()
                                    .map(|&(score, label)| (score * label <= 1e-8) as usize)
                                    .sum();
    (error as f32) / (scores_labels.len() as f32)
}


pub fn get_auprc(sorted_scores_labels: &Vec<(f32, f32)>) -> f32 {
    let (fps, tps, _) = get_fps_tps(sorted_scores_labels);

    let num_positive = tps[tps.len() - 1] as f32;
    let precision: Vec<f32> = tps.iter()
                                 .zip(fps.iter())
                                 .map(|(tp, fp)| (*tp as f32) / ((tp + fp) as f32))
                                 .collect();
    let recall: Vec<f32> = tps.iter()
                              .map(|tp| (*tp as f32) / num_positive)
                              .collect();
    let area_first_seg = (precision[0] as f32) * (recall[0] as f32);
    let mut points: Vec<(f32, f32)> = recall.into_iter()
                                            .zip(precision.into_iter())
                                            .collect();
    area_first_seg + get_auc(&mut points)
}


pub fn get_auroc(sorted_scores_labels: &Vec<(f32, f32)>) -> f32 {
    let (fps, tps, _) = get_fps_tps(sorted_scores_labels);

    let num_fp = fps[fps.len() - 1] as f32;
    let fpr: Vec<f32> = fps.into_iter()
                           .map(|a| (a as f32) / num_fp)
                           .collect();
    let num_tp = tps[tps.len() - 1] as f32;
    let tpr: Vec<f32> = tps.into_iter()
                           .map(|a| (a as f32) / num_tp)
                           .collect();

    let area_first_seg = fpr[0] * tpr[0] / 2.0;
    let mut points: Vec<(f32, f32)> = fpr.into_iter()
                                         .zip(tpr.into_iter())
                                         .collect();
    area_first_seg + get_auc(&mut points)
}


fn get_auc(points: &mut Vec<(f32, f32)>) -> f32 {
    let mut iter = points.iter();
    let (mut x0, mut y0) = iter.next().unwrap();
    let mut area = 0.0;
    for &(x1, y1) in iter {
        area += (x1 - x0) * (y0 + y1);
        x0 = x1;
        y0 = y1;
    }
    area / 2.0
}


fn get_fps_tps(sorted_scores_labels: &Vec<(f32, f32)>) -> (Vec<usize>, Vec<usize>, Vec<f32>) {
    let capacity = sorted_scores_labels.len();
    let mut fps = Vec::with_capacity(capacity);
    let mut tps = Vec::with_capacity(capacity);
    let mut thresholds = Vec::with_capacity(capacity);

    let mut iter = sorted_scores_labels.iter();
    let ret = iter.next().unwrap();
    let mut last_score = ret.0;
    let last_label = ret.1;
    let mut tp = is_zero(last_label - 1.0) as usize;
    let mut fp = 1 - tp;
    for &(score, label) in iter {
        if !is_zero(score - last_score) {
            fps.push(fp);
            tps.push(tp);
            thresholds.push(last_score);
        }
        let c = is_zero(label - 1.0) as usize;
        tp += c;
        fp += 1 - c;
        last_score = score;
    }
    fps.push(fp);
    tps.push(tp);
    thresholds.push(last_score);
    (fps, tps, thresholds)
}


#[cfg(test)]
mod tests {
    use super::super::utils::is_zero;
    use super::get_adaboost_loss;
    use super::get_error_rate;
    use super::get_auprc;
    use super::get_auroc;

    #[test]
    fn test_adaboost_loss() {
        let ans = 0.8856572;
        assert!(is_zero(get_adaboost_loss(&get_sorted_scores_labels()) - ans));
    }

    #[test]
    fn test_error_rate() {
        let ans = 0.24;
        assert!(is_zero(get_error_rate(&get_sorted_scores_labels()) - ans));
    }

    #[test]
    fn test_auprc() {
        let ans = 0.8759296479267399;
        assert!(is_zero(get_auprc(&get_sorted_scores_labels()) - ans));
    }

    #[test]
    fn test_auroc() {
        let ans = 0.8509615384615384;
        assert!(is_zero(get_auroc(&get_sorted_scores_labels()) - ans));
    }

    fn get_sorted_scores_labels() -> Vec<(f32, f32)> {
        let scores = vec![
            -0.20078777,  0.30424058,  0.20106318,  0.27524034,  0.42593626,
            -0.15043766, -0.08794325, -0.127333  ,  0.22931613, -0.23913315,
            -0.06385956, -0.14958104, -0.04914672,  0.09898447,  0.05156366,
            -0.11429083,  0.18900161,  0.04871914, -0.08257976, -0.2610529 ,
             0.24693203, -0.18318166, -0.38384888,  0.26336977,  0.12585524,
            -0.03990992,  0.39424863,  0.42411777, -0.47904305, -0.30528804,
            -0.09281708,  0.01213704, -0.20204097,  0.401491  , -0.0453569 ,
             0.12179294,  0.06494036, -0.07007126,  0.00329292, -0.39635519,
             0.02619553,  0.20018893,  0.06502325,  0.49589744, -0.28221727,
             0.31364864,  0.19062428,  0.11549715,  0.031461  ,  0.22408891,
        ];
        let labels: Vec<f32> = vec![
            1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0,
        ];
        let labels: Vec<f32> = labels.into_iter().map(|t| 2.0 * t - 1.0).collect();
        let mut scores_labels: Vec<(f32, f32)> = scores.into_iter().zip(labels.into_iter()).collect();
        scores_labels.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().reverse());
        scores_labels
    }
}