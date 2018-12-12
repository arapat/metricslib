extern crate metricslib;

use std::collections::HashMap;
use std::env;
use metricslib::EvalFunc;
use metricslib::validate;


fn main() {
    let eval_funcs = vec![EvalFunc::AdaBoostLoss, EvalFunc::AUPRC, EvalFunc::AUROC];

    let args: Vec<String> = env::args().collect();
    let parsed_args = parse_args(args);
    let usage_info = "Usage: ./metricslib --test <testing data path> \\
                    --scores <model predictions path> \\
                    --positive <label for the positive examples>";
    if parsed_args.is_none() {
        println!("Parameters are invalid.");
        println!("{}", usage_info);
        return;
    }
    let parsed_args = parsed_args.unwrap();
    if parsed_args.get("help").is_some() {
        println!("{}", usage_info);
    } else if parsed_args.get("test").is_none() || parsed_args.get("scores").is_none() ||
              parsed_args.get("positive").is_none() {
        println!("Parameters are invalid.");
        println!("{}", usage_info);
    } else {
        let perf = validate(parsed_args.get("test").unwrap(), parsed_args.get("scores").unwrap(),
                            eval_funcs, parsed_args.get("positive").unwrap());
        let perf: Vec<String> = perf.iter().map(|t| t.to_string()).collect();
        println!("{}", perf.join(", "));
    }
}


fn parse_args(args: Vec<String>) -> Option<HashMap<String, String>> {
    let mut k = 1;
    let mut parsed = HashMap::new();
    while k < args.len() {
        match args[k].as_ref() {
            "--test" | "--scores" | "--positive" => {
                parsed.insert(args[k][2..].to_string(), args[k + 1].to_string());
                k += 2;
            },
            "--help" => {
                parsed.insert("help".to_string(), "HELP".to_string());
                k += 1;
            },
            _ => {
                return None;
            }
        }
    }
    if parsed.len() <= 0 {
        None
    } else {
        Some(parsed)
    }
}