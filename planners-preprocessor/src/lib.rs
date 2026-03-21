use clap::Parser;
use planners_sas::numeric::numeric_parser::parse_numeric_sas_output;
use planners_sas::numeric::numeric_task::NumericRootTask;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about = "Numeric planner preprocessor")]
pub struct PlannersPreprocessorCli {
    #[arg(value_name = "INPUT", required = true, num_args = 1..=2)]
    pub inputs: Vec<String>,
}
