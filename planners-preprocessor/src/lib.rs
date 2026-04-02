use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about = "Numeric planner preprocessor")]
pub struct PlannersPreprocessorCli {
    #[arg(value_name = "INPUT", required = true, num_args = 1..=2)]
    pub inputs: Vec<String>,
}
