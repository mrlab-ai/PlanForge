#[cfg(test)]
mod regression_tests {
    use std::process::Command;
    use std::fs;

    #[test]
    fn translate_end_to_end_header_consistency() {
        // Run the wrapper translator (expects translate/translate.py to exist)
        let status = Command::new("./translate/translate.py")
            .arg("pddl/domain.pddl")
            .arg("pddl/pfile1.pddl")
            .status()
            .expect("failed to run translator wrapper");
        assert!(status.success(), "translator wrapper failed");

        let s = fs::read_to_string("output.sas").expect("read output.sas");
        // parse header counts
        let mut num_vars: Option<usize> = None;
        let mut num_numvars: Option<usize> = None;
        let mut num_ops_header: Option<usize> = None;
        for line in s.lines().take(200) {
            if line.starts_with("# variables:") {
                if let Some(n) = line.split_whitespace().nth(2) { num_vars = n.parse().ok(); }
            }
            if line.starts_with("# numeric variables:") {
                if let Some(n) = line.split_whitespace().nth(3) { num_numvars = n.parse().ok(); }
            }
            if line.starts_with("# operators:") {
                if let Some(n) = line.split_whitespace().nth(2) { num_ops_header = n.parse().ok(); }
            }
        }
        assert!(num_vars.is_some(), "header missing # variables");
        assert!(num_numvars.is_some(), "header missing # numeric variables");
        assert!(num_ops_header.is_some(), "header missing # operators");

        // count operator blocks and numeric var entries
        let op_blocks = s.matches("begin_operator").count();
        assert_eq!(op_blocks, num_ops_header.unwrap(), "operator count mismatch header vs blocks");

        let num_var_lines = s.lines().filter(|l| l.starts_with("var ")).count();
        assert_eq!(num_var_lines, num_vars.unwrap(), "variable count mismatch");

        let num_numvar_lines = s.lines().filter(|l| l.starts_with("num ")).count();
        assert_eq!(num_numvar_lines, num_numvars.unwrap(), "numeric variable count mismatch");
    }
}
