use std::io::Write;

/// A tiny adapter to run Prolog queries by invoking an external SWI-Prolog process.
/// This avoids adding a heavy embedded Prolog dependency while enabling useful checks.
pub struct PrologEngine {
    /// Optional path to swipl binary; defaults to "swipl" in PATH
    pub swipl_path: String,
}

impl Default for PrologEngine {
    fn default() -> Self { Self { swipl_path: "swipl".to_string() } }
}

impl PrologEngine {
    pub fn new() -> Self { Self::default() }

    /// Consult a single .pl text (written to a temp file) and run a query string.
    /// Returns the stdout from SWI-Prolog or a helpful error if swipl is missing.
    pub fn consult_and_query(&self, prolog_text: &str, query: &str) -> anyhow::Result<String> {
        let dir = tempfile::Builder::new().prefix("pl-demo-").tempdir()?;
        let path = dir.path().join("facts.pl");
        {
            let mut f = std::fs::File::create(&path)?;
            f.write_all(prolog_text.as_bytes())?;
        }
        self.run_with_files(&[path.to_string_lossy().to_string()], query)
    }

    /// Consult multiple .pl files and run a query.
    pub fn run_with_files(&self, files: &[String], query: &str) -> anyhow::Result<String> {
        // Build a small Prolog goal that consults files, runs the query, and halts.
        // We'll pass it via -g to swipl.
        let mut goal_parts: Vec<String> = Vec::new();
        for f in files {
            // escape single quotes in path
            let esc = f.replace('\'', "\\'");
            goal_parts.push(format!("consult('{}')", esc));
        }
        // Run the provided query and then halt.
        // If the query doesn't end with a period, add one.
        let mut q = query.trim().to_string();
        if !q.ends_with('.') { q.push('.'); }
        goal_parts.push(q);
        goal_parts.push("halt".to_string());
        let goal = goal_parts.join(", ");

        let output = std::process::Command::new(&self.swipl_path)
            .args(["-q", "-g", &goal])
            .output();
        match output {
            Ok(out) => {
                if out.status.success() {
                    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                    Ok(stdout)
                } else {
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    Err(anyhow::anyhow!(format!("swipl failed: {}", stderr)))
                }
            }
            Err(e) => {
                Err(anyhow::anyhow!(format!(
                    "Failed to execute '{}': {}. Ensure SWI-Prolog is installed and 'swipl' is on PATH.",
                    &self.swipl_path, e
                )))
            }
        }
    }
}
