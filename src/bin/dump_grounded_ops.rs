use std::fs::File;
use std::io::Write;
use anyhow::Context;
use planners::translate::pddl::PddlTask;
use planners::translate::pddl_ast::{Domain, Problem};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: dump_grounded_ops DOMAIN.pddl PROBLEM.pddl OUT_FILE");
        std::process::exit(2);
    }
    let domain = std::path::Path::new(&args[1]);
    let problem = std::path::Path::new(&args[2]);
    let out = &args[3];

    let task = PddlTask::from_files(domain, problem).context("parsing files")?;
    let dom = Domain::from_sexprs(&task.domain_forms).context("domain->ast")?;
    let prob = Problem::from_sexprs(&task.problem_forms).context("problem->ast")?;

    let ops = planners::translate::instantiate::ground(&dom, &prob);

    let mut f = File::create(out)?;
    writeln!(f, "grounded_ops: {}", ops.len())?;
    for op in ops {
        // name(args) pre_count eff_count
        let pre_count = match &op.pre { Some(p) => format!("{:?}", p), None => "".to_string() };
        let eff_count = match &op.eff { Some(e) => format!("{:?}", e), None => "".to_string() };
        writeln!(f, "{}\tpre={}\teff={}", op.name, pre_count, eff_count)?;
    }
    println!("WROTE {}", out);
    Ok(())
}
