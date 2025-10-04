use planners::translate::pddl_parser::parse_sexprs;
use planners::translate::pddl_to_prolog::{domain_to_prolog, problem_to_prolog};
use std::fs::File;
use std::io::{Read, Write};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: dump_prolog DOMAIN.pddl PROBLEM.pddl OUT_DIR");
        std::process::exit(2);
    }
    let domain = &args[1];
    let problem = &args[2];
    let outdir = std::path::Path::new(&args[3]);
    std::fs::create_dir_all(outdir)?;
    let mut d = String::new();
    let mut p = String::new();
    std::fs::File::open(domain)?.read_to_string(&mut d)?;
    std::fs::File::open(problem)?.read_to_string(&mut p)?;
    let df = parse_sexprs(&d).map_err(|e| anyhow::anyhow!(e))?;
    let pf = parse_sexprs(&p).map_err(|e| anyhow::anyhow!(e))?;
    let dom_pl = domain_to_prolog(&df);
    let prob_pl = problem_to_prolog(&pf);
    let mut fd = File::create(outdir.join("rust_domain.pl"))?;
    fd.write_all(dom_pl.as_bytes())?;
    let mut fp = File::create(outdir.join("rust_problem.pl"))?;
    fp.write_all(prob_pl.as_bytes())?;
    println!(
        "WROTE {} and {}",
        outdir.join("rust_domain.pl").display(),
        outdir.join("rust_problem.pl").display()
    );
    Ok(())
}
