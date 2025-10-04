use std::fs::File;
use std::io::{self, Read, Write};
use planners::translate::pddl_parser::parse_sexprs;

fn dump(inpath: &str, outpath: &str) -> io::Result<()> {
    let mut s = String::new();
    File::open(inpath)?.read_to_string(&mut s)?;
    let sexprs = parse_sexprs(&s).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    let mut out = File::create(outpath)?;
    for sex in sexprs {
        writeln!(out, "{:?}", sex)?;
    }
    Ok(())
}

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: dump_sexprs DOMAIN.pddl PROBLEM.pddl OUT_DIR");
        std::process::exit(2);
    }
    let out_dir = std::path::Path::new(&args[3]);
    std::fs::create_dir_all(out_dir)?;
    let domain_out = out_dir.join("rust_domain_sexpr.txt");
    let problem_out = out_dir.join("rust_problem_sexpr.txt");
    dump(&args[1], domain_out.to_str().unwrap())?;
    dump(&args[2], problem_out.to_str().unwrap())?;
    println!("Wrote rust domain and problem sexpr files to {}", out_dir.display());
    Ok(())
}
