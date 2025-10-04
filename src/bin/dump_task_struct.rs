use planners::translate::pddl_ast::{Domain, Problem};
use planners::translate::pddl_parser::parse_sexprs;
use serde_json::json;
use std::fs::File;
use std::io::Read;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: dump_task_struct DOMAIN.pddl PROBLEM.pddl OUT_JSON");
        std::process::exit(2);
    }
    let domain = &args[1];
    let problem = &args[2];
    let out = &args[3];
    let mut d = String::new();
    let mut p = String::new();
    File::open(domain)?.read_to_string(&mut d)?;
    File::open(problem)?.read_to_string(&mut p)?;
    let df = parse_sexprs(&d).map_err(|e| anyhow::anyhow!(e))?;
    let pf = parse_sexprs(&p).map_err(|e| anyhow::anyhow!(e))?;
    let dom = Domain::from_sexprs(&df).ok_or_else(|| anyhow::anyhow!("domain parse failed"))?;
    let prob = Problem::from_sexprs(&pf).ok_or_else(|| anyhow::anyhow!("problem parse failed"))?;
    let domain_json = json!({
        "name": dom.name,
        "predicates": dom.predicates.iter().map(|(n, args)| {
            let a: Vec<_> = args.iter().map(|(p,t)| json!({"name": p, "type": t})).collect();
            json!({"name": n, "args": a})
        }).collect::<Vec<_>>() ,
        "functions": dom.functions.iter().map(|(n,args)| {
            let a: Vec<_> = args.iter().map(|(p,t)| json!({"name": p, "type": t})).collect();
            json!({"name": n, "args": a})
        }).collect::<Vec<_>>()
    });
    let problem_json = json!({
        "name": prob.name,
        "objects": prob.objects.iter().map(|(n,t)| json!({"name": n, "type": t})).collect::<Vec<_>>(),
        "init": prob.init.iter().map(|s| format!("{:?}", s)).collect::<Vec<_>>(),
        "goal": prob.goal.as_ref().map(|g| format!("{:?}", g)),
    });
    let whole = json!({"domain": domain_json, "problem": problem_json});
    std::fs::write(out, serde_json::to_string_pretty(&whole)?)?;
    println!("WROTE {}", out);
    Ok(())
}
