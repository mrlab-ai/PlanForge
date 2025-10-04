use crate::translate::pddl_parser::SExpr;

fn as_atom(s: &SExpr) -> Option<&String> {
	match s {
		SExpr::Atom(a) => Some(a),
		_ => None,
	}
}

fn lower(s: &str) -> String {
	s.to_lowercase()
}

fn parse_typed_list(list: &[SExpr]) -> Vec<(String, Option<String>)> {
	let mut out = Vec::new();
	let mut buf: Vec<String> = Vec::new();
	let mut i = 0usize;
	while i < list.len() {
		match &list[i] {
			SExpr::Atom(a) if a == "-" => {
				if i + 1 < list.len() {
					if let SExpr::Atom(t) = &list[i + 1] {
						for name in buf.drain(..) {
							out.push((name, Some(t.clone())));
						}
						i += 2;
						continue;
					}
				}
				i += 1;
			}
			SExpr::Atom(a) => {
				buf.push(a.clone());
				i += 1;
			}
			_ => { i += 1; }
		}
	}
	for name in buf.drain(..) { out.push((name, None)); }
	out
}

fn sexpr_to_atom_name(s: &SExpr) -> String {
	format_sexpr_term(s)
}

fn sanitize_atom(a: &str) -> String {
	a.replace('-', "_")
}

fn format_sexpr_term(s: &SExpr) -> String {
	match s {
		SExpr::Atom(a) => sanitize_atom(a),
		SExpr::List(list) => {
			if list.is_empty() {
				return "list()".to_string();
			}
			// if first element is an atom, render as functor(args...)
			if let SExpr::Atom(head) = &list[0] {
				let head_s = sanitize_atom(head);
				// special-case assignment: (= (f args...) val)
				if head == "=" && list.len() == 3 {
					if let SExpr::List(lhs) = &list[1] {
						if let SExpr::Atom(fname) = &lhs[0] {
							let args: Vec<String> = lhs[1..].iter().map(|x| format_sexpr_term(x)).collect();
							let rhs = format_sexpr_term(&list[2]);
							return format!("assign({},{},{})", sanitize_atom(fname), args.join(","), rhs);
						}
					}
				}
				let args: Vec<String> = list[1..].iter().map(|x| format_sexpr_term(x)).collect();
				format!("{}({})", head_s, args.join(", "))
			} else {
				// generic list
				let parts: Vec<String> = list.iter().map(|x| format_sexpr_term(x)).collect();
				format!("list({})", parts.join(", "))
			}
		}
	}
}

/// Convert domain SExpr forms (as returned by `parse_sexprs`) into a
/// Prolog-like string of facts.
pub fn domain_to_prolog(forms: &[SExpr]) -> String {
	let mut lines: Vec<String> = Vec::new();
	for f in forms {
		if let SExpr::List(items) = f {
			if items.len() >= 2 {
				if let Some(SExpr::Atom(a0)) = items.get(0) {
					if a0.to_lowercase() == "define" {
						// domain name usually at items[1] -> (domain NAME)
						if let Some(SExpr::List(domain_spec)) = items.get(1) {
							if domain_spec.len() >= 2 {
								if let SExpr::Atom(domain_kw) = &domain_spec[0] {
									if domain_kw.to_lowercase() == "domain" {
										if let SExpr::Atom(name) = &domain_spec[1] {
											lines.push(format!("domain({}).", name));
										}
									}
								}
							}
						}
						// iterate over sections
						for section in &items[2..] {
							if let SExpr::List(list) = section {
								if list.is_empty() { continue; }
								if let SExpr::Atom(key) = &list[0] {
									match key.to_lowercase().as_str() {
										":predicates" | "predicates" => {
											for item in &list[1..] {
												if let SExpr::List(p) = item {
													if !p.is_empty() {
														if let SExpr::Atom(nm) = &p[0] {
															// count args ignoring type markers
															let parsed = parse_typed_list(&p[1..]);
															lines.push(format!("predicate({},{}).", nm, parsed.len()));
														}
													}
												}
											}
										}
										":functions" | "functions" => {
											for item in &list[1..] {
												if let SExpr::List(p) = item {
													if !p.is_empty() {
														if let SExpr::Atom(nm) = &p[0] {
															let parsed = parse_typed_list(&p[1..]);
															lines.push(format!("function({},{}).", nm, parsed.len()));
														}
													}
												} else if let SExpr::Atom(nm) = item {
													// zero-arg function like (cost)
													lines.push(format!("function({},0).", nm));
												}
											}
										}
										":action" | "action" => {
											// content often: name, :parameters, (...), :precondition, (...), :effect, (...)
											if list.len() >= 2 {
												if let SExpr::Atom(name) = &list[1] {
													lines.push(format!("action({}).", name));
													// scan for parameters, precondition, effect
													let mut i = 2usize;
													while i < list.len() {
														if let SExpr::Atom(k) = &list[i] {
															let kl = k.to_lowercase();
															if kl == ":parameters" && i + 1 < list.len() {
																if let SExpr::List(params) = &list[i + 1] {
																	let parsed = parse_typed_list(params);
																	for (idx, (pname, ptype)) in parsed.iter().enumerate() {
																		match ptype {
																			Some(t) => lines.push(format!("action_param({}, {}, {}, type({})).", name, idx, pname, t)),
																			None => lines.push(format!("action_param({}, {}, {}).", name, idx, pname)),
																		}
																	}
																}
																i += 2;
																continue;
															} else if kl == ":precondition" && i + 1 < list.len() {
																// precondition is an SExpr; emit simple atom facts inside (and ...)
																if let Some(pre) = list.get(i + 1) {
																	match pre {
																		SExpr::List(pl) => {
																			for item in &pl[1..] {
																				if let SExpr::List(a) = item {
																					if let Some(SExpr::Atom(pred)) = a.get(0) {
																						let args: Vec<String> = a[1..].iter().filter_map(|x| match x { SExpr::Atom(s) => Some(s.clone()), _ => None }).collect();
																						lines.push(format!("action_pre({}, {}({})).", name, pred, args.join(", ")));
																					}
																				}
																			}
																		}
																		_ => {}
																	}
																}
																i += 2;
																continue;
															} else if kl == ":effect" && i + 1 < list.len() {
																if let Some(eff) = list.get(i + 1) {
																	if let SExpr::List(el) = eff {
																		for item in &el[1..] {
																			match item {
																				SExpr::List(a) => {
																					if let Some(SExpr::Atom(tag)) = a.get(0) {
																						match tag.as_str() {
																							"not" => {
																								if let Some(SExpr::List(inner)) = a.get(1) {
																									if let Some(SExpr::Atom(pred)) = inner.get(0) {
																										let args: Vec<String> = inner[1..].iter().filter_map(|x| match x { SExpr::Atom(s) => Some(s.clone()), _ => None }).collect();
																										lines.push(format!("action_eff_del({}, {}({})).", name, pred, args.join(", ")));
																									}
																								}
																							}
																							"increase" | "decrease" => {
																								// numeric effect (increase (f) val)
																								if a.len() >= 3 {
																									if let SExpr::List(flst) = &a[1] {
																										if let Some(SExpr::Atom(fname)) = flst.get(0) {
																											if let SExpr::Atom(v) = &a[2] {
																												lines.push(format!("action_eff_num({}, {}, {}).", name, fname, v));
																											}
																										}
																									}
																								}
																							}
																							_ => {
																								// add effect
																								let pred = tag;
																								let args: Vec<String> = a[1..].iter().filter_map(|x| match x { SExpr::Atom(s) => Some(s.clone()), _ => None }).collect();
																								lines.push(format!("action_eff_add({}, {}({})).", name, pred, args.join(", ")));
																							}
																						}
																					}
																				}
																				SExpr::Atom(a) => {
																					lines.push(format!("action_eff_add({}, {}).", name, a));
																				}
																			}
																		}
																	}
																}
																i += 2;
																continue;
															}
														}
														i += 1;
													}
												}
											}
										}
										_ => {}
									}
								}
							}
						}
					}
				}
			}
		}
	}
	lines.join("\n")
}

/// Convert problem SExpr forms into Prolog-like facts (objects, init, goal, metric).
pub fn problem_to_prolog(forms: &[SExpr]) -> String {
	let mut lines: Vec<String> = Vec::new();
	for f in forms {
		if let SExpr::List(items) = f {
			if items.len() >= 2 {
				if let Some(SExpr::Atom(a0)) = items.get(0) {
					if a0.to_lowercase() == "define" {
						// iterate parts
						for part in &items[1..] {
							if let SExpr::List(inner) = part {
								if inner.is_empty() { continue; }
								if let SExpr::Atom(atom0) = &inner[0] {
									match atom0.to_lowercase().as_str() {
										":problem" | "problem" => {
											if inner.len() >= 2 {
												if let SExpr::Atom(n) = &inner[1] {
													lines.push(format!("problem({}).", n));
												}
											}
										}
										":objects" | "objects" => {
											let typed = parse_typed_list(&inner[1..]);
											for (n, t) in typed {
												if let Some(tp) = t {
													lines.push(format!("object({}, type({})).", n, tp));
												} else {
													lines.push(format!("object({}).", n));
												}
											}
										}
										":init" | "init" => {
											for token in &inner[1..] {
												// print raw atom-like facts
												match token {
													SExpr::List(lst) => {
														if let Some(SExpr::Atom(name)) = lst.get(0) {
															let args: Vec<String> = lst[1..].iter().map(|a| sexpr_to_atom_name(a)).collect();
															lines.push(format!("init({}({})).", name.replace('-', "_"), args.join(", ")));
														}
													}
													SExpr::Atom(a) => {
														lines.push(format!("init({}).", a.replace('-', "_")));
													}
												}
											}
										}
										":goal" | "goal" => {
											if inner.len() >= 2 {
												if let SExpr::List(g) = &inner[1] {
													if let Some(SExpr::Atom(gn)) = g.get(0) {
														let args: Vec<String> = g[1..].iter().map(|a| sexpr_to_atom_name(a)).collect();
														lines.push(format!("goal({}({})).", gn.replace('-', "_"), args.join(", ")));
													}
												}
											}
										}
										":metric" | "metric" => {
											if inner.len() >= 2 {
												let metric_kind = if let SExpr::Atom(k) = &inner[1] { k.clone() } else { "minimize".to_string() };
												let metric_expr = if inner.len() >= 3 { format!("{:?}", inner[2]) } else { "".to_string() };
												lines.push(format!("metric({}, {}).", metric_kind, metric_expr));
											}
										}
										_ => {}
									}
								}
							}
						}
					}
				}
			}
		}
	}
	lines.join("\n")
}
