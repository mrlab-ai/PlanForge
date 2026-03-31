use serde::{Deserialize, Serialize};
use std::fmt;

use nom::{
    IResult,
    branch::alt,
    bytes::complete::tag_no_case,
    character::complete::{char, multispace0, one_of},
    combinator::{all_consuming, cut, map, opt},
    error::{VerboseError, convert_error},
    sequence::{delimited, terminated, tuple},
};

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum HeuristicSpec {
    Blind,
    #[serde(rename = "domain_abstraction")]
    DomainAbstraction,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SearchSpec {
    Astar(HeuristicSpec),
    #[serde(rename = "da_debug")]
    DaDebug,
    #[serde(rename = "astar_da_debug")]
    AstarDaDebug,
}

impl fmt::Display for HeuristicSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HeuristicSpec::Blind => write!(f, "blind()"),
            HeuristicSpec::DomainAbstraction => write!(f, "domain_abstraction()"),
        }
    }
}

impl fmt::Display for SearchSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SearchSpec::Astar(h) => write!(f, "astar({h})"),
            SearchSpec::DaDebug => write!(f, "da_debug()"),
            SearchSpec::AstarDaDebug => write!(f, "astar_da_debug()"),
        }
    }
}

pub fn parse_search_spec(raw: &str) -> Result<SearchSpec, String> {
    let input = raw;
    match all_consuming(ws(terminated(search_spec, opt(ws(one_of(".;"))))))(input) {
        Ok((_, spec)) => Ok(spec),
        Err(nom::Err::Error(e)) | Err(nom::Err::Failure(e)) => Err(format!(
            "Invalid --search config:\n{}",
            convert_error(input, e)
        )),
        Err(nom::Err::Incomplete(_)) => Err("Invalid --search config: incomplete input".into()),
    }
}

type Res<'a, T> = IResult<&'a str, T, VerboseError<&'a str>>;

fn ws<'a, F: 'a, O>(inner: F) -> impl FnMut(&'a str) -> Res<'a, O>
where
    F: FnMut(&'a str) -> Res<'a, O>,
{
    delimited(multispace0, inner, multispace0)
}

fn empty_parens(input: &str) -> Res<'_, ()> {
    map(delimited(ws(char('(')), multispace0, ws(char(')'))), |_| ())(input)
}

fn heuristic_spec(input: &str) -> Res<'_, HeuristicSpec> {
    let blind = map(
        tuple((ws(tag_no_case("blind")), opt(ws(empty_parens)))),
        |_| HeuristicSpec::Blind,
    );

    let domain_abstraction = map(
        tuple((ws(tag_no_case("domain_abstraction")), opt(ws(empty_parens)))),
        |_| HeuristicSpec::DomainAbstraction,
    );

    ws(alt((domain_abstraction, blind)))(input)
}

fn search_spec(input: &str) -> Res<'_, SearchSpec> {
    let da_debug = map(
        tuple((ws(tag_no_case("da_debug")), opt(ws(empty_parens)))),
        |_| SearchSpec::DaDebug,
    );

    let astar_da_debug = map(
        tuple((ws(tag_no_case("astar_da_debug")), opt(ws(empty_parens)))),
        |_| SearchSpec::AstarDaDebug,
    );

    let astar = map(
        tuple((
            ws(tag_no_case("astar")),
            ws(char('(')),
            cut(heuristic_spec),
            ws(char(')')),
        )),
        |(_, _, h, _)| SearchSpec::Astar(h),
    );
    ws(alt((astar, astar_da_debug, da_debug)))(input)
}

#[cfg(test)]
mod tests;
