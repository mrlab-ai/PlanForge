use nom::branch::alt;
use nom::bytes::complete::{take_while, take_while1};
use nom::character::complete::{char, multispace1};
use nom::combinator::{map, recognize};
use nom::error::{convert_error, VerboseError};
use nom::multi::many0;
use nom::sequence::{delimited, preceded};
use nom::IResult;

const COMMENT_CHAR: char = ';';
const ERROR_SNIPPET_LEN: usize = 80;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SExpr {
    Atom(String),
    List(Vec<SExpr>),
}

fn is_atom_char(c: char) -> bool {
    !c.is_whitespace() && c != '(' && c != ')' && c != ';'
}

fn comment(input: &str) -> IResult<&str, &str, VerboseError<&str>> {
    preceded(char(COMMENT_CHAR), take_while(|c| c != '\n'))(input)
}

fn ws(input: &str) -> IResult<&str, (), VerboseError<&str>> {
    let (i, _) = many0(alt((multispace1, map(comment, |_| ""))))(input)?;
    Ok((i, ()))
}

fn atom(input: &str) -> IResult<&str, SExpr, VerboseError<&str>> {
    let (i, s) = recognize(take_while1(is_atom_char))(input)?;
    Ok((i, SExpr::Atom(s.to_string())))
}

fn list(input: &str) -> IResult<&str, SExpr, VerboseError<&str>> {
    let (i, vec) = delimited(
        preceded(ws, char('(')),
        many0(preceded(ws, sexpr)),
        preceded(ws, char(')')),
    )(input)?;
    Ok((i, SExpr::List(vec)))
}

fn sexpr(input: &str) -> IResult<&str, SExpr, VerboseError<&str>> {
    alt((list, preceded(ws, atom)))(input)
}

fn consume_ws_comments(input: &str) -> Result<&str, String> {
    ws(input)
        .map(|(i, _)| i)
        .map_err(|e| format_parse_error(input, e))
}

fn format_parse_error(input: &str, err: nom::Err<VerboseError<&str>>) -> String {
    match err {
        nom::Err::Error(e) | nom::Err::Failure(e) => {
            let snippet = input.trim_start();
            let preview = &snippet[..snippet.len().min(ERROR_SNIPPET_LEN)];
            format!(
                "parse error:\n{}\nnear: {}",
                convert_error(input, e),
                preview
            )
        }
        nom::Err::Incomplete(_) => "parse error: incomplete input".to_string(),
    }
}

pub fn parse_sexprs(input: &str) -> Result<Vec<SExpr>, String> {
    let mut rest = input;
    let mut out = Vec::new();
    loop {
        rest = consume_ws_comments(rest)?;
        if rest.trim().is_empty() {
            break;
        }
        match sexpr(rest) {
            Ok((i, s)) => {
                out.push(s);
                rest = i;
            }
            Err(e) => {
                return Err(format_parse_error(rest, e));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn parse_pfile1_smoke() {
        let s = fs::read_to_string("misc/plant-watering/domain.pddl").expect("read pddl file");
        let sexprs = parse_sexprs(&s).expect("parse should succeed");
        assert!(!sexprs.is_empty());
        // first form should be (define ...)
        match &sexprs[0] {
            SExpr::List(items) => match &items[0] {
                SExpr::Atom(a) => assert_eq!(a.to_lowercase(), "define"),
                _ => panic!("expected atom define"),
            },
            _ => panic!("expected list"),
        }
    }
}
