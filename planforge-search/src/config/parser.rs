//! Text to AST for the option language.
//!
//! Calls contain positional values and named `key=value` arguments. Values
//! are calls, recursive lists or scalars; quoted scalars preserve delimiters.
//! Bare identifiers and zero-argument calls share the same call node. Paths
//! without whitespace or delimiters remain accepted as unquoted scalars for
//! compatibility. Search engine options are decoded by [`super::SearchConfig`].

use super::{ConfigArg, ConfigCall, ConfigValue, HeuristicSpec};

/// Parse one call, rejecting anything left over. A trailing `.` or `;` is
/// tolerated because shells and papers both like to put one there.
pub fn parse_call(raw: &str) -> Result<ConfigCall, String> {
    let trimmed = raw.trim();
    let input = trimmed
        .strip_suffix('.')
        .or_else(|| trimmed.strip_suffix(';'))
        .unwrap_or(trimmed)
        .trim();
    ConfigParser::new(input).parse_all()
}

/// Parse a heuristic configuration such as `scp(domain(max_states=100),
/// online=false)`.
pub fn parse_heuristic_spec(raw: &str) -> Result<HeuristicSpec, String> {
    parse_call(raw)
}

struct ConfigParser<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> ConfigParser<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, pos: 0 }
    }

    fn parse_all(mut self) -> Result<ConfigCall, String> {
        let mut call = self.parse_call_or_bare()?;
        call.name.make_ascii_lowercase();
        self.skip_ws();
        if self.pos != self.input.len() {
            return Err(format!(
                "Invalid configuration: unexpected input at byte {} near `{}`",
                self.pos,
                &self.input[self.pos..]
            ));
        }
        Ok(call)
    }

    fn parse_call_or_bare(&mut self) -> Result<ConfigCall, String> {
        self.skip_ws();
        let name = self.parse_identifier()?;
        self.skip_ws();
        if !self.consume_char('(') {
            return Ok(ConfigCall {
                name,
                args: Vec::new(),
            });
        }

        let mut args = Vec::new();
        self.skip_ws();
        if self.consume_char(')') {
            return Ok(ConfigCall { name, args });
        }

        loop {
            args.push(
                self.parse_arg()
                    .map_err(|error| format!("{name}(...): {error}"))?,
            );
            self.skip_ws();
            if self.consume_char(',') {
                self.skip_ws();
                if self.consume_char(')') {
                    break;
                }
                continue;
            }
            self.expect_char(')')?;
            break;
        }

        Ok(ConfigCall { name, args })
    }

    fn parse_arg(&mut self) -> Result<ConfigArg, String> {
        self.skip_ws();
        let checkpoint = self.pos;
        if let Ok(key) = self.parse_identifier() {
            self.skip_ws();
            if self.consume_char('=') {
                let value = self.parse_value()?;
                return Ok(ConfigArg {
                    key: Some(key.to_ascii_lowercase()),
                    value,
                });
            }
        }

        self.pos = checkpoint;
        let value = self.parse_value()?;
        Ok(ConfigArg { key: None, value })
    }

    fn parse_value(&mut self) -> Result<ConfigValue, String> {
        self.skip_ws();
        if self.peek_char() == Some('[') {
            return self.parse_list();
        }
        if matches!(self.peek_char(), Some('"' | '\'')) {
            return self.parse_quoted().map(ConfigValue::Atom);
        }
        let checkpoint = self.pos;
        if self.parse_identifier().is_ok() {
            self.skip_ws();
            if matches!(self.peek_char(), Some('(' | ',' | ')' | ']') | None) {
                self.pos = checkpoint;
                return self.parse_call_or_bare().map(ConfigValue::Call);
            }
        }
        self.pos = checkpoint;
        Ok(ConfigValue::Atom(self.parse_scalar()?))
    }

    fn parse_quoted(&mut self) -> Result<String, String> {
        let quote = self.peek_char().expect("quoted value starts with a quote");
        self.pos += 1;
        let mut value = String::new();
        while let Some(ch) = self.peek_char() {
            self.pos += ch.len_utf8();
            if ch == quote {
                return Ok(value);
            }
            if ch == '\\' {
                let escaped = self
                    .peek_char()
                    .ok_or_else(|| self.error("unterminated escape"))?;
                self.pos += escaped.len_utf8();
                value.push(match escaped {
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    '\\' => '\\',
                    '\'' => '\'',
                    '"' => '"',
                    _ => return Err(self.error("unknown quoted escape")),
                });
            } else {
                value.push(ch);
            }
        }
        Err(self.error("unterminated quoted value"))
    }

    fn error(&self, message: &str) -> String {
        format!("Invalid configuration: {message} at byte {}", self.pos)
    }

    fn parse_list(&mut self) -> Result<ConfigValue, String> {
        self.expect_char('[')?;
        let mut values = Vec::new();
        self.skip_ws();
        if self.consume_char(']') {
            return Ok(ConfigValue::List(values));
        }

        loop {
            let value = self
                .parse_value()
                .map_err(|error| format!("list: {error}"))?;
            values.push(value);
            self.skip_ws();
            if self.consume_char(',') {
                self.skip_ws();
                if self.consume_char(']') {
                    break;
                }
                continue;
            }
            self.expect_char(']')?;
            break;
        }

        Ok(ConfigValue::List(values))
    }

    fn parse_identifier(&mut self) -> Result<String, String> {
        self.skip_ws();
        let start = self.pos;
        if !self
            .peek_char()
            .is_some_and(|ch| ch.is_ascii_alphabetic() || ch == '_')
        {
            return Err(self.error("expected identifier"));
        }
        while let Some(ch) = self.peek_char() {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                self.pos += ch.len_utf8();
            } else {
                break;
            }
        }
        if self.pos == start {
            Err(format!(
                "Invalid configuration: expected identifier at byte {}",
                start
            ))
        } else {
            Ok(self.input[start..self.pos].to_string())
        }
    }

    fn parse_scalar(&mut self) -> Result<String, String> {
        self.skip_ws();
        let start = self.pos;
        while let Some(ch) = self.peek_char() {
            if matches!(ch, ',' | ')' | ']' | '[' | '(' | '=' | '\'' | '"') || ch.is_whitespace() {
                break;
            }
            self.pos += ch.len_utf8();
        }
        let value = self.input[start..self.pos].trim();
        if value.is_empty() {
            Err(format!(
                "Invalid configuration: expected value at byte {start}"
            ))
        } else {
            Ok(value.to_string())
        }
    }

    fn skip_ws(&mut self) {
        while let Some(ch) = self.peek_char() {
            if ch.is_whitespace() {
                self.pos += ch.len_utf8();
            } else {
                break;
            }
        }
    }

    fn peek_char(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn consume_char(&mut self, expected: char) -> bool {
        self.skip_ws();
        if self.peek_char() == Some(expected) {
            self.pos += expected.len_utf8();
            true
        } else {
            false
        }
    }

    fn expect_char(&mut self, expected: char) -> Result<(), String> {
        if self.consume_char(expected) {
            Ok(())
        } else {
            Err(format!(
                "Invalid configuration: expected `{expected}` at byte {}",
                self.pos
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FromOptionValue;

    #[test]
    fn bare_names_and_empty_calls_are_one_node() {
        assert_eq!(
            parse_call("root(child)").unwrap(),
            parse_call("root(child())").unwrap()
        );
    }

    #[test]
    fn quoted_paths_keep_case_and_delimiters_and_round_trip() {
        let spec =
            parse_call(r#"root(path="Dir/My,File)=x.txt", other='A\\B', flag=TrUe)"#).unwrap();
        assert_eq!(spec.args[0].value.as_atom().unwrap(), "Dir/My,File)=x.txt");
        assert_eq!(spec.args[1].value.as_atom().unwrap(), "A\\B");
        assert!(bool::from_option_value(&spec.args[2].value).unwrap());
        assert_eq!(parse_call(&spec.to_string()).unwrap(), spec);
        let unquoted = parse_call("root(path=Dir/MyFile.txt)").unwrap();
        assert_eq!(unquoted.args[0].value.as_atom().unwrap(), "Dir/MyFile.txt");
    }

    #[test]
    fn nested_lists_are_values_but_assignments_are_only_call_arguments() {
        let spec = parse_call("root([[child()], [1, 2,],],)").unwrap();
        assert_eq!(parse_call(&spec.to_string()).unwrap(), spec);
        assert!(parse_call("root([child(), partitioning=region])").is_err());
        assert!(parse_call("root(value=one=two)").is_err());
    }

    #[test]
    fn errors_identify_the_failed_construct() {
        let error = parse_call("outer(inner(value=))").unwrap_err();
        assert!(error.contains("inner(...):"), "{error}");
        assert!(error.contains("expected value"), "{error}");
        assert!(!error.contains("--search"));
        assert!(
            parse_call("root(value=\"unterminated)")
                .unwrap_err()
                .contains("unterminated quoted value")
        );
    }
}
