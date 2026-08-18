//! Text to AST for the option language.
//!
//! One grammar: `name`, `name(...)`, and inside the parentheses a
//! comma-separated list of positional values and `key=value` pairs, where a
//! value is itself a call or a bare scalar. Every `--search` heuristic, every
//! abstraction source and every nested collection is spelled in it, so it is
//! parsed once, here, next to the [`ConfigArg`](super::ConfigArg) nodes it
//! produces and the configs those populate.
//!
//! Which *search engine* a spec names is a separate, smaller grammar, and lives
//! with the CLI in `planforge-searcher`.

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
    let call = parse_call(raw)?;
    Ok(HeuristicSpec::new(call.name, call.args))
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
        let call = self.parse_call_or_bare()?;
        self.skip_ws();
        if self.pos != self.input.len() {
            return Err(format!(
                "Invalid --search config: unexpected input at byte {} near `{}`",
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
            args.push(self.parse_arg()?);
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
                    key: Some(key),
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
        let checkpoint = self.pos;
        if let Ok(name) = self.parse_identifier() {
            self.skip_ws();
            let is_call = self.peek_char() == Some('(');
            if is_call {
                self.pos = checkpoint;
                let call = self.parse_call_or_bare()?;
                return if call.args.is_empty() {
                    Ok(ConfigValue::Atom(call.name))
                } else {
                    Ok(ConfigValue::Call(call))
                };
            }
            if matches!(self.peek_char(), Some(',') | Some(')') | Some(']') | None) {
                return Ok(ConfigValue::Atom(name));
            }
        }
        self.pos = checkpoint;
        Ok(ConfigValue::Atom(self.parse_scalar()?))
    }

    fn parse_list(&mut self) -> Result<ConfigValue, String> {
        self.expect_char('[')?;
        let mut values = Vec::new();
        self.skip_ws();
        if self.consume_char(']') {
            return Ok(ConfigValue::List(values));
        }

        loop {
            let value = self.parse_value()?;
            if matches!(value, ConfigValue::List(_)) {
                return Err("Invalid --search config: nested lists are not supported".to_string());
            }
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
        while let Some(ch) = self.peek_char() {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                self.pos += ch.len_utf8();
            } else {
                break;
            }
        }
        if self.pos == start {
            Err(format!(
                "Invalid --search config: expected identifier at byte {}",
                start
            ))
        } else {
            Ok(self.input[start..self.pos].to_ascii_lowercase())
        }
    }

    fn parse_scalar(&mut self) -> Result<String, String> {
        self.skip_ws();
        let start = self.pos;
        while let Some(ch) = self.peek_char() {
            if ch == ',' || ch == ')' || ch == ']' || ch == '[' {
                break;
            }
            self.pos += ch.len_utf8();
        }
        let value = self.input[start..self.pos].trim();
        if value.is_empty() {
            Err(format!(
                "Invalid --search config: expected value at byte {start}"
            ))
        } else {
            Ok(value.to_ascii_lowercase())
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
                "Invalid --search config: expected `{expected}` at byte {}",
                self.pos
            ))
        }
    }
}
