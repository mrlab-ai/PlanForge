//! PDDL lisp parser
//! Port of python/translate/pddl_parser/lisp_parser.py

use std::str::Chars;
use std::iter::Peekable;
use super::SExpr;

pub struct LispParser {
    chars: Peekable<Chars<'static>>,
}

impl LispParser {
    pub fn new(input: &'static str) -> Self {
        Self {
            chars: input.chars().peekable(),
        }
    }

    pub fn parse(&mut self) -> Result<SExpr, String> {
        self.skip_whitespace();
        
        if self.chars.peek() == Some(&'(') {
            self.parse_list()
        } else {
            self.parse_atom()
        }
    }

    fn parse_list(&mut self) -> Result<SExpr, String> {
        self.chars.next(); // consume '('
        let mut elements = Vec::new();
        
        loop {
            self.skip_whitespace();
            
            if self.chars.peek() == Some(&')') {
                self.chars.next(); // consume ')'
                break;
            }
            
            if self.chars.peek().is_none() {
                return Err("Unexpected end of input in list".to_string());
            }
            
            elements.push(self.parse()?);
        }
        
        Ok(SExpr::List(elements))
    }

    fn parse_atom(&mut self) -> Result<SExpr, String> {
        let mut atom = String::new();
        
        while let Some(&ch) = self.chars.peek() {
            if ch.is_whitespace() || ch == '(' || ch == ')' {
                break;
            }
            atom.push(ch);
            self.chars.next();
        }
        
        if atom.is_empty() {
            Err("Empty atom".to_string())
        } else {
            Ok(SExpr::Atom(atom.to_lowercase()))  // Convert to lowercase to match Python behavior
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(&ch) = self.chars.peek() {
            if ch.is_whitespace() {
                self.chars.next();
            } else if ch == ';' {
                // Skip comment line
                while let Some(&ch) = self.chars.peek() {
                    self.chars.next();
                    if ch == '\n' {
                        break;
                    }
                }
            } else {
                break;
            }
        }
    }
}

pub fn parse_lisp(input: &'static str) -> Result<Vec<SExpr>, String> {
    let mut parser = LispParser::new(input);
    let mut results = Vec::new();
    
    while parser.chars.peek().is_some() {
        parser.skip_whitespace();
        if parser.chars.peek().is_some() {
            results.push(parser.parse()?);
        }
    }
    
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_atom() {
        let result = parse_lisp("hello").unwrap();
        assert_eq!(result, vec![SExpr::Atom("hello".to_string())]);
    }

    #[test]
    fn test_parse_simple_list() {
        let result = parse_lisp("(define domain)").unwrap();
        assert_eq!(result, vec![SExpr::List(vec![
            SExpr::Atom("define".to_string()),
            SExpr::Atom("domain".to_string())
        ])]);
    }
}
