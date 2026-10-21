use std::fmt;

/// Invalid configuration, with the calls, options and list indices leading to it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError {
    path: Vec<String>,
    message: String,
    rendered: String,
    leading_option: bool,
}

impl ConfigError {
    pub fn path(&self) -> &[String] {
        &self.path
    }
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Flattened configurations delegate the same option without adding a
    /// second path segment for it.
    pub fn at_option(self, key: &str) -> Self {
        if self.leading_option && self.path.first().is_some_and(|first| first == key) {
            return self;
        }
        let mut error = self.at(key);
        error.leading_option = true;
        error
    }

    pub fn at(mut self, segment: impl Into<String>) -> Self {
        self.leading_option = false;
        self.path.insert(0, segment.into());
        let mut path = String::new();
        for segment in &self.path {
            if !path.is_empty() && !segment.starts_with('[') {
                path.push('.');
            }
            path.push_str(segment);
        }
        self.rendered = format!("{path}: {}", self.message);
        self
    }
}
impl From<String> for ConfigError {
    fn from(message: String) -> Self {
        Self {
            rendered: message.clone(),
            leading_option: false,
            message,
            path: Vec::new(),
        }
    }
}
impl From<&str> for ConfigError {
    fn from(message: &str) -> Self {
        message.to_owned().into()
    }
}
impl From<ConfigError> for String {
    fn from(error: ConfigError) -> Self {
        error.rendered
    }
}
impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.rendered)
    }
}
impl std::error::Error for ConfigError {}
impl std::ops::Deref for ConfigError {
    type Target = str;
    fn deref(&self) -> &str {
        &self.rendered
    }
}
impl PartialEq<&str> for ConfigError {
    fn eq(&self, other: &&str) -> bool {
        self.rendered == *other
    }
}
