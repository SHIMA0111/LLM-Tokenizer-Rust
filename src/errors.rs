use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, PartialEq)]
pub enum CounterError {
    ModelNotFound(String),
    RegexError(String),
    KeyError(String),
    ValueError(String),
}

impl Display for CounterError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ModelNotFound(e) => write!(f, "Input model '{}' not found.", e),
            Self::RegexError(e) => write!(f, "Regex failed due to {}", e),
            Self::KeyError(e) => write!(f, "Key Error occurred from key: '{}'", e),
            Self::ValueError(e) => write!(f, "Value Error occurred due to {}", e),
        }
    }
}

impl Error for CounterError {}

pub type CounterResult<T> = Result<T, CounterError>;
