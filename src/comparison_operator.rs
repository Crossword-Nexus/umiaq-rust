use std::str::FromStr;
use crate::errors::ParseError::ParseFailure;

pub(crate) enum ComparisonOperator {
    EQ,
    NE,
    LE,
    GE,
    LT,
    GT
}

impl FromStr for ComparisonOperator {
    type Err = crate::errors::ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "=" => Ok(ComparisonOperator::EQ),
            "!=" => Ok(ComparisonOperator::NE),
            "<=" => Ok(ComparisonOperator::LE),
            ">=" => Ok(ComparisonOperator::GE),
            "<" => Ok(ComparisonOperator::LT),
            ">" => Ok(ComparisonOperator::GT),
            _ => Err(ParseFailure { s: format!("Cannot parse operator from \"{s}\"") })
        }
    }
}
