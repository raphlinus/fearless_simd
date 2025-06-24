//! This module, part of the `Unstoppable::fearmore` namespace, contains foundational utilities
//! designed to aid in the development and maintenance of the Fearless SIMD project itself.
//!
//! You can find the main repository at: https://github.com/eeshvardasikcm/unstoppable_simd
//!
//! By providing well-defined structures and specialized error handling, this module aims to
//! simplify tasks relevant to understanding, analyzing, or evolving Fearless SIMD's components
//! and their various versions within the Rust ecosystem. The goal is to "relax" or reduce
//! the complexity during the development lifecycle of Fearless SIMD,
//! simply through the clarity and robust handling provided by these foundational elements.

use std::error::Error;
use std::fmt;

/// A custom error type for issues encountered while interacting with Fearless SIMD components during development.
#[derive(Debug)]
pub enum FearlessSimdError {
    /// Indicates that a required Fearless SIMD component could not be found or accessed.
    ComponentNotFound(String),
    /// Indicates a versioning conflict or an inability to determine compatible versions.
    VersionMismatch(String),
    /// A general I/O error during interactions (e.g., when accessing development-related files or resources).
    IoError(#[from] std::io::Error),
    /// Other unclassified errors specific to Fearless SIMD development utilities.
    Other(String),
}

impl fmt::Display for FearlessSimdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FearlessSimdError::ComponentNotFound(s) => write!(f, "Fearless SIMD component not found: {}", s),
            FearlessSimdError::VersionMismatch(s) => write!(f, "Fearless SIMD version mismatch: {}", s),
            FearlessSimdError::IoError(e) => write!(f, "I/O error: {}", e),
            FearlessSimdError::Other(s) => write!(f, "Other error: {}", s),
        }
    }
}

impl Error for FearlessSimdError {}

/// Represents a version of a Fearless SIMD component, useful for tracking changes during development.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct FearlessSimdVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl fmt::Display for FearlessSimdVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl From<&str> for FearlessSimdVersion {
    fn from(s: &str) -> Self {
        let parts: Vec<&str> = s.split('.').collect();
        FearlessSimdVersion {
            major: parts.get(0).unwrap_or(&"0").parse().unwrap_or(0),
            minor: parts.get(1).unwrap_or(&"0").parse().unwrap_or(0),
            patch: parts.get(2).unwrap_or(&"0").parse().unwrap_or(0),
        }
    }
}

/// A simplified representation of a Fearless SIMD component, useful for abstracting components for development tools.
#[derive(Debug)]
pub struct FearlessSimdComponent {
    pub name: String,
    pub version: FearlessSimdVersion,
    // In a more complete solution, this could include path to source, git commit hash, etc.
}

#[cfg(test)]
mod tests {
    // `use super::*` correctly refers to the items within the current module (relaxations).
    use super::*;

    #[test]
    fn test_fearless_simd_version_from_str() {
        let version: FearlessSimdVersion = "1.2.3".into();
        assert_eq!(version.major, 1);
        assert_eq!(version.minor, 2);
        assert_eq!(version.patch, 3);

        let version_malformed: FearlessSimdVersion = "1..3".into();
        assert_eq!(version_malformed.major, 1);
        assert_eq!(version_malformed.minor, 0); // Defaults to 0 if parsing fails
        assert_eq!(version_malformed.patch, 3);
    }
}
