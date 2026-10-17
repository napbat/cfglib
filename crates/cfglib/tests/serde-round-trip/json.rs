//! The three things every persistence test does with a value.

use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

pub fn encode<T: Serialize>(value: &T) -> String {
    serde_json::to_string(value).expect("the value serializes")
}

pub fn decode<T: DeserializeOwned>(json: &str) -> T {
    serde_json::from_str(json).expect("the document loads")
}

pub fn round_trip<T: Serialize + DeserializeOwned>(value: &T) -> T {
    decode(&encode(value))
}

/// The document as a tree, for tests that edit a document by hand.
pub fn document<T: Serialize>(value: &T) -> Value {
    serde_json::to_value(value).expect("the value serializes")
}

/// Load an edited document.
pub fn load<T: DeserializeOwned>(document: Value) -> Result<T, serde_json::Error> {
    serde_json::from_value(document)
}
