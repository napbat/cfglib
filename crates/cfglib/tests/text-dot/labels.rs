//! Small-scope exhaustive label generation: an escaping bug needs only a
//! short string over the right few characters, so checking every short string
//! over an adversarial alphabet beats sampling.

/// Every string of at most `max_length` characters over `alphabet`, the empty
/// string first, shorter strings before longer ones.
pub fn strings_over(alphabet: &[char], max_length: usize) -> Vec<String> {
    let mut all = vec![String::new()];
    let mut layer = all.clone();
    for _ in 0..max_length {
        layer = layer
            .iter()
            .flat_map(|prefix| alphabet.iter().map(move |ch| format!("{prefix}{ch}")))
            .collect();
        all.extend(layer.iter().cloned());
    }
    all
}

/// The alphabet that attacks the line-oriented text grammar: the escape
/// character and the letter its escapes use, line breaks of both kinds, the
/// arrow's hyphen and `>`, a comment marker, and a space.
pub const TEXT_ALPHABET: [char; 9] = ['a', ' ', '-', '\\', 'n', '\n', '\r', '>', '#'];

/// The alphabet that attacks DOT quoting: quote, backslash, the letters DOT's
/// own escapes use (`\n`, `\l`), a real line break, and a space.
pub const DOT_ALPHABET: [char; 7] = ['a', '"', '\\', '\n', 'l', 'n', ' '];
