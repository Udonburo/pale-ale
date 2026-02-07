use crate::blake3_hex;
use serde::Serialize;

#[derive(Serialize)]
struct CanonicalInputs<'a> {
    v: u8,
    query: &'a str,
    context: &'a str,
    answer: &'a str,
}

pub fn compute_inputs_hash(query: &str, context: &str, answer: &str) -> String {
    let normalized_query = normalize_newlines(query);
    let normalized_context = normalize_newlines(context);
    let normalized_answer = normalize_newlines(answer);
    let canonical = CanonicalInputs {
        v: 1,
        query: &normalized_query,
        context: &normalized_context,
        answer: &normalized_answer,
    };
    let bytes = serde_jcs::to_vec(&canonical).expect("JCS serialization failed for inputs");
    blake3_hex(&bytes)
}

fn normalize_newlines(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut iter = input.chars().peekable();
    while let Some(ch) = iter.next() {
        if ch == '\r' {
            if iter.peek() == Some(&'\n') {
                iter.next();
            }
            out.push('\n');
        } else {
            out.push(ch);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inputs_hash_stability() {
        let hash = compute_inputs_hash("Q", "C", "A");
        assert_eq!(
            hash,
            "feb305b14d23dbd498034fcac4c04aa2db6018a3282217c1020fe996539d7011"
        );
    }

    #[test]
    fn newline_normalization() {
        let a = compute_inputs_hash("q\r\nx", "c\ry", "a\r\nz");
        let b = compute_inputs_hash("q\nx", "c\ny", "a\nz");
        assert_eq!(a, b);
    }

    #[test]
    fn jcs_ordering_stability() {
        let normalized_query = normalize_newlines("qq");
        let normalized_context = normalize_newlines("cc");
        let normalized_answer = normalize_newlines("aa");
        let bytes = serde_jcs::to_vec(&CanonicalInputs {
            v: 1,
            query: &normalized_query,
            context: &normalized_context,
            answer: &normalized_answer,
        })
        .expect("jcs");
        let expected = blake3_hex(&bytes);
        let actual = compute_inputs_hash("qq", "cc", "aa");
        assert_eq!(actual, expected);
    }
}
