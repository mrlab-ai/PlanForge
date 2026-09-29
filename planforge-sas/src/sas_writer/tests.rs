use super::*;
use crate::numeric_parser::parse_sas_parts;

/// The writer is the reader's inverse, byte for byte.
///
/// That is what makes the file an interoperability surface rather than a
/// dump: a task read from a file and written back out is the same file, so
/// the two halves cannot drift apart on a field one of them spells
/// differently.
fn assert_round_trips(sas_text: &str, what: &str) {
    let (rest, parts) =
        parse_sas_parts(sas_text).unwrap_or_else(|error| panic!("{what} parses: {error}"));
    assert_eq!(rest, "", "{what} was not read to its end");

    let mut written: Vec<u8> = Vec::new();
    write_sas(&parts, &mut written).expect("writing into a `Vec` cannot fail");
    let written = String::from_utf8(written).expect("the writer emits UTF-8");
    assert_eq!(written, sas_text, "{what} changed on the way back out");
}

#[test]
fn the_checked_in_sas_tasks_survive_a_round_trip() {
    let assets =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/assets/numeric_sas");
    for fixture in ["example2.sas", "example5.sas"] {
        let path = assets.join(fixture);
        let sas_text = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
        assert_round_trips(&sas_text, fixture);
    }
}
