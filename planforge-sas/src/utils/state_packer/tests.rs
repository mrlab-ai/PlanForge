use super::*;

fn setup() -> StatePacker {
    let ranges = vec![100, 200, 300, 400, 500, u64::MAX];
    StatePacker::new(&ranges)
}

#[test]
fn pack_and_unpack_ints() {
    let packer = setup();
    let buffer = &mut [0u64; 6];

    // Pack some integers.
    packer.set(buffer, 0, 42);
    packer.set(buffer, 1, 84);
    packer.set(buffer, 2, 126);

    // Unpack and assert.
    assert_eq!(packer.get(buffer, 0), 42);
    assert_eq!(packer.get(buffer, 1), 84);
    assert_eq!(packer.get(buffer, 2), 126);
}

#[test]
fn pack_and_unpack_doubles() {
    let packer = setup();
    let buffer = &mut [0u64; 6];

    let double_var_id = 5;

    for double_value in [0.5, 1.0, 2.0, 4.0, std::f64::consts::PI] {
        let packed = packer.pack_double(double_value);

        packer.set(buffer, double_var_id, packed);

        let unpacked = packer.get_double(buffer, double_var_id);
        assert!(crate::utils::float_tolerance::equal(unpacked, double_value));
    }
}

#[test]
fn pack_double_canonicalizes_close_values() {
    let packer = setup();

    assert_eq!(packer.pack_double(0.1 + 0.2), packer.pack_double(0.3));
}

#[test]
fn packer_handles_single_value_domains() {
    let packer = StatePacker::new(&[1, u64::MAX]);
    let mut buffer = vec![0; packer.num_bins()];

    packer.set(&mut buffer, 0, 0);
    assert_eq!(packer.get(&buffer, 0), 0);
}

#[test]
fn values_straddle_word_boundaries_without_padding() {
    let range = 1u64 << 40;
    let packer = StatePacker::new(&[range, range, range]);
    assert_eq!(packer.num_bins(), 2);

    let mut buffer = vec![0; packer.num_bins()];
    let values = [0x0012_3456_789a, 0x0076_5432_10fe, 0x000f_edcb_a987];
    for (var, &value) in values.iter().enumerate() {
        packer.set(&mut buffer, var, value);
    }
    for (var, &value) in values.iter().enumerate() {
        assert_eq!(packer.get(&buffer, var), value);
    }

    assert!(packer.var_infos[1].is_straddling());
}

#[test]
fn a_straddling_variable_contributes_both_words_to_subset_mask() {
    let range = 1u64 << 40;
    let packer = StatePacker::new(&[range, range, range]);

    assert_eq!(
        packer.build_var_subset_mask(&[1]),
        vec![0xffff_ff00_0000_0000, 0x0000_0000_0000_ffff]
    );
}

#[test]
fn full_width_values_keep_the_single_word_fast_path() {
    let range = 1u64 << 40;
    let packer = StatePacker::new(&[range, u64::MAX, range]);

    assert_eq!(packer.var_infos[1].bin_index, 0);
    assert_eq!(packer.var_infos[1].shift, 0);
    assert!(!packer.var_infos[1].is_straddling());
}
