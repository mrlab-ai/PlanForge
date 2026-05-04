use super::*;

fn setup() -> IntDoublePacker {
    let ranges = vec![100, 200, 300, 400, 500, u64::MAX];
    IntDoublePacker::new(&ranges)
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
        assert!(crate::numeric::utils::float_tolerance::equal(
            unpacked,
            double_value
        ));
    }
}

#[test]
fn pack_double_canonicalizes_close_values() {
    let packer = setup();

    assert_eq!(packer.pack_double(0.1 + 0.2), packer.pack_double(0.3));
}

#[test]
fn packer_handles_single_value_domains() {
    let packer = IntDoublePacker::new(&[1, u64::MAX]);
    let mut buffer = vec![0; packer.num_bins()];

    packer.set(&mut buffer, 0, 0);
    assert_eq!(packer.get(&buffer, 0), 0);
}
