use super::*;

fn setup() -> IntDoublePacker {
    let ranges = vec![100, 200, 300, 400, 500, u64::MAX];
    IntDoublePacker::new(&ranges)
}

#[test]
fn pack_and_unpack_ints() {
    let mut packer = setup();
    let buffer = &mut [0u64; 6];

    // Pack some integers
    packer.set(buffer, 0, 42);
    packer.set(buffer, 1, 84);
    packer.set(buffer, 2, 126);

    // Unpack and assert
    assert_eq!(packer.get(buffer, 0), 42);
    assert_eq!(packer.get(buffer, 1), 84);
    assert_eq!(packer.get(buffer, 2), 126);
}

#[test]
fn pack_and_unpack_doubles() {
    let mut packer = setup();
    let buffer = &mut [0u64; 6];

    let double_var_id = 5;

    let double_value: f64 = std::f64::consts::PI;
    let packed = packer.pack_double(double_value);

    packer.set(buffer, double_var_id, packed);

    let unpacked = packer.get_double(buffer, double_var_id);
    assert_eq!(unpacked, double_value);
}
