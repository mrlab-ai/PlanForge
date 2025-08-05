const BITS_PER_BIN: i32 = (std::mem::size_of::<u64>() * 8) as i32;

fn get_bit_size_for_range(range: u64) -> i32 {
    if range == u64::MAX {
        return BITS_PER_BIN;
    }
    let mut num_bits = 0;
    while 1u64 << num_bits < range {
        num_bits += 1;
    }

    num_bits
}

fn get_bit_mask(from: i32, to: i32) -> u64 {
    assert!(from >= 0);
    assert!(to >= from);
    assert!(to <= BITS_PER_BIN);
    let length = to - from;
    if length == BITS_PER_BIN {
        assert!(from == 0 && to == BITS_PER_BIN);
        return !0u64;
    }
    ((1 << length) - 1) << from
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VariableInfo {
    range: u64,
    bin_index: i32,
    shift: i32,
    read_mask: u64,
    clear_mask: u64,
}

impl Default for VariableInfo {
    fn default() -> Self {
        VariableInfo {
            range: 0,
            bin_index: -1,
            shift: 0,
            read_mask: 0,
            clear_mask: 0,
        }
    }
}

impl VariableInfo {
    pub fn new(range: u64, bin_index: i32, shift: i32) -> Self {
        let bit_size = get_bit_size_for_range(range);
        let read_mask = get_bit_mask(0, bit_size as i32);
        let clear_mask = !read_mask;
        VariableInfo {
            range,
            bin_index,
            shift,
            read_mask,
            clear_mask,
        }
    }

    pub fn get(&self, buffer: &[u64]) -> u64 {
        (buffer[self.bin_index as usize] & self.read_mask) >> self.shift
    }

    fn set(&self, buffer: &mut [u64], value: u64) {
        let bin_index = self.bin_index as usize;
        let bin = buffer[bin_index];
        buffer[bin_index] = (bin & self.clear_mask) | (value << self.shift);
    }

    //NOTE: associated functions
}

pub struct IntDoublePacker {
    var_infos: Vec<VariableInfo>,
    num_bins: i32,
}

impl IntDoublePacker {
    pub fn new(ranges: Vec<u64>) -> Self {

        let mut packer = IntDoublePacker { var_infos: vec![], num_bins: 0 };
        packer.pack_bins(&ranges);
        packer
    }

    pub fn num_bins(&self) -> i32 {
        self.num_bins
    }

    fn pack_one_bin(&mut self, ranges: &Vec<u64>, bits_to_var: &Vec<Vec<i32>>) -> i32 {
        self.num_bins += 1;
        let bin_index = self.num_bins - 1;
        let mut used_bits = 0;
        let mut num_vars_in_bin = 0;

        loop {
            let mut bits = BITS_PER_BIN - used_bits;
            while bits > 0 && bits_to_var[bits as usize].is_empty() {
                bits -= 1;
            }
            if bits == 0 {
                return num_vars_in_bin;
            }
            let best_fit_vars = &bits_to_var[bits as usize];
            let var = best_fit_vars.last().unwrap(); //TODO: Replace the unwrap with proper error handling
            self.var_infos[*var as usize] = VariableInfo::new(
                ranges[*var as usize],
                bin_index,
                used_bits
            );
            used_bits += bits;
            num_vars_in_bin += 1;
        }
    }

    fn pack_bins(&mut self, ranges: &Vec<u64>) {
        assert!(self.var_infos.is_empty());

        let num_vars = ranges.len();
        self.var_infos.resize(num_vars, VariableInfo::default());

        let mut bits_to_var: Vec<Vec<i32>> = vec![vec![]; (BITS_PER_BIN + 1) as usize];

        for var in (0..num_vars).rev() {
            let bits = get_bit_size_for_range(ranges[var]);
            assert!(bits <= BITS_PER_BIN);
            bits_to_var[bits as usize].push(var as i32);
        }

        let mut packed_vars: i32 = 0;
        while packed_vars < (num_vars as i32) {
            let num_vars_in_bin = self.pack_one_bin(ranges, &bits_to_var);
            packed_vars += num_vars_in_bin;
        }
    }

    pub fn pack_double(&self, plain_double: f64) -> u64 {
        plain_double.to_bits()
    }

    pub fn unpack_double(&self, packed_double: u64) -> f64 {
        f64::from_bits(packed_double)
    }

    pub fn get_double(&self, buffer: &[u64], var: i32, var_infos: Vec<VariableInfo>) -> f64 {
        let packed_double = var_infos[var as usize].get(buffer);
        self.unpack_double(packed_double)
    }

    pub fn get(&self, buffer: &[u64], var: i32) -> u64 {
        self.var_infos[var as usize].get(buffer)
    }

    pub fn set(&mut self, buffer: &mut [u64], var: i32, value: u64) {
        self.var_infos[var as usize].set(buffer, value);
    }
}
