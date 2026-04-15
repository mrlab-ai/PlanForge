use crate::numeric::numeric_task::{AbstractNumericTask, NumericRootTask, NumericType};

#[cfg(test)]
mod tests;

const BITS_PER_BIN: u64 = (std::mem::size_of::<u64>() * 8) as u64;

fn get_bit_size_for_range(range: u64) -> u64 {
    if range == u64::MAX {
        return BITS_PER_BIN;
    }
    let mut num_bits = 0;
    while 1 << num_bits < range {
        num_bits += 1;
    }

    num_bits
}

fn get_bit_mask(from: u64, to: u64) -> u64 {
    debug_assert!(to >= from);
    debug_assert!(to <= BITS_PER_BIN);
    let length = to - from;
    if length == BITS_PER_BIN {
        debug_assert!(from == 0 && to == BITS_PER_BIN);
        return !0;
    }
    ((1 << length) - 1) << from
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct VariableInfo {
    range: u64,
    bin_index: usize,
    shift: u64,
    read_mask: u64,
    clear_mask: u64,
}

impl VariableInfo {
    pub fn new(range: u64, bin_index: usize, shift: u64) -> Self {
        let bit_size = get_bit_size_for_range(range);
        let read_mask = get_bit_mask(shift, shift + bit_size);
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
        (buffer[self.bin_index] & self.read_mask) >> self.shift
    }

    pub fn set(&self, buffer: &mut [u64], value: u64) {
        let bin_index = self.bin_index;
        let bin = buffer[bin_index];
        buffer[bin_index] = (bin & self.clear_mask) | ((value << self.shift) & self.read_mask);
    }
}

pub struct IntDoublePacker {
    var_infos: Vec<VariableInfo>,
    num_bins: usize,
}

impl IntDoublePacker {
    pub fn new(ranges: &[u64]) -> Self {
        let mut packer = IntDoublePacker {
            var_infos: vec![],
            num_bins: 0,
        };
        packer.pack_bins(ranges);
        packer
    }

    pub fn from_task(task: &NumericRootTask) -> Self {
        Self::from_abstract_task(task)
    }

    pub fn from_abstract_task(task: &dyn AbstractNumericTask) -> Self {
        let mut domain_sizes = vec![];
        for var in task.variables().iter() {
            domain_sizes.push(var.domain_size() as u64);
        }
        for numeric_var in task.numeric_variables().iter() {
            if numeric_var.get_type() == &NumericType::Regular {
                domain_sizes.push(u64::MAX);
            }
        }
        IntDoublePacker::new(&domain_sizes)
    }

    pub fn num_bins(&self) -> usize {
        self.num_bins
    }

    fn pack_one_bin(&mut self, ranges: &[u64], bits_to_var: &mut [Vec<usize>]) -> usize {
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

            // Get mutable reference to the best-fit list.
            let best_fit_vars = &mut bits_to_var[bits as usize];

            // Pop the last variable index if available
            if let Some(var) = best_fit_vars.pop() {
                self.var_infos[var] = VariableInfo::new(ranges[var], bin_index, used_bits);
                used_bits += bits;
                num_vars_in_bin += 1;
            } else {
                // This shouldn't happen because of the `is_empty()` check above
                eprintln!(
                    "Unexpected: no variable with {} bits available for bin {}",
                    bits, bin_index
                );
                return num_vars_in_bin;
            }
        }
    }

    fn pack_bins(&mut self, ranges: &[u64]) {
        debug_assert!(self.var_infos.is_empty());

        let num_vars = ranges.len();
        self.var_infos.resize(num_vars, VariableInfo::default());

        let mut bits_to_var: Vec<Vec<usize>> = vec![vec![]; (BITS_PER_BIN + 1) as usize];

        for var in (0..num_vars).rev() {
            let bits = get_bit_size_for_range(ranges[var]);
            debug_assert!(bits <= BITS_PER_BIN);
            bits_to_var[bits as usize].push(var);
        }

        let mut packed_vars = 0;
        while packed_vars < num_vars {
            let num_vars_in_bin = self.pack_one_bin(ranges, &mut bits_to_var);
            packed_vars += num_vars_in_bin;
        }
    }

    pub fn pack_double(&self, plain_double: f64) -> u64 {
        plain_double.to_bits()
    }

    pub fn unpack_double(&self, packed_double: u64) -> f64 {
        f64::from_bits(packed_double)
    }

    pub fn get_double(&self, buffer: &[u64], var: usize) -> f64 {
        let packed_double = self.var_infos[var].get(buffer);
        self.unpack_double(packed_double)
    }

    pub fn get(&self, buffer: &[u64], var: usize) -> u64 {
        self.var_infos[var].get(buffer)
    }

    pub fn set(&self, buffer: &mut [u64], var: usize, value: u64) {
        self.var_infos[var].set(buffer, value);
    }
}
