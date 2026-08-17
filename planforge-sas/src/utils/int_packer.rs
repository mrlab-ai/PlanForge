use crate::numeric_task::{AbstractNumericTask, NumericRootTask, NumericType};
use crate::utils::float_tolerance;

#[cfg(test)]
mod tests;

const BITS_PER_BIN: u64 = (std::mem::size_of::<u64>() * 8) as u64;

fn get_bit_size_for_range(range: u64) -> u64 {
    if range == u64::MAX {
        return BITS_PER_BIN;
    }
    if range <= 1 {
        return 1;
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

    /// Bits a value written to this slot may occupy, aligned at bit zero.
    #[inline]
    fn value_mask(&self) -> u64 {
        self.read_mask >> self.shift
    }

    #[inline]
    pub fn set(&self, buffer: &mut [u64], value: u64) {
        // The slot is exactly as wide as `range` needs, so a wider value would
        // be truncated by `read_mask` and read back as a different value. That
        // is silent data loss, so it fails here instead.
        assert_eq!(
            value & self.value_mask(),
            value,
            "value {value} does not fit the {} bits packed for range {}",
            self.read_mask.count_ones(),
            self.range
        );
        let bin_index = self.bin_index;
        let bin = buffer[bin_index];
        buffer[bin_index] = (bin & self.clear_mask) | (value << self.shift);
    }
}

#[derive(Clone)]
pub struct IntDoublePacker {
    var_infos: Vec<VariableInfo>,
    num_bins: usize,
    /// First slot of the regular-numeric section of the buffer.
    ///
    /// A concrete-state packer lays out the propositional variables first and
    /// the regular numeric variables after them, so this is the one place that
    /// decides where the numeric section starts. Packers without a numeric
    /// section report the slot count, which makes "slot below the offset" mean
    /// "propositional slot" for every packer.
    numeric_slot_offset: usize,
}

impl IntDoublePacker {
    /// Packer for `ranges` alone, with no regular-numeric section.
    pub fn new(ranges: &[u64]) -> Self {
        Self::with_numeric_slot_offset(ranges, ranges.len())
    }

    fn with_numeric_slot_offset(ranges: &[u64], numeric_slot_offset: usize) -> Self {
        assert!(
            numeric_slot_offset <= ranges.len(),
            "numeric section starts at slot {numeric_slot_offset}, past the {} packed slots",
            ranges.len()
        );
        let mut packer = IntDoublePacker {
            var_infos: vec![],
            num_bins: 0,
            numeric_slot_offset,
        };
        packer.pack_bins(ranges);
        packer
    }

    pub fn from_task(task: &NumericRootTask) -> Self {
        Self::from_abstract_task(task)
    }

    pub fn from_abstract_task(task: &dyn AbstractNumericTask) -> Self {
        Self::from_abstract_task_with_numeric_range(task, u64::MAX)
    }

    /// Packer for the *concrete* states of `task`.
    pub fn from_abstract_task_with_numeric_range(
        task: &dyn AbstractNumericTask,
        numeric_range: u64,
    ) -> Self {
        let mut domain_sizes: Vec<u64> = task
            .variables()
            .iter()
            .map(|var| var.domain_size() as u64)
            .collect();
        let numeric_slot_offset = domain_sizes.len();
        for numeric_var in task.numeric_variables().iter() {
            if numeric_var.get_type() == &NumericType::Regular {
                domain_sizes.push(numeric_range);
            }
        }
        IntDoublePacker::with_numeric_slot_offset(&domain_sizes, numeric_slot_offset)
    }

    pub fn num_bins(&self) -> usize {
        self.num_bins
    }

    /// First slot of the regular-numeric section; equivalently, the number of
    /// propositional slots. Consumers must read the boundary from here instead
    /// of recomputing it from the task, so the layout has a single definition.
    pub fn numeric_slot_offset(&self) -> usize {
        self.numeric_slot_offset
    }

    /// Build a per-bin bit mask covering the slots of the variables in
    /// `included_var_ids`. Useful for hashing/comparing buffers while ignoring
    /// the bits owned by other (e.g. axiom-derived) variables.
    pub fn build_var_subset_mask(&self, included_var_ids: &[usize]) -> Vec<u64> {
        let mut mask = vec![0u64; self.num_bins];
        for &var in included_var_ids {
            if let Some(info) = self.var_infos.get(var) {
                mask[info.bin_index] |= info.read_mask;
            }
        }
        mask
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
                unreachable!(
                    "non-empty {bits}-bit variable bucket became empty while filling bin {bin_index}"
                );
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
        float_tolerance::canonical_bits(plain_double)
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
