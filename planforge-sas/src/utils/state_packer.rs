use crate::numeric_task::{AbstractNumericTask, NumericRootTask, NumericType};
use crate::utils::float_tolerance;

#[cfg(test)]
mod tests;

const BITS_PER_BIN: u64 = (std::mem::size_of::<u64>() * 8) as u64;

fn get_bit_size_for_range(range: u64) -> u64 {
    if range <= 1 {
        return 1;
    }
    u64::from(u64::BITS - (range - 1).leading_zeros())
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SingleWordInfo {
    range: u64,
    bin_index: usize,
    shift: u64,
    read_mask: u64,
    clear_mask: u64,
}

impl SingleWordInfo {
    fn new(range: u64, bin_index: usize, shift: u64, bit_size: u64) -> Self {
        assert!(
            shift + bit_size <= BITS_PER_BIN,
            "single-word slot crosses a bin boundary"
        );
        let read_mask = get_bit_mask(shift, shift + bit_size);
        let clear_mask = !read_mask;
        Self {
            range,
            bin_index,
            shift,
            read_mask,
            clear_mask,
        }
    }

    #[inline]
    fn get(&self, buffer: &[u64]) -> u64 {
        (buffer[self.bin_index] & self.read_mask) >> self.shift
    }

    /// Bits a value written to this slot may occupy, aligned at bit zero.
    #[inline]
    fn value_mask(&self) -> u64 {
        self.read_mask >> self.shift
    }

    #[inline]
    fn set(&self, buffer: &mut [u64], value: u64) {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StraddlingInfo {
    range: u64,
    first_bin_index: usize,
    first_shift: u64,
    first_read_mask: u64,
    first_clear_mask: u64,
    second_read_mask: u64,
    second_clear_mask: u64,
    first_bit_count: u64,
    value_mask: u64,
}

impl StraddlingInfo {
    fn new(range: u64, first_bin_index: usize, first_shift: u64, bit_size: u64) -> Self {
        assert!(first_shift > 0, "a straddling slot must use both bins");
        assert!(
            first_shift + bit_size > BITS_PER_BIN,
            "a straddling slot must cross a bin boundary"
        );
        assert!(
            bit_size < BITS_PER_BIN,
            "full-width slots must be word-aligned"
        );
        let first_bit_count = BITS_PER_BIN - first_shift;
        let second_bit_count = bit_size - first_bit_count;
        let first_read_mask = get_bit_mask(first_shift, BITS_PER_BIN);
        let second_read_mask = get_bit_mask(0, second_bit_count);
        Self {
            range,
            first_bin_index,
            first_shift,
            first_read_mask,
            first_clear_mask: !first_read_mask,
            second_read_mask,
            second_clear_mask: !second_read_mask,
            first_bit_count,
            value_mask: get_bit_mask(0, bit_size),
        }
    }

    #[inline]
    fn get(&self, buffer: &[u64]) -> u64 {
        let low = (buffer[self.first_bin_index] & self.first_read_mask) >> self.first_shift;
        let high = buffer[self.first_bin_index + 1] & self.second_read_mask;
        low | (high << self.first_bit_count)
    }

    #[inline]
    fn set(&self, buffer: &mut [u64], value: u64) {
        assert_eq!(
            value & self.value_mask,
            value,
            "value {value} does not fit the {} bits packed for range {}",
            self.value_mask.count_ones(),
            self.range
        );
        let first_bin = buffer[self.first_bin_index];
        buffer[self.first_bin_index] =
            (first_bin & self.first_clear_mask) | (value << self.first_shift);
        let second_bin = buffer[self.first_bin_index + 1];
        buffer[self.first_bin_index + 1] =
            (second_bin & self.second_clear_mask) | (value >> self.first_bit_count);
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum VariableInfo {
    #[default]
    Unassigned,
    Single(SingleWordInfo),
    Straddling(StraddlingInfo),
}

impl VariableInfo {
    fn new(range: u64, bin_index: usize, shift: u64) -> Self {
        let bit_size = get_bit_size_for_range(range);
        if shift + bit_size <= BITS_PER_BIN {
            Self::Single(SingleWordInfo::new(range, bin_index, shift, bit_size))
        } else {
            Self::Straddling(StraddlingInfo::new(range, bin_index, shift, bit_size))
        }
    }

    #[inline]
    fn get(&self, buffer: &[u64]) -> u64 {
        match self {
            Self::Single(info) => info.get(buffer),
            Self::Straddling(info) => info.get(buffer),
            Self::Unassigned => unreachable!("every packed variable must have a slot"),
        }
    }

    #[inline]
    fn set(&self, buffer: &mut [u64], value: u64) {
        match self {
            Self::Single(info) => info.set(buffer, value),
            Self::Straddling(info) => info.set(buffer, value),
            Self::Unassigned => unreachable!("every packed variable must have a slot"),
        }
    }

    fn add_to_mask(&self, mask: &mut [u64]) {
        match self {
            Self::Single(info) => mask[info.bin_index] |= info.read_mask,
            Self::Straddling(info) => {
                mask[info.first_bin_index] |= info.first_read_mask;
                mask[info.first_bin_index + 1] |= info.second_read_mask;
            }
            Self::Unassigned => unreachable!("every packed variable must have a slot"),
        }
    }
}

#[derive(Clone)]
pub struct StatePacker {
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

impl StatePacker {
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
        let mut packer = StatePacker {
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
        StatePacker::with_numeric_slot_offset(&domain_sizes, numeric_slot_offset)
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
            self.var_infos[var].add_to_mask(&mut mask);
        }
        mask
    }

    fn pack_bins(&mut self, ranges: &[u64]) {
        debug_assert!(self.var_infos.is_empty());

        let num_vars = ranges.len();
        self.var_infos.resize(num_vars, VariableInfo::default());

        // Full-width numeric values are the hottest large slots and cannot share
        // a bin. Align them first so they retain the one-load fast path.
        for (var, &range) in ranges.iter().enumerate() {
            if get_bit_size_for_range(range) == BITS_PER_BIN {
                self.var_infos[var] = VariableInfo::new(range, self.num_bins, 0);
                self.num_bins += 1;
            }
        }

        // Pack every narrower slot consecutively. A slot may cross one word
        // boundary, so the only unused bits are at the end of the final bin.
        let narrow_bin_offset = self.num_bins;
        let mut narrow_bit_offset = 0usize;
        for (var, &range) in ranges.iter().enumerate() {
            let bit_size = get_bit_size_for_range(range);
            if bit_size == BITS_PER_BIN {
                continue;
            }
            let bin_index = narrow_bin_offset + narrow_bit_offset / BITS_PER_BIN as usize;
            let shift = (narrow_bit_offset % BITS_PER_BIN as usize) as u64;
            self.var_infos[var] = VariableInfo::new(range, bin_index, shift);
            narrow_bit_offset = narrow_bit_offset
                .checked_add(bit_size as usize)
                .expect("packed state bit count overflow");
        }
        if narrow_bit_offset > 0 {
            self.num_bins += narrow_bit_offset.div_ceil(BITS_PER_BIN as usize);
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
