use nom::bits;


const BITS_PER_BIN: i32 = (std::mem::size_of::<u64>() * 8) as i32;
struct VariableInfo {
    range: u64, 
    bin_index: i32,
    shift: i32,
    read_mask: u64, 
    clear_mask: u64,
}

impl VariableInfo {
    pub fn new(range: u64, bin_index: i32, shift: i32) -> Self {

        let bit_size = Self::get_bit_size_for_range(range);
        let read_mask = Self::get_bit_mask(0, bit_size as i32);
        let clear_mask = !read_mask;
        VariableInfo {
            range,
            bin_index,
            shift,
            read_mask,
            clear_mask, 
        }
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

    pub fn get_double(&self, buffer: &[u64], var: i32, var_infos: Vec<VariableInfo>) -> f64 {
        let packed_double = var_infos[var as usize].get(buffer);
        self.unpack_double(packed_double)
    }

    fn unpack_double(&self, packed_double: u64) -> f64 {
        f64::from_bits(packed_double)
    }

    pub fn get(&self, buffer: &[u64]) -> u64 {
        (buffer[self.bin_index as usize] & self.read_mask) >> self.shift
    }

    fn set(&self, buffer: &mut [u64], value: u64) {
        // Assertions are often a good idea here to make sure 'value'
        // fits within the range.
        // assert!(value < self.range);
        let bin_index = self.bin_index as usize;
        let bin = buffer[bin_index];
        buffer[bin_index] = (bin & self.clear_mask) | (value << self.shift);
    }


    //NOTE: associated functions

    fn get_bit_size_for_range(range: u64) -> i32 {
        if range == u64::MAX {
            return BITS_PER_BIN;
        } 
        let mut num_bits = 0;
        while (1u64 << num_bits) < range {
            num_bits += 1;
        }

        num_bits
    }

}

struct IntDoublePacker {
    car_infos: Vec<VariableInfo>,
    num_bins: i32,
    
}


impl IntDoublePacker {
    pub fn new(num_bins: i32, car_infos: Vec<VariableInfo>) -> Self {
        IntDoublePacker { car_infos, num_bins }
    }

    fn pack_one_bin(ranges: &Vec<u64>, bits_to_var: &Vec<Vec<i32>>) -> i32 {
        todo!()
    }

    fn pack_bins(ranges: &Vec<u64>) {
        todo!()
    }


    
}