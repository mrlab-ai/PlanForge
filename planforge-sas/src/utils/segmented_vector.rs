use std::cmp;

const SEGMENT_BYTES: usize = 8192;

type SegmentAndOffset = (usize, usize);

#[derive(Debug)]
pub struct SegmentedArrayVector<T: Clone> {
    elements_per_array: usize,
    segment_shift: u32,
    segment_mask: usize,
    elements_per_segment: usize,
    // Flat arrays per segment.
    pub segments: Vec<Box<[T]>>,
    // Number of elements (each of size `elements_per_array`).
    size: usize,
}

impl<T: Clone + Default> SegmentedArrayVector<T> {
    pub fn new(elements_per_array: usize) -> Self {
        debug_assert!(elements_per_array > 0);
        let max_arrays_per_segment = cmp::max(
            SEGMENT_BYTES / (elements_per_array * std::mem::size_of::<T>()),
            1,
        );
        // A power-of-two count turns the hot index quotient and remainder
        // into a shift and a mask while keeping every ordinary segment within
        // SEGMENT_BYTES.
        let arrays_per_segment = 1usize << max_arrays_per_segment.ilog2();
        let elements_per_segment = elements_per_array * arrays_per_segment;
        Self {
            elements_per_array,
            segment_shift: arrays_per_segment.trailing_zeros(),
            segment_mask: arrays_per_segment - 1,
            elements_per_segment,
            segments: Vec::new(),
            size: 0,
        }
    }

    #[inline]
    fn get_segment_index(&self, index: usize) -> SegmentAndOffset {
        let segment = index >> self.segment_shift;
        let offset = (index & self.segment_mask) * self.elements_per_array;
        (segment, offset)
    }

    fn add_segment(&mut self) {
        // Allocate new zeroed segment of size elements_per_segment.
        let new_segment = vec![T::default(); self.elements_per_segment].into_boxed_slice();
        self.segments.push(new_segment);
    }

    pub fn push_back(&mut self, entry: &[T]) {
        debug_assert_eq!(entry.len(), self.elements_per_array);
        let (segment, offset) = self.get_segment_index(self.size);
        if segment == self.segments.len() {
            self.add_segment();
        }
        self.segments[segment][offset..offset + self.elements_per_array].clone_from_slice(entry);
        self.size += 1;
    }

    pub fn push_copy(&mut self, index: usize) {
        debug_assert!(index < self.size);
        let (source_segment, source_offset) = self.get_segment_index(index);
        let source_ptr = self.segments[source_segment][source_offset..].as_ptr();

        let (dest_segment, dest_offset) = self.get_segment_index(self.size);
        if dest_segment == self.segments.len() {
            self.add_segment();
        }

        let dest_ptr = self.segments[dest_segment][dest_offset..].as_mut_ptr();
        for offset in 0..self.elements_per_array {
            unsafe {
                *dest_ptr.add(offset) = (*source_ptr.add(offset)).clone();
            }
        }
        self.size += 1;
    }

    pub fn pop_back(&mut self) {
        debug_assert!(self.size > 0);
        self.size -= 1;
        // No actual deallocation; reuse on next push_back.
    }

    pub fn resize(&mut self, new_size: usize, entry: &[T]) {
        debug_assert_eq!(entry.len(), self.elements_per_array);
        while self.size < new_size {
            self.push_back(entry);
        }
        while self.size > new_size {
            self.pop_back();
        }
    }

    pub fn get(&self, index: usize) -> Option<&[T]> {
        if index >= self.size {
            return None;
        }
        let (segment, offset) = self.get_segment_index(index);
        Some(&self.segments[segment][offset..offset + self.elements_per_array])
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut [T]> {
        if index >= self.size {
            return None;
        }
        let (segment, offset) = self.get_segment_index(index);
        Some(&mut self.segments[segment][offset..offset + self.elements_per_array])
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

#[cfg(test)]
mod tests {
    use super::SegmentedArrayVector;

    #[test]
    fn indexing_crosses_a_power_of_two_segment_boundary() {
        let mut values = SegmentedArrayVector::<u32>::new(3);
        let arrays_per_segment = values.segment_mask + 1;

        for value in 0..=arrays_per_segment {
            values.push_back(&[value as u32; 3]);
        }

        assert!(arrays_per_segment.is_power_of_two());
        assert_eq!(values.segments.len(), 2);
        let last_in_first = (arrays_per_segment - 1) as u32;
        let first_in_second = arrays_per_segment as u32;
        assert_eq!(
            values.get(arrays_per_segment - 1),
            Some(&[last_in_first; 3][..])
        );
        assert_eq!(
            values.get(arrays_per_segment),
            Some(&[first_in_second; 3][..])
        );
    }
}
