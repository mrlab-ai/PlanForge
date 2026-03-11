//NOTE: This code is unsafe. Once we know it does not cause seg faults, we can replace the safe but slow implementation.

use std::alloc::{ alloc, dealloc, Layout };
use std::ptr::{ self, NonNull };
use std::ops::{ Index, IndexMut };

const SEGMENT_BYTES: usize = 8192;

pub struct SegmentedVector<T> {
    segments: Vec<NonNull<T>>,
    the_size: usize,
    segment_elements: usize,
    element_layout: Layout,
}

impl<T> SegmentedVector<T> {
    pub fn new() -> Self {
        let element_size = std::mem::size_of::<T>();
        debug_assert!(
            element_size > 0,
            "SegmentedVector cannot be used with zero-sized types"
        );
        if element_size == 0 {
            panic!("SegmentedVector cannot be used with zero-sized types");
        }
        let segment_elements = (SEGMENT_BYTES / element_size).max(1);
        let element_layout = Layout::array::<T>(segment_elements).unwrap();

        Self {
            segments: Vec::new(),
            the_size: 0,
            segment_elements,
            element_layout,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let mut s = Self::new();
        s.reserve(capacity);
        s
    }

    fn get_segment_index(&self, index: usize) -> usize {
        index / self.segment_elements
    }

    fn get_offset_in_segment(&self, index: usize) -> usize {
        index % self.segment_elements
    }

    fn add_segment(&mut self) {
        let new_segment_ptr = unsafe { alloc(self.element_layout) as *mut T };
        if new_segment_ptr.is_null() {
            std::alloc::handle_alloc_error(self.element_layout);
        }
        self.segments.push(NonNull::new(new_segment_ptr).unwrap());
    }

    pub fn push_back(&mut self, entry: T) {
        let segment_idx = self.get_segment_index(self.the_size);
        let offset_in_segment = self.get_offset_in_segment(self.the_size);

        if segment_idx == self.segments.len() {
            self.add_segment();
        }

        unsafe {
            let segment_ptr = self.segments[segment_idx].as_ptr();
            ptr::write(segment_ptr.add(offset_in_segment), entry);
        }
        self.the_size += 1;
    }

    pub fn pop_back(&mut self) -> Option<T> {
        if self.the_size == 0 {
            return None;
        }

        self.the_size -= 1;
        let segment_idx = self.get_segment_index(self.the_size);
        let offset_in_segment = self.get_offset_in_segment(self.the_size);

        unsafe {
            let segment_ptr = self.segments[segment_idx].as_ptr();
            Some(ptr::read(segment_ptr.add(offset_in_segment)))
        }
    }

    pub fn size(&self) -> usize {
        self.the_size
    }

    pub fn is_empty(&self) -> bool {
        self.the_size == 0
    }

    pub fn capacity(&self) -> usize {
        self.segments.len() * self.segment_elements
    }

    pub fn reserve(&mut self, additional: usize) {
        let required_capacity = self.the_size + additional;
        let current_capacity = self.capacity();

        if required_capacity > current_capacity {
            let segments_needed =
                (required_capacity + self.segment_elements - 1) / self.segment_elements;
            let segments_to_add = segments_needed.saturating_sub(self.segments.len());
            for _ in 0..segments_to_add {
                self.add_segment();
            }
        }
    }

    pub fn resize(&mut self, new_size: usize, value: T) where T: Clone {
        while new_size < self.the_size {
            self.pop_back();
        }
        while new_size > self.the_size {
            self.push_back(value.clone());
        }
    }
}

impl<T> Drop for SegmentedVector<T> {
    fn drop(&mut self) {
        for i in 0..self.the_size {
            let segment_idx = self.get_segment_index(i);
            let offset_in_segment = self.get_offset_in_segment(i);
            unsafe {
                let segment_ptr = self.segments[segment_idx].as_ptr();
                ptr::drop_in_place(segment_ptr.add(offset_in_segment));
            }
        }

        for segment_ptr in &self.segments {
            unsafe {
                dealloc(segment_ptr.as_ptr() as *mut u8, self.element_layout);
            }
        }
    }
}

impl<T> Index<usize> for SegmentedVector<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index < self.the_size, "index out of bounds");
        if index >= self.the_size {
            panic!("index out of bounds");
        }
        let segment_idx = self.get_segment_index(index);
        let offset_in_segment = self.get_offset_in_segment(index);
        unsafe {
            let segment_ptr = self.segments[segment_idx].as_ptr();
            &*segment_ptr.add(offset_in_segment)
        }
    }
}

impl<T> IndexMut<usize> for SegmentedVector<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        debug_assert!(index < self.the_size, "index out of bounds");
        if index >= self.the_size {
            panic!("index out of bounds");
        }
        let segment_idx = self.get_segment_index(index);
        let offset_in_segment = self.get_offset_in_segment(index);
        unsafe {
            let segment_ptr = self.segments[segment_idx].as_ptr();
            &mut *segment_ptr.add(offset_in_segment)
        }
    }
}

pub struct SegmentedArrayVector<T> {
    segments: Vec<NonNull<T>>,
    the_size: usize,
    elements_per_array: usize,
    arrays_per_segment: usize,
    elements_per_segment: usize,
    element_layout: Layout,
}

impl<T> SegmentedArrayVector<T> {
    pub fn new(elements_per_array: usize) -> Self {
        debug_assert!(
            elements_per_array > 0,
            "elements_per_array must be greater than 0"
        );
        if elements_per_array == 0 {
            panic!("elements_per_array must be greater than 0");
        }
        let element_size = std::mem::size_of::<T>();
        debug_assert!(
            element_size > 0,
            "SegmentedArrayVector cannot be used with zero-sized types"
        );
        if element_size == 0 {
            panic!("SegmentedArrayVector cannot be used with zero-sized types");
        }

        let arrays_per_segment = (SEGMENT_BYTES / (elements_per_array * element_size)).max(1);
        let elements_per_segment = elements_per_array * arrays_per_segment;
        let element_layout = Layout::array::<T>(elements_per_segment).unwrap();

        Self {
            segments: Vec::new(),
            the_size: 0,
            elements_per_array,
            arrays_per_segment,
            elements_per_segment,
            element_layout,
        }
    }

    pub fn with_capacity(elements_per_array: usize, capacity: usize) -> Self {
        let mut s = Self::new(elements_per_array);
        s.reserve(capacity);
        s
    }

    fn get_segment_index(&self, index: usize) -> usize {
        index / self.arrays_per_segment
    }

    fn get_offset_in_segment(&self, index: usize) -> usize {
        (index % self.arrays_per_segment) * self.elements_per_array
    }

    fn add_segment(&mut self) {
        let new_segment_ptr = unsafe { alloc(self.element_layout) as *mut T };
        if new_segment_ptr.is_null() {
            std::alloc::handle_alloc_error(self.element_layout);
        }
        self.segments.push(NonNull::new(new_segment_ptr).unwrap());
    }

    pub fn push_back(&mut self, entry_array: &[T]) {
        debug_assert_eq!(
            entry_array.len(),
            self.elements_per_array,
            "input array must match elements_per_array"
        );
        if entry_array.len() != self.elements_per_array {
            panic!("input array must match elements_per_array");
        }

        let segment_idx = self.get_segment_index(self.the_size);
        let offset_in_segment = self.get_offset_in_segment(self.the_size);

        if segment_idx == self.segments.len() {
            self.add_segment();
        }

        unsafe {
            let segment_ptr = self.segments[segment_idx].as_ptr();
            let dest_ptr = segment_ptr.add(offset_in_segment);
            ptr::copy_nonoverlapping(entry_array.as_ptr(), dest_ptr, self.elements_per_array);
        }
        self.the_size += 1;
    }

    pub fn pop_back(&mut self) {
        debug_assert!(self.the_size > 0, "pop_back on empty SegmentedArrayVector");
        if self.the_size == 0 {
            panic!("pop_back on empty SegmentedArrayVector");
        }

        self.the_size -= 1;
        let segment_idx = self.get_segment_index(self.the_size);
        let offset_in_segment = self.get_offset_in_segment(self.the_size);

        unsafe {
            let segment_ptr = self.segments[segment_idx].as_ptr();
            let element_ptr = segment_ptr.add(offset_in_segment);
            for i in 0..self.elements_per_array {
                ptr::drop_in_place(element_ptr.add(i));
            }
        }
    }

    pub fn size(&self) -> usize {
        self.the_size
    }

    pub fn is_empty(&self) -> bool {
        self.the_size == 0
    }

    pub fn capacity(&self) -> usize {
        self.segments.len() * self.arrays_per_segment
    }

    pub fn reserve(&mut self, additional: usize) {
        let required_capacity = self.the_size + additional;
        let current_capacity = self.capacity();

        if required_capacity > current_capacity {
            let segments_needed =
                (required_capacity + self.arrays_per_segment - 1) / self.arrays_per_segment;
            let segments_to_add = segments_needed.saturating_sub(self.segments.len());
            for _ in 0..segments_to_add {
                self.add_segment();
            }
        }
    }

    pub fn resize(&mut self, new_size: usize, value_array: &[T]) where T: Clone {
        while new_size < self.the_size {
            self.pop_back();
        }
        while new_size > self.the_size {
            self.push_back(value_array);
        }
    }
}

impl<T> Drop for SegmentedArrayVector<T> {
    fn drop(&mut self) {
        for i in 0..self.the_size {
            let segment_idx = self.get_segment_index(i);
            let offset_in_segment = self.get_offset_in_segment(i);
            unsafe {
                let segment_ptr = self.segments[segment_idx].as_ptr();
                let array_ptr = segment_ptr.add(offset_in_segment);
                for j in 0..self.elements_per_array {
                    ptr::drop_in_place(array_ptr.add(j));
                }
            }
        }

        for segment_ptr in &self.segments {
            unsafe {
                dealloc(segment_ptr.as_ptr() as *mut u8, self.element_layout);
            }
        }
    }
}

impl<T> Index<usize> for SegmentedArrayVector<T> {
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index < self.the_size, "index out of bounds");
        if index >= self.the_size {
            panic!("index out of bounds");
        }
        let segment_idx = self.get_segment_index(index);
        let offset_in_segment = self.get_offset_in_segment(index);
        unsafe {
            let segment_ptr = self.segments[segment_idx].as_ptr();
            std::slice::from_raw_parts(segment_ptr.add(offset_in_segment), self.elements_per_array)
        }
    }
}

impl<T> IndexMut<usize> for SegmentedArrayVector<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        debug_assert!(index < self.the_size, "index out of bounds");
        if index >= self.the_size {
            panic!("index out of bounds");
        }
        let segment_idx = self.get_segment_index(index);
        let offset_in_segment = self.get_offset_in_segment(index);
        unsafe {
            let segment_ptr = self.segments[segment_idx].as_ptr();
            std::slice::from_raw_parts_mut(
                segment_ptr.add(offset_in_segment),
                self.elements_per_array
            )
        }
    }
}
