use std::{
    alloc::{self, Layout},
    ptr, slice,
};

mod keys_block;
mod keys_mask;

use keys_block::{KeySearchResult, KeysBlock, U32KeysBlock, U64KeysBlock};

/// Helper function to round up the division between a and b.
#[inline]
fn round_up(a: usize, b: usize) -> usize {
    assert_ne!(b, 0);
    (a + b - 1) / b
}

/// Enum with the only error that can happen when you try to insert and the map is full.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TryInsertError {
    OutOfSpace,
}

/// A very fast hashmap designed specifically for some use cases.
pub struct FastMap<B: KeysBlock, V: Copy> {
    buffer: *mut u8,
    keys_blocks: *mut B,
    values: *mut V,
    // Number of blocks allocated
    allocated_blocks: usize,
    // Maximum number of elements we can store in the map
    max_in_use_elements: usize,
    // Current number of elements used in the map
    in_use_elements: usize,
}

unsafe impl<B: KeysBlock, V: Copy> Send for FastMap<B, V> {}
unsafe impl<B: KeysBlock, V: Copy> Sync for FastMap<B, V> {}

impl<B: KeysBlock, V: Copy> Drop for FastMap<B, V> {
    fn drop(&mut self) {
        if self.allocated_blocks != 0 {
            unsafe { alloc::dealloc(self.buffer, Self::buffer_layout(self.allocated_blocks).0) }
        }
    }
}

impl<B: KeysBlock, V: Copy> Default for FastMap<B, V> {
    fn default() -> Self {
        Self {
            buffer: ptr::null_mut(),
            keys_blocks: ptr::null_mut(),
            values: ptr::null_mut(),
            allocated_blocks: 0,
            max_in_use_elements: 0,
            in_use_elements: 0,
        }
    }
}

impl<B: KeysBlock, V: Copy> FastMap<B, V> {
    /// Maximum load factor multiplied by 100.
    const MAX_LOAD_FACTOR_100: usize = 85;
    /// Keys block alignment.
    const B_ALIGN: usize = std::mem::align_of::<B>();
    /// Value alignment.
    const V_ALIGN: usize = std::mem::align_of::<V>();

    /// Compute layout of the buffer for storing the data.
    fn buffer_layout(total_blocks: usize) -> (Layout, usize) {
        use std::mem;

        let largest_alignment = Self::B_ALIGN.max(Self::V_ALIGN);
        let round_to_alignment =
            |size: usize| -> usize { round_up(size, largest_alignment) * largest_alignment };

        let kb_size = round_to_alignment(total_blocks * mem::size_of::<B>());
        let values_size =
            round_to_alignment(total_blocks * B::KEYS_PER_BLOCK * mem::size_of::<V>());

        (
            Layout::from_size_align(kb_size + values_size, largest_alignment)
                .expect("failed to compute map memory layout"),
            kb_size,
        )
    }

    /// Helper function to compute the maximum number of indices in use in the map.
    fn compute_max_in_use_elements(size: usize) -> usize {
        let (r, overflow) = size.overflowing_mul(Self::MAX_LOAD_FACTOR_100);
        if overflow {
            // If the multiplication did overflow, we first divide and then multiply
            (size / 100) * Self::MAX_LOAD_FACTOR_100
        } else {
            r / 100
        }
    }

    /// Compute number of blocks in the map that satisfies the capacity.
    fn blocks_for_capacity(capacity: usize) -> usize {
        // First we check if we can compute the size at all
        let r = capacity
            .checked_mul(100)
            .expect("failed to compute map capacity");
        // Compute the size that satisfies the capacity
        let min_size = round_up(r, Self::MAX_LOAD_FACTOR_100);
        // Compute the final number of blocks for the map
        round_up(min_size, B::KEYS_PER_BLOCK).next_power_of_two()
    }

    /// Helper function to compute the index of the given key.
    #[inline(always)]
    fn compute_key_block_index(&self, key: B::Key) -> usize {
        debug_assert!(self.allocated_blocks.is_power_of_two());
        B::hash(key) & (self.allocated_blocks - 1)
    }

    /// Internal helper function to insert a new element. This method assumes that there is
    /// at least one empty element in the map to insert it.
    /// Panics in debug if the key is already in the map or there is not enough space to insert.
    #[inline]
    fn insert_internal(&mut self, key: B::Key, value: V) {
        debug_assert!(self.capacity() > self.len());

        unsafe {
            let keys_blocks_end = self.keys_blocks.add(self.allocated_blocks);

            // Get a pointer to the block where we should try to insert
            let mut candidate_block_ptr = self.keys_blocks.add(self.compute_key_block_index(key));

            // Iterate until we find a block where we can insert the key
            loop {
                let candidate_block = &mut *candidate_block_ptr;

                // Check if we can insert in the block
                match candidate_block.try_insert(key) {
                    Some(block_offset) => {
                        // Compute the offset where we should insert the value
                        let values_offset = candidate_block_ptr.offset_from(self.keys_blocks)
                            as usize
                            * B::KEYS_PER_BLOCK
                            + block_offset as usize;
                        self.values.add(values_offset).write(value);
                        self.in_use_elements += 1;
                        break;
                    }
                    None => {
                        // Go to the next element, wrapping at the end of the buffer
                        candidate_block_ptr = candidate_block_ptr.add(1);
                        if candidate_block_ptr == keys_blocks_end {
                            candidate_block_ptr = self.keys_blocks;
                        }
                    }
                }
            }
        }
    }

    /// Directly insert the key and index couple in the map without checking if we need to resize.
    /// # Safety
    /// This function assumes that you know there is enough space to do the insertion.
    /// Panics in debug if that's not the case.
    pub unsafe fn insert_direct(&mut self, key: B::Key, value: V) {
        self.insert_internal(key, value);
    }

    /// Try to insert a new element without resizing the map. Returns Ok if the element can be inserted,
    /// Err if we are out of space.
    pub fn try_insert(&mut self, key: B::Key, value: V) -> Result<(), TryInsertError> {
        // Check if we are out of space on the map to insert further indices
        if self.capacity() > self.len() {
            // SAFETY: We just checked there is enough space
            unsafe { self.insert_direct(key, value) };
            Ok(())
        } else {
            Err(TryInsertError::OutOfSpace)
        }
    }

    /// Create new map with at least the required capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 {
            Self::default()
        } else {
            // Compute number of blocks for the map
            let total_blocks = Self::blocks_for_capacity(capacity);

            // Compute the number of elements we can use in the map
            let max_in_use_elements =
                Self::compute_max_in_use_elements(total_blocks * B::KEYS_PER_BLOCK);
            debug_assert!(max_in_use_elements >= capacity);

            let (buffer_layout, kb_size) = Self::buffer_layout(total_blocks);

            unsafe {
                let buffer = alloc::alloc(buffer_layout);
                if buffer.is_null() {
                    alloc::handle_alloc_error(buffer_layout);
                }

                let keys_blocks = buffer as *mut B;
                // Set all the blocks to empty
                slice::from_raw_parts_mut(keys_blocks, total_blocks).fill(B::default());

                let values = buffer.add(kb_size) as *mut V;

                Self {
                    buffer,
                    keys_blocks,
                    values,
                    allocated_blocks: total_blocks,
                    in_use_elements: 0,
                    max_in_use_elements,
                }
            }
        }
    }

    /// Create new map from the given iterator of couples.
    pub fn from_exact_iter(it: impl ExactSizeIterator<Item = (B::Key, V)>) -> Self {
        let mut map = Self::with_capacity(it.len());
        it.for_each(|(key, value)| unsafe { map.insert_direct(key, value) });
        map
    }

    /// Look for the given key in the map, returns Some with a reference to the value if found, otherwise None.
    pub fn get(&self, key: B::Key) -> Option<&V> {
        unsafe {
            let keys_block_end = self.keys_blocks.add(self.allocated_blocks);

            let mut candidate_block_ptr = self.keys_blocks.add(self.compute_key_block_index(key));

            loop {
                let candidate_block = &*candidate_block_ptr;
                match candidate_block.search(key) {
                    KeySearchResult::Found(block_offset) => {
                        let values_offset = candidate_block_ptr.offset_from(self.keys_blocks)
                            as usize
                            * B::KEYS_PER_BLOCK
                            + block_offset as usize;
                        return Some(&*self.values.add(values_offset));
                    }
                    KeySearchResult::NotFoundStop => return None,
                    KeySearchResult::NotFoundContinue => {
                        candidate_block_ptr = candidate_block_ptr.add(1);
                        if candidate_block_ptr == keys_block_end {
                            candidate_block_ptr = self.keys_blocks;
                        }
                    }
                }
            }
        }
    }

    /// Get reference to the value for the given key assuming it exists.
    /// # Safety
    /// The function assumes the key will be found in the map, it's UB to use it with a key that is
    /// not in the map.
    pub unsafe fn get_existing(&self, key: B::Key) -> &V {
        let keys_block_end = self.keys_blocks.add(self.allocated_blocks);

        let mut candidate_block_ptr = self.keys_blocks.add(self.compute_key_block_index(key));

        loop {
            let candidate_block = &*candidate_block_ptr;
            match candidate_block.search(key) {
                KeySearchResult::Found(block_offset) => {
                    let values_offset = candidate_block_ptr.offset_from(self.keys_blocks) as usize
                        * B::KEYS_PER_BLOCK
                        + block_offset as usize;
                    return &*self.values.add(values_offset);
                }
                KeySearchResult::NotFoundStop => std::hint::unreachable_unchecked(),
                KeySearchResult::NotFoundContinue => {
                    candidate_block_ptr = candidate_block_ptr.add(1);
                    if candidate_block_ptr == keys_block_end {
                        candidate_block_ptr = self.keys_blocks;
                    }
                }
            }
        }
    }

    /// Execute the given function on each element of the map.
    pub fn for_each<F>(&self, mut f: F)
    where
        F: FnMut(B::Key, V),
    {
        unsafe {
            let mut values_offset = 0;
            slice::from_raw_parts(self.keys_blocks, self.allocated_blocks)
                .iter()
                .for_each(|block| {
                    for key_index in 0..block.total_keys() {
                        f(
                            block.get_key(key_index),
                            self.values.add(values_offset + key_index as usize).read(),
                        );
                    }
                    values_offset += B::KEYS_PER_BLOCK;
                });
        }
    }

    /// Return the capacity of the map, the number of elements that can be added before resizing.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.max_in_use_elements
    }

    /// Current number of indices in the map.
    #[inline]
    pub fn len(&self) -> usize {
        self.in_use_elements
    }

    /// Check if the map is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub type FastMapU32<V> = FastMap<U32KeysBlock, V>;
pub type FastMapU64<V> = FastMap<U64KeysBlock, V>;

#[cfg(test)]
mod tests {
    use metrohash::MetroHashSet;
    use rand::distributions::{Distribution, Uniform};

    use super::*;

    #[test]
    fn test_creation_u32() {
        for capacity in 0..(1 << 12) {
            let mut map = FastMapU32::<u32>::with_capacity(capacity);
            for i in 0..capacity {
                map.try_insert(i as u32, i as u32).unwrap();
            }
            assert_eq!(map.len(), capacity);
            assert!(map.len() <= map.capacity());

            for i in 0..capacity {
                assert_eq!(unsafe { *map.get_existing(i as u32) }, i as u32);
            }
        }
    }

    #[test]
    fn test_creation_u64() {
        for capacity in 0..(1 << 12) {
            let mut map = FastMapU64::<u32>::with_capacity(capacity);
            for i in 0..capacity {
                map.try_insert((i + 1) as u64, i as u32).unwrap();
            }
            assert_eq!(map.len(), capacity);
            assert!(map.len() <= map.capacity());

            for i in 0..capacity {
                assert_eq!(unsafe { *map.get_existing((i + 1) as u64) }, i as u32);
            }
        }
    }

    #[test]
    fn test_get_u32() {
        const N: usize = 100_000;
        let mut added = MetroHashSet::default();

        let interval = Uniform::new(0, 3 * N as u32);
        let mut rng = rand::thread_rng();

        let mut map = FastMapU32::<u32>::with_capacity(N);
        while map.len() < N {
            let i = interval.sample(&mut rng);
            if !added.contains(&i) {
                added.insert(i);
                map.try_insert(i, i).unwrap();
            }
        }

        const TESTS: usize = 200_000;
        for _ in 0..TESTS {
            let i = interval.sample(&mut rng);
            if added.contains(&i) {
                assert_eq!(*map.get(i).unwrap(), i);
                assert_eq!(unsafe { *map.get_existing(i) }, i);
            } else {
                assert!(map.get(i).is_none());
            }
        }
    }

    #[test]
    fn test_get_u64() {
        const N: usize = 100_000;
        let mut added = MetroHashSet::default();

        let interval = Uniform::new(0, 3 * N as u32);
        let mut rng = rand::thread_rng();

        let mut map = FastMapU64::<u32>::with_capacity(N);
        while map.len() < N {
            let i = interval.sample(&mut rng);
            if !added.contains(&i) {
                added.insert(i);
                map.try_insert(i as u64, i).unwrap();
            }
        }

        const TESTS: usize = 200_000;
        for _ in 0..TESTS {
            let i = interval.sample(&mut rng);
            if added.contains(&i) {
                assert_eq!(*map.get(i as u64).unwrap(), i);
                assert_eq!(unsafe { *map.get_existing(i as u64) }, i);
            } else {
                assert!(map.get(i as u64).is_none());
            }
        }
    }

    #[test]
    fn test_for_each() {
        let mut map = FastMapU32::with_capacity(100);
        for i in 0..100 {
            map.try_insert(i, i).unwrap();
        }

        map.for_each(|key, value| assert_eq!(key, value));
    }
}
