use std::{
    alloc::{self, Layout},
    num::NonZeroU32,
    ptr::{self, NonNull},
    slice,
};

fn round_up(a: usize, b: usize) -> usize {
    assert_ne!(b, 0);
    (a + b - 1) / b
}

/// Trait exposing the methods a block of key should have.
pub trait KeysBlock: Copy {
    /// Type of the key
    type Key: Copy;

    /// Number of keys in a block.
    const KEYS_PER_BLOCK: usize;

    /// Constant representing an empty block.
    const EMPTY: Self;

    /// Hash the given key.
    fn hash(key: Self::Key) -> usize;

    /// Get the offset in the block where the key is or None if the block is full but without the key.
    fn get(&self, key: Self::Key) -> Option<usize>;

    /// Try to insert the given key in the block, returns the offset if successful.
    fn try_insert(&mut self, key: Self::Key) -> Option<usize>;
}

#[derive(Copy, Clone, Default)]
#[repr(transparent)]
struct KeysMask<const IN_USE_BITS: u32> {
    mask: u32,
}

impl<const IN_USE_BITS: u32> KeysMask<IN_USE_BITS> {
    const EMPTY: Self = Self { mask: 0 };

    /// Set the bit for the given index to 1.
    #[inline(always)]
    fn set_in_use(&mut self, index: u32) {
        debug_assert_eq!((self.mask >> index) & 1, 0);
        debug_assert!(index < IN_USE_BITS);
        self.mask |= 1 << index;
    }

    /// Check if the chunk is full.
    #[inline(always)]
    fn is_full(self) -> bool {
        self.mask == (1 << IN_USE_BITS) - 1
    }

    /// Check if there is still space in the chunk.
    #[inline(always)]
    fn has_space(self) -> bool {
        !self.is_full()
    }
}

/// Struct representing a block of keys where we use u32::MAX as flag for empty slots.
#[derive(Copy, Clone)]
#[cfg_attr(target_arch = "x86_64", repr(align(128), C))]
pub struct U32KeysBlock {
    keys_mask: KeysMask<31>,
    keys: [u32; Self::TOTAL_KEYS],
}

impl U32KeysBlock {
    /// For this case we use the mask with all bits set as en empty value since we can.
    const FREE_KEY: u32 = u32::MAX;

    /// The number of keys fits perfectly in two cache lines for x86_64 architecture.
    const TOTAL_KEYS: usize = 31;

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn simd_chunk(&self, chunk: usize) -> std::arch::x86_64::__m256i {
        use std::arch::x86_64::{_mm256_castps_si256, _mm256_load_ps};
        let ptr: *const u32 = &self.keys_mask.mask;
        _mm256_castps_si256(_mm256_load_ps((ptr as *const f32).add(8 * chunk)))
    }
}

impl KeysBlock for U32KeysBlock {
    type Key = u32;

    const KEYS_PER_BLOCK: usize = Self::TOTAL_KEYS;

    const EMPTY: Self = Self {
        keys_mask: KeysMask::EMPTY,
        keys: [Self::FREE_KEY; Self::TOTAL_KEYS],
    };

    #[inline(always)]
    fn hash(key: Self::Key) -> usize {
        // This is quite "simple" but the benchmark tells that this is working very well
        key as usize
    }

    #[inline]
    fn get(&self, key: Self::Key) -> Option<usize> {
        debug_assert_ne!(key, Self::FREE_KEY);
        debug_assert!(!self.keys_mask.has_space() || self.keys.iter().any(|&k| k == key));

        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{
                _mm256_castsi256_ps, _mm256_cmpeq_epi32, _mm256_movemask_ps, _mm256_set1_epi32,
            };

            let key_splat = _mm256_set1_epi32(key as i32);

            let chunk_has_key_mask = |chunk: usize| -> u32 {
                let has_key = _mm256_cmpeq_epi32(key_splat, self.simd_chunk(chunk));
                let has_key_mask = _mm256_movemask_ps(_mm256_castsi256_ps(has_key)) as u32;
                debug_assert!(has_key_mask <= 1 << 7);
                debug_assert!(has_key_mask == 0 || has_key_mask.is_power_of_two());
                has_key_mask << (8 * chunk)
            };

            let has_key_mask = (chunk_has_key_mask(0)
                | chunk_has_key_mask(1)
                | chunk_has_key_mask(2)
                | chunk_has_key_mask(3))
                >> 1;

            (has_key_mask != 0)
                .then_some(NonZeroU32::new_unchecked(has_key_mask).trailing_zeros() as usize)
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            for i in 0..self.keys.len() {
                let k = self.keys[i];
                debug_assert_ne!(k, Self::FREE_KEY);
                if k == key {
                    return Some(i);
                }
            }

            None
        }
    }

    #[inline]
    fn try_insert(&mut self, key: Self::Key) -> Option<usize> {
        debug_assert_ne!(key, Self::FREE_KEY);
        debug_assert!(self.keys.iter().all(|&k| k != key));

        if self.keys_mask.is_full() {
            return None;
        }

        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{
                _mm256_castsi256_ps, _mm256_cmpeq_epi32, _mm256_movemask_ps, _mm256_set1_epi32,
            };

            let chunk_has_empty_mask = |chunk: usize| -> u32 {
                let empty_key_splat = _mm256_set1_epi32(Self::FREE_KEY as i32);
                let has_empty = _mm256_cmpeq_epi32(self.simd_chunk(chunk), empty_key_splat);
                let has_empty_mask = _mm256_movemask_ps(_mm256_castsi256_ps(has_empty));
                debug_assert!(has_empty_mask <= u8::MAX as i32);
                (has_empty_mask as u32) << (8 * chunk)
            };

            let empty_slot_mask = (chunk_has_empty_mask(0)
                | chunk_has_empty_mask(1)
                | chunk_has_empty_mask(2)
                | chunk_has_empty_mask(3))
                >> 1;

            let offset = NonZeroU32::new_unchecked(empty_slot_mask).trailing_zeros();
            debug_assert_eq!(self.keys[offset as usize], Self::FREE_KEY);

            self.keys_mask.set_in_use(offset);
            self.keys[offset as usize] = key;

            Some(offset as usize)
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            for i in 0..self.keys.len() {
                if self.keys[i] == Self::FREE_KEY {
                    self.keys[i] = key;
                    return Some(i);
                }
            }

            None
        }
    }
}

/// Enum with the only error that can happen when you try to insert and the map is full.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TryInsertError {
    OutOfSpace,
}

/// A very fast hashmap designed specifically for some use cases.
pub struct FastMap<B: KeysBlock, V: Copy> {
    buffer: NonNull<u8>,
    keys_blocks: *mut B,
    values: *mut V,
    // Number of blocks allocated
    allocated_blocks: usize,
    // Maximum number of elements we can store in the map
    max_in_use_elements: usize,
    // Current number of elements used in the map
    in_use_elements: usize,
}

impl<B: KeysBlock, V: Copy> Drop for FastMap<B, V> {
    fn drop(&mut self) {
        unsafe {
            alloc::dealloc(
                self.buffer.as_ptr(),
                Self::buffer_layout(self.allocated_blocks).0,
            )
        }
    }
}

impl<B: KeysBlock, V: Copy> Default for FastMap<B, V> {
    fn default() -> Self {
        Self {
            buffer: NonNull::dangling(),
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
                            + block_offset;
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
                let buffer = match NonNull::new(alloc::alloc(buffer_layout)) {
                    Some(b) => b,
                    None => alloc::handle_alloc_error(buffer_layout),
                };

                let keys_blocks = buffer.as_ptr() as *mut B;
                // Set all the blocks to empty
                slice::from_raw_parts_mut(keys_blocks, total_blocks).fill(B::EMPTY);

                let values = buffer.as_ptr().add(kb_size) as *mut V;

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

    /// Get reference to the value for the given key assuming it exists.
    /// # Safety
    /// The function assumes the key will be found in the map, it's UB to use it with a key that is
    /// not in the map.
    pub unsafe fn get_existing(&self, key: B::Key) -> &V {
        unsafe {
            let keys_block_end = self.keys_blocks.add(self.allocated_blocks);

            let mut candidate_block_ptr = self.keys_blocks.add(self.compute_key_block_index(key));

            loop {
                let candidate_block = &*candidate_block_ptr;

                if let Some(block_offset) = candidate_block.get(key) {
                    let values_offset = candidate_block_ptr.offset_from(self.keys_blocks) as usize
                        * B::KEYS_PER_BLOCK
                        + block_offset;
                    return &*self.values.add(values_offset);
                }

                candidate_block_ptr = candidate_block_ptr.add(1);
                if candidate_block_ptr == keys_block_end {
                    candidate_block_ptr = self.keys_blocks;
                }
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation() {
        let mut map = FastMap::<U32KeysBlock, u32>::with_capacity(100);

        for i in 0..100 {
            map.try_insert(i, i).unwrap();
        }

        assert_eq!(map.len(), 100);
        assert_eq!(map.max_in_use_elements, 105);

        for i in 0..100 {
            assert_eq!(unsafe { *map.get_existing(i) }, i);
        }
    }
}
