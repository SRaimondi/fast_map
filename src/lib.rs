use std::{
    alloc::{self, Layout},
    num::{NonZeroU16, NonZeroU32},
    ptr::{self, NonNull},
    slice,
};

/// Helper function to round up the division between a and b.
#[inline]
fn round_up(a: usize, b: usize) -> usize {
    assert_ne!(b, 0);
    (a + b - 1) / b
}

/// Helper struct representing a bitset that we use to indicate what keys are used and what not.
#[derive(Copy, Clone, Default)]
#[repr(transparent)]
struct KeysMask<const IN_USE_BITS: usize> {
    mask: u32,
}

impl<const IN_USE_BITS: usize> KeysMask<IN_USE_BITS> {
    /// Add key to the mask and return the index where it's inserted.
    #[inline(always)]
    fn add(&mut self) -> u32 {
        let index = self.mask.trailing_ones();
        debug_assert!(index < IN_USE_BITS as u32);
        self.mask |= 1 << index;
        index
    }

    /// Count how many keys are in use.
    #[inline(always)]
    fn in_use_keys(self) -> u32 {
        self.mask.trailing_ones()
    }

    /// Check if the chunk is full.
    #[inline(always)]
    fn is_full(self) -> bool {
        self.mask == (1 << IN_USE_BITS) - 1
    }

    /// Check if the block is empty.
    #[inline(always)]
    fn is_empty(self) -> bool {
        self.mask == 0
    }

    /// Check if there is still space in the chunk.
    #[inline(always)]
    fn has_space(self) -> bool {
        !self.is_full()
    }
}

/// Helper function to insert the given key in the keys with the given mask.
#[inline(always)]
fn insert_with_mask<const N: usize, T: Copy>(
    keys_mask: &mut KeysMask<N>,
    keys: &mut [T; N],
    key: T,
) -> Option<usize> {
    if keys_mask.is_full() {
        None
    } else {
        let free_slot_index = keys_mask.free_slot_index();

        keys_mask.set_in_use(free_slot_index);
        unsafe { *keys.get_unchecked_mut(free_slot_index as usize) = key };
        Some(free_slot_index as usize)
    }
}

/// Helper enum for the result of searching the given key in the block.
#[derive(Copy, Clone)]
pub enum KeySearchResult {
    Found(usize),
    NotFoundStop,
    NotFoundContinue,
}

/// Trait exposing the methods a block of key should have.
pub trait KeysBlock: Copy + Default {
    /// Type of the key
    type Key: Copy;

    /// Number of keys in a block.
    const KEYS_PER_BLOCK: usize;

    /// Hash the given key.
    fn hash(key: Self::Key) -> usize;

    /// Get the offset in the block where the key is or None if the block is full but without the key.
    fn get(&self, key: Self::Key) -> Option<usize>;

    /// Search for the given key in the block.
    fn search(&self, key: Self::Key) -> KeySearchResult;

    /// Try to insert the given key in the block, returns the offset if successful.
    fn try_insert(&mut self, key: Self::Key) -> Option<usize>;
}

/// Struct representing a block of keys where we use u32::MAX as flag for empty slots.
#[derive(Copy, Clone, Default)]
#[cfg_attr(target_arch = "x86_64", repr(align(128), C))]
pub struct U32KeysBlock {
    in_use_keys_mask: KeysMask<{ Self::TOTAL_KEYS }>,
    keys: [u32; Self::TOTAL_KEYS],
}

impl U32KeysBlock {
    /// The number of keys fits perfectly in two cache lines for x86_64 architecture.
    const TOTAL_KEYS: usize = 31;

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn simd_chunk(&self, chunk: usize) -> std::arch::x86_64::__m256i {
        use std::arch::x86_64::{_mm256_castps_si256, _mm256_load_ps};
        let ptr: *const u32 = &self.in_use_keys_mask.mask;
        _mm256_castps_si256(_mm256_load_ps((ptr as *const f32).add(8 * chunk)))
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn has_key_mask(&self, key: u32) -> u32 {
        use std::arch::x86_64::{
            _mm256_castsi256_ps, _mm256_cmpeq_epi32, _mm256_movemask_ps, _mm256_set1_epi32,
        };

        let key_splat = _mm256_set1_epi32(key as i32);

        let chunk_has_key_mask = |chunk: usize| -> u32 {
            let has_key = _mm256_cmpeq_epi32(key_splat, self.simd_chunk(chunk));
            let has_key_mask = _mm256_movemask_ps(_mm256_castsi256_ps(has_key)) as u32;
            has_key_mask << (8 * chunk)
        };

        // First we compose the mask by putting in a u32 all the bits
        let mut has_key_mask = chunk_has_key_mask(0)
            | chunk_has_key_mask(1)
            | chunk_has_key_mask(2)
            | chunk_has_key_mask(3);
        // Next we shift right by one since the first element is the mask
        has_key_mask >>= 1;
        // Finally we & each bit with the mask of the keys in use to avoid taking into account wrong elements
        has_key_mask & self.in_use_keys_mask.mask
    }
}

impl KeysBlock for U32KeysBlock {
    type Key = u32;

    const KEYS_PER_BLOCK: usize = Self::TOTAL_KEYS;

    #[inline(always)]
    fn hash(key: Self::Key) -> usize {
        // This is quite "simple" but the benchmark tells that this is working very well
        key as usize
    }

    #[inline(always)]
    fn get(&self, key: Self::Key) -> Option<usize> {
        #[cfg(target_arch = "x86_64")]
        {
            let has_key_mask = unsafe { self.has_key_mask(key) };

            debug_assert!(
                has_key_mask != 0 || self.in_use_keys_mask.is_full(),
                "if the key is not in the block, it must be full"
            );

            (has_key_mask != 0).then_some(
                unsafe { NonZeroU32::new_unchecked(has_key_mask) }.trailing_zeros() as usize,
            )
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            find_in_keys(&self.keys, key, Self::FREE_KEY)
        }
    }

    #[inline(always)]
    fn search(&self, key: Self::Key) -> KeySearchResult {
        // If the block is empty we can stop immediately
        if self.in_use_keys_mask.is_empty() {
            return KeySearchResult::NotFoundStop;
        }

        #[cfg(target_arch = "x86_64")]
        {
            let has_key_mask = unsafe { self.has_key_mask(key) };

            if has_key_mask != 0 {
                KeySearchResult::Found(
                    unsafe { NonZeroU32::new_unchecked(has_key_mask) }.trailing_zeros() as usize,
                )
            } else if self.in_use_keys_mask.has_space() {
                KeySearchResult::NotFoundStop
            } else {
                KeySearchResult::NotFoundContinue
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            for i in 0..self.keys_mask.in_use_keys() {
                debug_assert!((self.keys_mask.mask >> i) & 1 == 1);
                if unsafe { *self.keys.get_unchecked(i as usize) == key } {
                    return KeySearchResult::Found(i as usize);
                }
            }

            // If we get here, the key was not found
            if self.keys_mask.has_space() {
                KeySearchResult::NotFoundStop
            } else {
                KeySearchResult::NotFoundContinue
            }
        }
    }

    #[inline(always)]
    fn try_insert(&mut self, key: Self::Key) -> Option<usize> {
        insert_with_mask(&mut self.in_use_keys_mask, &mut self.keys, key)
    }
}

#[derive(Copy, Clone, Default)]
#[cfg_attr(target_arch = "x86_64", repr(align(128), C))]
pub struct U64KeysBlock {
    keys_mask: KeysMask<15>,
    _padding: u32,
    keys: [u64; Self::TOTAL_KEYS],
}

impl U64KeysBlock {
    const TOTAL_KEYS: usize = 15;

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn simd_chunk(&self, chunk: usize) -> std::arch::x86_64::__m256i {
        use std::arch::x86_64::{_mm256_castpd_si256, _mm256_load_pd};
        let ptr: *const u32 = &self.keys_mask.mask;
        _mm256_castpd_si256(_mm256_load_pd((ptr as *const f64).add(4 * chunk)))
    }
}

// impl KeysBlock for U64KeysBlock {
//     type Key = u64;
//
//     const KEYS_PER_BLOCK: usize = Self::TOTAL_KEYS;
//
//     #[inline(always)]
//     fn hash(key: Self::Key) -> usize {
//         key as usize
//     }
//
//     #[inline(always)]
//     fn get(&self, key: Self::Key) -> Option<usize> {
//         debug_assert!(!self.keys_mask.has_space() || self.keys.iter().any(|&k| k == key));
//
//         #[cfg(target_arch = "x86_64")]
//         unsafe {
//             use std::arch::x86_64::{
//                 _mm256_castsi256_pd, _mm256_cmpeq_epi64, _mm256_movemask_pd, _mm256_set1_epi64x,
//             };
//
//             let key_splat = _mm256_set1_epi64x(key as i64);
//
//             let chunk_has_key_mask = |chunk: usize| -> u16 {
//                 let has_key = _mm256_cmpeq_epi64(key_splat, self.simd_chunk(chunk));
//                 let has_key_mask = _mm256_movemask_pd(_mm256_castsi256_pd(has_key)) as u16;
//                 has_key_mask << (4 * chunk)
//             };
//
//             let has_key_mask = (chunk_has_key_mask(0)
//                 | chunk_has_key_mask(1)
//                 | chunk_has_key_mask(2)
//                 | chunk_has_key_mask(3))
//                 >> 1;
//
//             (has_key_mask != 0)
//                 .then_some(NonZeroU16::new_unchecked(has_key_mask).trailing_zeros() as usize)
//         }
//
//         #[cfg(not(target_arch = "x86_64"))]
//         {
//             find_in_keys(&self.keys, key, Self::FREE_KEY)
//         }
//     }
//
//     #[inline(always)]
//     fn try_insert(&mut self, key: Self::Key) -> Option<usize> {
//         insert_with_mask(&mut self.keys_mask, &mut self.keys, key)
//     }
// }

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
        if self.allocated_blocks != 0 {
            unsafe {
                alloc::dealloc(
                    self.buffer.as_ptr(),
                    Self::buffer_layout(self.allocated_blocks).0,
                )
            }
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
                slice::from_raw_parts_mut(keys_blocks, total_blocks).fill(B::default());

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
                            + block_offset;
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
    use metrohash::MetroHashSet;
    use rand::distributions::{Distribution, Uniform};

    use super::*;

    #[test]
    fn test_creation_u32() {
        for capacity in 0..(1 << 12) {
            let mut map = FastMap::<U32KeysBlock, u32>::with_capacity(capacity);
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

    // #[test]
    // fn test_creation_u64() {
    //     for capacity in 0..(1 << 12) {
    //         let mut map = FastMap::<U64KeysBlock, u32>::with_capacity(capacity);
    //         for i in 0..capacity {
    //             map.try_insert((i + 1) as u64, i as u32).unwrap();
    //         }
    //         assert_eq!(map.len(), capacity);
    //         assert!(map.len() <= map.capacity());
    //
    //         for i in 0..capacity {
    //             assert_eq!(unsafe { *map.get_existing((i + 1) as u64) }, i as u32);
    //         }
    //     }
    // }

    #[test]
    fn test_get_u32() {
        const N: usize = 100_000;
        let mut added = MetroHashSet::default();

        let interval = Uniform::new(0, 3 * N as u32);
        let mut rng = rand::thread_rng();

        let mut map = FastMap::<U32KeysBlock, u32>::with_capacity(N);
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
}
