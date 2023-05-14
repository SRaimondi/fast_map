use std::{
    alloc::{self, Layout},
    ptr::{self, NonNull},
    slice,
};

fn round_up(a: usize, b: usize) -> usize {
    assert_ne!(b, 0);
    (a + b - 1) / b
}

/// Enum representing the result of looking for a key in a block.
#[derive(Copy, Clone)]
pub enum KeysBlockLookup {
    Found(usize),
    NotFoundStop,
    NotFoundContinue,
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

    /// Check if the block contains the given key.
    fn contains(&self, key: Self::Key) -> KeysBlockLookup;

    /// Check if the block has space.
    #[inline]
    fn has_space(&self) -> bool {
        !self.full()
    }

    /// Check if the block is full.
    fn full(&self) -> bool;

    /// Insert the given key in the block assuming there is space.
    fn insert(&mut self, key: Self::Key) -> usize;
}

/// Struct representing a block of keys where we use u32::MAX as flag for empty slots.
#[derive(Copy, Clone)]
#[cfg_attr(target_arch = "x86_64", repr(align(128)))]
pub struct U32KeysBlock {
    keys: [u32; Self::TOTAL_KEYS],
}

impl U32KeysBlock {
    /// For this case we use the mask with all bits set as en empty value since we can.
    const FREE_KEY: u32 = u32::MAX;

    /// The number of keys fits perfectly in two cache lines for x86_64 architecture.
    const TOTAL_KEYS: usize = 32;
}

impl KeysBlock for U32KeysBlock {
    type Key = u32;

    const KEYS_PER_BLOCK: usize = Self::TOTAL_KEYS;

    const EMPTY: Self = Self {
        keys: [Self::FREE_KEY; Self::TOTAL_KEYS],
    };

    #[inline]
    fn hash(key: Self::Key) -> usize {
        let mut h = key ^ (key >> 16);
        h = h.wrapping_mul(0x85EBCA6B);
        h ^= h >> 13;
        h = h.wrapping_mul(0xC2B2AE35);
        h ^= h >> 16;
        h as usize
    }

    #[inline]
    fn contains(&self, key: Self::Key) -> KeysBlockLookup {
        debug_assert_ne!(key, Self::FREE_KEY);

        for offset in 0..Self::TOTAL_KEYS {
            let offset_key = self.keys[offset];
            if offset_key == Self::FREE_KEY {
                return KeysBlockLookup::NotFoundStop;
            } else if offset_key == key {
                return KeysBlockLookup::Found(offset);
            }
        }

        KeysBlockLookup::NotFoundContinue
    }

    #[inline]
    fn full(&self) -> bool {
        self.keys.iter().all(|&key| key != Self::FREE_KEY)
    }

    #[inline]
    fn insert(&mut self, key: Self::Key) -> usize {
        debug_assert!(self.has_space());
        for i in 0..Self::TOTAL_KEYS {
            if self.keys[i] == Self::FREE_KEY {
                self.keys[i] = key;
                return i;
            }
        }
        unreachable!("the block has space so we must be able to insert");
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TryInsertError {
    OutOfSpace,
}

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
                if candidate_block.has_space() {
                    // Insert the key in the block and get the offset
                    let block_offset = candidate_block.insert(key);
                    // Now we also write the value at the corresponding index
                    let values_offset = candidate_block_ptr.offset_from(self.keys_blocks) as usize
                        * B::KEYS_PER_BLOCK
                        + block_offset;
                    self.values.add(values_offset).write(value);
                    // Increment the number of elements in the map
                    self.in_use_elements += 1;
                    // We have inserted the pair, we can exit
                    break;
                }

                // Go to the next element, wrapping at the end of the buffer
                candidate_block_ptr = candidate_block_ptr.add(1);
                if candidate_block_ptr == keys_blocks_end {
                    candidate_block_ptr = self.keys_blocks;
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

    /// Search for element in the map, returns None if the key is not present.
    #[inline]
    pub fn get(&self, key: B::Key) -> Option<&V> {
        if self.is_empty() {
            None
        } else {
            unsafe {
                let keys_block_end = self.keys_blocks.add(self.allocated_blocks);

                let mut candidate_block_ptr =
                    self.keys_blocks.add(self.compute_key_block_index(key));

                loop {
                    let candidate_block = &*candidate_block_ptr;

                    match candidate_block.contains(key) {
                        KeysBlockLookup::Found(block_offset) => {
                            let values_offset = candidate_block_ptr.offset_from(self.keys_blocks)
                                as usize
                                * B::KEYS_PER_BLOCK
                                + block_offset;
                            return Some(&*self.values.add(values_offset));
                        }
                        KeysBlockLookup::NotFoundStop => return None,
                        KeysBlockLookup::NotFoundContinue => (),
                    }

                    candidate_block_ptr = candidate_block_ptr.add(1);
                    if candidate_block_ptr == keys_block_end {
                        candidate_block_ptr = self.keys_blocks;
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

                match candidate_block.contains(key) {
                    KeysBlockLookup::Found(block_offset) => {
                        let values_offset = candidate_block_ptr.offset_from(self.keys_blocks)
                            as usize
                            * B::KEYS_PER_BLOCK
                            + block_offset;
                        return &*self.values.add(values_offset);
                    }
                    KeysBlockLookup::NotFoundStop => std::hint::unreachable_unchecked(),
                    KeysBlockLookup::NotFoundContinue => (),
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
        assert_eq!(map.max_in_use_elements, 108);

        for i in 0..100 {
            assert_eq!(*map.get(i).unwrap(), i);
            assert_eq!(unsafe { *map.get_existing(i) }, i);
        }
        for i in 100..1000 {
            assert!(map.get(i).is_none());
        }
    }
}

//     /// Get the value for the given key assuming the key is in the table.
//     pub unsafe fn get_existing(&self, key: K) -> V {
//         let keys_start = self.keys.as_ptr();
//         let keys_end = keys_start.add(self.buffer_len());
//
//         let mut candidate_key_ptr = keys_start.add(self.compute_key_index(key));
//
//         loop {
//             let candidate_key = candidate_key_ptr.read();
//             debug_assert!(!candidate_key.empty());
//             // We either found the slot with the key or we continue
//             if candidate_key.contains(key) {
//                 let offset = candidate_key_ptr.offset_from(keys_start) as usize;
//                 return self.values.get_unchecked(offset).assume_init_read();
//             }
//
//             candidate_key_ptr = candidate_key_ptr.add(1);
//             if candidate_key_ptr == keys_end {
//                 candidate_key_ptr = keys_start;
//             }
//         }
//     }
//
//     /// Call the given function over each element of the map if it's there.
//     pub fn for_each<F>(&self, mut f: F)
//     where
//         F: FnMut(K, V),
//     {
//         for i in 0..self.buffer_len() {
//             unsafe {
//                 let key = self.keys.get_unchecked(i);
//                 if !key.empty() {
//                     f(key.get(), self.values.get_unchecked(i).assume_init_read());
//                 }
//             }
//         }
//     }
// }
//
// use std::num::NonZeroU64;
// use std::ptr::NonNull;
// use std::{alloc, mem, ptr};
//
// impl KeyStorage<NonZeroU64> for Option<NonZeroU64> {
//     const EMPTY: Self = None;
//
//     #[inline]
//     fn empty(self) -> bool {
//         self.is_none()
//     }
//
//     #[inline]
//     fn write(&mut self, k: NonZeroU64) {
//         let _ = self.insert(k);
//     }
//
//     #[inline]
//     unsafe fn get(self) -> NonZeroU64 {
//         self.unwrap_unchecked()
//     }
// }
//
// impl Key for NonZeroU64 {
//     type Storage = Option<NonZeroU64>;
//
//     #[inline]
//     fn hash(self) -> usize {
//         let mut h = self.get() ^ (self.get() >> 33);
//         h = h.wrapping_mul(0xFF51AFD7ED558CCD);
//         h ^= h >> 33;
//         h = h.wrapping_mul(0xC4CEB9FE1A85EC53);
//         (h ^ (h >> 33)) as usize
//     }
// }
//
// pub fn test(map: &mut FastMap<NonZeroU64, u32>, k: NonZeroU64, i: u32) -> u32 {
//     // unsafe{ map.insert_direct(k, i) };
//     unsafe { map.get_existing(k) }
// }
