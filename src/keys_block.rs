use crate::keys_mask::{U32KeysMask, U64KeysMask};

use std::{mem::MaybeUninit, num::NonZeroU32};

/// Struct storing a block of u32 keys.
#[derive(Copy, Clone)]
#[cfg_attr(target_arch = "x86_64", repr(align(128), C))]
pub struct U32KeysBlock {
    keys_mask: U32KeysMask,
    keys: [MaybeUninit<u32>; U32KeysMask::IN_USE_BITS as usize],
}

impl Default for U32KeysBlock {
    #[inline(always)]
    fn default() -> Self {
        Self {
            keys_mask: U32KeysMask::default(),
            keys: [MaybeUninit::uninit(); U32KeysMask::IN_USE_BITS as usize],
        }
    }
}

impl U32KeysBlock {
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn has_key_mask(&self, key: u32) -> u32 {
        use std::arch::x86_64::{
            _mm256_castps_si256, _mm256_castsi256_ps, _mm256_cmpeq_epi32, _mm256_load_ps,
            _mm256_movemask_ps, _mm256_set1_epi32,
        };

        const SIMD_WIDTH: usize = 8;
        let key_splat = _mm256_set1_epi32(key as i32);

        let chunk_has_key_mask = |chunk_index: usize| -> u32 {
            let chunk = _mm256_castps_si256(_mm256_load_ps(
                (self.keys_mask.as_ptr() as *const f32).add(SIMD_WIDTH * chunk_index),
            ));
            let has_key = _mm256_cmpeq_epi32(key_splat, chunk);
            let has_key_mask = _mm256_movemask_ps(_mm256_castsi256_ps(has_key)) as u32;
            has_key_mask << (SIMD_WIDTH * chunk_index)
        };

        // First we compose the mask by putting in a u32 all the bits
        let mut has_key_mask = chunk_has_key_mask(0)
            | chunk_has_key_mask(1)
            | chunk_has_key_mask(2)
            | chunk_has_key_mask(3);
        // Next we shift right by one since the first element is the mask
        has_key_mask >>= 1;
        // Finally we & each bit with the mask of the keys in use to avoid taking into account wrong elements
        has_key_mask & self.keys_mask.bits()
    }
}

/// Struct storing a block of u64 keys.
#[derive(Copy, Clone)]
#[cfg_attr(target_arch = "x86_64", repr(align(128), C))]
pub struct U64KeysBlock {
    keys_mask: U64KeysMask,
    _padding: u32,
    keys: [MaybeUninit<u64>; U64KeysMask::IN_USE_BITS as usize],
}

impl Default for U64KeysBlock {
    #[inline(always)]
    fn default() -> Self {
        Self {
            keys_mask: U64KeysMask::default(),
            _padding: 0,
            keys: [MaybeUninit::uninit(); U64KeysMask::IN_USE_BITS as usize],
        }
    }
}

impl U64KeysBlock {
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn has_key_mask(&self, key: u64) -> u32 {
        use std::arch::x86_64::{
            _mm256_castpd_si256, _mm256_castsi256_pd, _mm256_cmpeq_epi64, _mm256_load_pd,
            _mm256_movemask_pd, _mm256_set1_epi64x,
        };

        const SIMD_WIDTH: usize = 4;
        let key_splat = _mm256_set1_epi64x(key as i64);

        let chunk_has_key_mask = |chunk_index: usize| -> u32 {
            let chunk = _mm256_castpd_si256(_mm256_load_pd(
                (self.keys_mask.as_ptr() as *const f64).add(SIMD_WIDTH * chunk_index),
            ));
            let has_key = _mm256_cmpeq_epi64(key_splat, chunk);
            let has_key_mask = _mm256_movemask_pd(_mm256_castsi256_pd(has_key)) as u32;
            has_key_mask << (SIMD_WIDTH * chunk_index)
        };

        // First we compose the mask by putting in a u32 all the bits
        let mut has_key_mask = chunk_has_key_mask(0)
            | chunk_has_key_mask(1)
            | chunk_has_key_mask(2)
            | chunk_has_key_mask(3);
        // Next we shift right by one since the first element is the mask
        has_key_mask >>= 1;
        // Finally we & each bit with the mask of the keys in use to avoid taking into account wrong elements
        has_key_mask & self.keys_mask.bits()
    }
}

/// Helper enum for the result of searching the given key in the block.
#[derive(Copy, Clone)]
pub enum KeySearchResult {
    Found(u32),
    NotFoundStop,
    NotFoundContinue,
}

/// Trait representing the common behaviour that both blocks should have for the map.
pub trait KeysBlock: Copy + Default {
    /// Type of the key in the block.
    type Key: Copy;

    /// Number of keys in each block.
    const KEYS_PER_BLOCK: usize;

    /// Hash the given key to ket an index for it.
    fn hash(key: Self::Key) -> usize;

    /// Search for a key in the block.
    fn search(&self, key: Self::Key) -> KeySearchResult;

    /// Try to insert a key in the block, returns None if the block is full or the insertion offset.
    fn try_insert(&mut self, key: Self::Key) -> Option<u32>;
}

macro_rules! impl_block {
    ($block_name:ident, $key:ty, $mask:ty) => {
        impl KeysBlock for $block_name {
            type Key = $key;

            const KEYS_PER_BLOCK: usize = <$mask>::IN_USE_BITS as usize;

            #[inline(always)]
            fn hash(key: Self::Key) -> usize {
                key as usize
            }

            #[inline(always)]
            fn search(&self, key: Self::Key) -> KeySearchResult {
                #[cfg(target_arch = "x86_64")]
                {
                    let has_key_mask = unsafe { self.has_key_mask(key) };
                    if has_key_mask != 0 {
                        KeySearchResult::Found(
                            unsafe { NonZeroU32::new_unchecked(has_key_mask) }.trailing_zeros(),
                        )
                    } else if self.keys_mask.has_space() {
                        KeySearchResult::NotFoundStop
                    } else {
                        KeySearchResult::NotFoundContinue
                    }
                }

                #[cfg(not(target_arch = "x86_64"))]
                {
                    {
                        for i in 0..self.keys_mask.total_keys() {
                            if unsafe { *self.keys.get_unchecked(i as usize) == key } {
                                return KeySearchResult::Found(i);
                            }
                        }

                        if self.keys_mask.has_space() {
                            KeySearchResult::NotFoundStop
                        } else {
                            KeySearchResult::NotFoundContinue
                        }
                    }
                }
            }

            #[inline(always)]
            fn try_insert(&mut self, key: Self::Key) -> Option<u32> {
                debug_assert!((0..self.keys_mask.total_keys())
                    .all(|i| unsafe { self.keys[i as usize].assume_init_read() } != key));
                self.keys_mask.add_key().map(|offset| unsafe {
                    self.keys.get_unchecked_mut(offset as usize).write(key);
                    offset
                })
            }
        }
    };
}

impl_block!(U32KeysBlock, u32, U32KeysMask);
impl_block!(U64KeysBlock, u64, U64KeysMask);
