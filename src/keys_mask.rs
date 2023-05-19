#[derive(Copy, Clone, Default)]
#[repr(transparent)]
pub(crate) struct KeysMask<const N: usize> {
    mask: u32,
}

impl<const N: usize> KeysMask<N> {
    /// Number of bits used in the mask.
    pub(crate) const IN_USE_BITS: u32 = N as u32;

    /// Get the index where the key should be added, if there is still space.
    #[inline(always)]
    pub(crate) fn add_key(&mut self) -> Option<u32> {
        let index = self.total_keys();
        debug_assert!(index <= Self::IN_USE_BITS);
        (index != Self::IN_USE_BITS).then(|| {
            debug_assert_eq!((self.mask >> index) & 1, 0);
            self.mask |= 1 << index;
            index
        })
    }

    /// Get the total number of keys in the block.
    #[inline(always)]
    pub(crate) fn total_keys(self) -> u32 {
        self.mask.trailing_ones()
    }

    /// Check if there is still space in the mask.
    #[inline(always)]
    pub(crate) fn has_space(self) -> bool {
        self.mask != (1 << N) - 1
    }

    /// Get the bits in the mask.
    #[inline(always)]
    pub(crate) fn bits(self) -> u32 {
        self.mask
    }

    /// Get ptr to the internal mask.
    #[inline(always)]
    pub(crate) fn as_ptr(&self) -> *const u32 {
        &self.mask as *const u32
    }
}

pub(crate) type U32KeysMask = KeysMask<31>;
pub(crate) type U64KeysMask = KeysMask<15>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask_u32() {
        let mut mask_u32 = U32KeysMask::default();

        for i in 0..U32KeysMask::IN_USE_BITS {
            assert_eq!(mask_u32.add_key(), Some(i));
        }
        assert!(mask_u32.add_key().is_none());
    }

    #[test]
    fn test_mask_u64() {
        let mut mask_u64 = U64KeysMask::default();

        for i in 0..U64KeysMask::IN_USE_BITS {
            assert_eq!(mask_u64.add_key(), Some(i));
        }
        assert!(mask_u64.add_key().is_none());
    }
}
