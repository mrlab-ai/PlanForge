//! Hashing and comparing packed states and numeric values.

use std::hash::{BuildHasherDefault, Hasher};

/// Fast mixer for exact `f64` bit patterns. Raw IEEE-754 bits are not
/// well-distributed in their low bits (consecutive integers share long zero
/// suffixes), so a pass-through hasher makes the compact-value table
/// degenerate into long probe chains.
#[derive(Default)]
pub(super) struct MixedU64Hasher(u64);

impl Hasher for MixedU64Hasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        if bytes.len() == 8 {
            self.0 = u64::from_ne_bytes(bytes.try_into().unwrap());
        } else {
            for &byte in bytes {
                self.0 = self.0.rotate_left(5) ^ u64::from(byte);
            }
        }
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.0 = value;
    }

    #[inline]
    fn write_usize(&mut self, value: usize) {
        self.0 = value as u64;
    }

    #[inline]
    fn finish(&self) -> u64 {
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
}

pub(super) type MixedHasherBuilder = BuildHasherDefault<MixedU64Hasher>;

/// MurmurHash3's 64-bit finalizer (`fmix64`). Spreads bits well across both halves of the output
/// so the result is suitable as a key for hashbrown (which uses both the
/// low bits for the bucket index and the top 7 bits for the SIMD tag).
#[inline]
fn finalize_mix(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    x
}

/// 64-bit FNV-1a over the bytes of `words`, finished with [`finalize_mix`].
///
/// Hashes byte by byte rather than word by word: the word variant produces
/// noticeably more hashbrown bucket collisions on packed state buffers, whose
/// bins share long stretches of zero bits, and comparing the buffers of a
/// grown bucket dominates deduplication. The extra multiplications per bin
/// are cheap by comparison.
#[inline]
fn fnv1a_words(words: impl Iterator<Item = u64>) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x00000100000001B3;
    let mut hash = FNV_OFFSET;
    for word in words {
        for byte in word.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    finalize_mix(hash)
}

#[inline]
pub(super) fn fast_hash_bins(bins: &[u64]) -> u64 {
    fnv1a_words(bins.iter().copied())
}

/// The hash of only the bits the per-bin `mask` selects, to deduplicate
/// buffers whose bins also hold derived (axiom-computed) bits.
#[inline]
pub(super) fn fast_hash_bins_masked(bins: &[u64], mask: &[u64]) -> u64 {
    debug_assert_eq!(bins.len(), mask.len());
    fnv1a_words(bins.iter().zip(mask).map(|(&bin, &mask)| bin & mask))
}

/// Compare two buffers, considering only the bits selected by `mask`.
#[inline]
pub(super) fn bins_eq_masked(left: &[u64], right: &[u64], mask: &[u64]) -> bool {
    debug_assert_eq!(left.len(), right.len());
    debug_assert_eq!(left.len(), mask.len());
    for ((&l, &r), &m) in left.iter().zip(right.iter()).zip(mask.iter()) {
        if (l & m) != (r & m) {
            return false;
        }
    }
    true
}
