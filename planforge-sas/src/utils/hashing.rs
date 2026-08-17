use std::hash::Hasher;

/// Pass-through hasher for dense integer IDs and already-mixed state hashes.
#[derive(Default)]
pub struct IdentityU64Hasher(u64);

impl Hasher for IdentityU64Hasher {
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
        self.0
    }
}
