//! Fixed-size slab pool of pre-allocated, page-aligned RAM buffers.
//!
//! The spec calls for "hot-swapping experts ... into a **pre-allocated** RAM
//! buffer". A naive implementation would `Vec::with_capacity` per cache miss,
//! which (a) is not aligned for `O_DIRECT`, (b) thrashes the allocator, and
//! (c) makes total RAM use unbounded.
//!
//! Instead, at startup we allocate `slots` aligned buffers of `expert_size`
//! bytes and hand them out as [`PooledBuffer`] guards. When a guard is dropped
//! its buffer is returned to the pool's free list — so the LRU eviction of an
//! expert automatically frees a slot for the next miss.
//!
//! # Primary / Shadow split (gist Phase 1, "zero-stall pipeline")
//!
//! When the engine wants to overlap the current token's compute with the
//! *next* token's speculative prefetch, the primary pool — which backs the
//! resident LRU — must never be raided by speculation. The pool therefore
//! supports an optional **shadow** half: a second free-list of the same
//! size buffers, served by [`BufferPool::try_acquire_shadow`]. Speculative
//! prefetches go there; on confirmation the engine calls
//! [`BufferPool::promote_shadow`] to atomically swap the shadow buffer
//! into the primary half so the cache can install it as a resident
//! without re-reading the SSD. The two halves share one [`Notify`], so
//! a `Drop` on either side wakes any waiter on the other.

use crate::aligned_buffer::AlignedBuffer;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::Notify;

static EXPERT_BUFFER_POOL_PRIMARY_BYTES: AtomicU64 = AtomicU64::new(0);
static EXPERT_BUFFER_POOL_SHADOW_BYTES: AtomicU64 = AtomicU64::new(0);

/// Bytes allocated by the primary slots of all live expert buffer pools.
pub fn expert_buffer_pool_primary_bytes() -> u64 {
    EXPERT_BUFFER_POOL_PRIMARY_BYTES.load(Ordering::Relaxed)
}

/// Bytes allocated by the shadow slots of all live expert buffer pools.
pub fn expert_buffer_pool_shadow_bytes() -> u64 {
    EXPERT_BUFFER_POOL_SHADOW_BYTES.load(Ordering::Relaxed)
}

/// Bytes allocated by every slot of all live expert buffer pools.
pub fn expert_buffer_pool_allocated_bytes() -> u64 {
    expert_buffer_pool_primary_bytes().saturating_add(expert_buffer_pool_shadow_bytes())
}

struct Inner {
    /// Primary free list — slots backing the resident LRU.
    free: Mutex<Vec<AlignedBuffer>>,
    /// Optional shadow free list — slots backing speculative prefetches.
    /// `None` keeps the legacy single-pool semantics with zero overhead.
    shadow: Option<Mutex<Vec<AlignedBuffer>>>,
    notify: Notify,
    /// Separate wakeup channel for shadow-pool waiters. A single shared
    /// `Notify` would let a shadow-buffer drop consume the wakeup of a
    /// primary waiter (and vice versa), losing the notification for the
    /// waiter that could actually make progress.
    notify_shadow: Notify,
    primary_slots: usize,
    shadow_slots: usize,
    buffer_size: usize,
    align: usize,
}

impl Drop for Inner {
    fn drop(&mut self) {
        let primary = (self.primary_slots as u64).saturating_mul(self.buffer_size as u64);
        let shadow = (self.shadow_slots as u64).saturating_mul(self.buffer_size as u64);
        EXPERT_BUFFER_POOL_PRIMARY_BYTES.fetch_sub(primary, Ordering::Relaxed);
        EXPERT_BUFFER_POOL_SHADOW_BYTES.fetch_sub(shadow, Ordering::Relaxed);
    }
}

/// A bounded pool of aligned RAM buffers.
#[derive(Clone)]
pub struct BufferPool {
    inner: Arc<Inner>,
}

impl BufferPool {
    /// Pre-allocate `slots` buffers of `buffer_size` bytes each, aligned to `align`.
    pub fn new(slots: usize, buffer_size: usize, align: usize) -> Self {
        Self::new_with_shadow(slots, 0, buffer_size, align)
    }

    /// Pre-allocate `primary_slots + shadow_slots` aligned buffers. The
    /// **primary** half backs the resident LRU; the **shadow** half is
    /// reserved for speculative prefetches (see [`Self::try_acquire_shadow`]).
    /// Pass `shadow_slots == 0` to disable the shadow path entirely —
    /// `try_acquire_shadow` then always returns `None` and the engine's
    /// prefetch path falls back to its previous "non-evicting `try_acquire`
    /// against the primary pool" behaviour.
    pub fn new_with_shadow(
        primary_slots: usize,
        shadow_slots: usize,
        buffer_size: usize,
        align: usize,
    ) -> Self {
        assert!(primary_slots > 0, "primary pool must have at least one slot");
        let mut free = Vec::with_capacity(primary_slots);
        for _ in 0..primary_slots {
            free.push(AlignedBuffer::new(buffer_size, align));
        }
        let shadow = if shadow_slots > 0 {
            let mut s = Vec::with_capacity(shadow_slots);
            for _ in 0..shadow_slots {
                s.push(AlignedBuffer::new(buffer_size, align));
            }
            Some(Mutex::new(s))
        } else {
            None
        };
        let primary_bytes = (primary_slots as u64).saturating_mul(buffer_size as u64);
        let shadow_bytes = (shadow_slots as u64).saturating_mul(buffer_size as u64);
        EXPERT_BUFFER_POOL_PRIMARY_BYTES.fetch_add(primary_bytes, Ordering::Relaxed);
        EXPERT_BUFFER_POOL_SHADOW_BYTES.fetch_add(shadow_bytes, Ordering::Relaxed);
        Self {
            inner: Arc::new(Inner {
                free: Mutex::new(free),
                shadow,
                notify: Notify::new(),
                notify_shadow: Notify::new(),
                primary_slots,
                shadow_slots,
                buffer_size,
                align,
            }),
        }
    }

    /// Total number of **primary** slots in the pool. Preserved across
    /// the primary/shadow refactor so existing callers (cache sizing,
    /// iovec registration) keep working.
    pub fn capacity(&self) -> usize {
        self.inner.primary_slots
    }

    /// Number of **shadow** slots, or 0 if the shadow path is disabled.
    pub fn shadow_capacity(&self) -> usize {
        self.inner.shadow_slots
    }

    /// Size of each buffer in the pool, in bytes.
    pub fn buffer_size(&self) -> usize {
        self.inner.buffer_size
    }

    /// Fixed primary allocation for this pool, independent of occupancy.
    pub fn primary_allocated_bytes(&self) -> usize {
        self.capacity().saturating_mul(self.buffer_size())
    }

    /// Fixed shadow allocation for this pool, independent of occupancy.
    pub fn shadow_allocated_bytes(&self) -> usize {
        self.shadow_capacity().saturating_mul(self.buffer_size())
    }

    /// Fixed total allocation for this pool, independent of occupancy.
    pub fn allocated_bytes(&self) -> usize {
        self.primary_allocated_bytes()
            .saturating_add(self.shadow_allocated_bytes())
    }

    /// Snapshot the raw `(ptr, len)` of every currently-free buffer
    /// (primary **and** shadow, in that order).
    ///
    /// Intended for the `io_uring` registered-fixed-buffers path: at
    /// startup, callers register every pool buffer with the kernel
    /// once, then submit reads referencing buffer indices instead of
    /// per-read iovecs. Returned pointers are stable for the lifetime
    /// of the pool (each `AlignedBuffer` owns a heap allocation that
    /// the pool keeps alive).
    ///
    /// **Safety contract:** the caller must not write to these
    /// pointers concurrently with another holder of the corresponding
    /// `PooledBuffer`. In practice this means: register the iovecs
    /// once at startup, before any `acquire()` call, and never modify
    /// the pool's free-list contents while reads are in flight.
    pub fn raw_iovecs(&self) -> Vec<(*mut u8, usize)> {
        let mut out = Vec::with_capacity(self.inner.primary_slots + self.inner.shadow_slots);
        {
            let mut g = self.inner.free.lock();
            out.extend(g.iter_mut().map(|b| (b.as_mut_slice().as_mut_ptr(), b.len())));
        }
        if let Some(s) = &self.inner.shadow {
            let mut g = s.lock();
            out.extend(g.iter_mut().map(|b| (b.as_mut_slice().as_mut_ptr(), b.len())));
        }
        out
    }

    /// Try to pop a free **primary** buffer immediately. Returns `None`
    /// if the primary pool is empty.
    pub fn try_acquire(&self) -> Option<PooledBuffer> {
        let buf = self.inner.free.lock().pop()?;
        Some(PooledBuffer {
            buffer: Some(buf),
            pool: self.inner.clone(),
            slot: Slot::Primary,
        })
    }

    #[cfg(test)]
    pub(crate) fn primary_available(&self) -> usize {
        self.inner.free.lock().len()
    }

    /// Try to pop a free **shadow** buffer immediately. Returns `None`
    /// if the shadow pool is disabled (`shadow_capacity() == 0`) **or**
    /// fully in flight. Speculative prefetches should always use this
    /// entry point so they cannot starve real work on the primary
    /// pool.
    pub fn try_acquire_shadow(&self) -> Option<PooledBuffer> {
        let shadow = self.inner.shadow.as_ref()?;
        let buf = shadow.lock().pop()?;
        Some(PooledBuffer {
            buffer: Some(buf),
            pool: self.inner.clone(),
            slot: Slot::Shadow,
        })
    }

    /// Wait asynchronously until a free buffer is available, then return it.
    ///
    /// This provides natural backpressure: when the cache is full and every
    /// resident expert is referenced, new fetches simply wait for one to be
    /// dropped (i.e. evicted *and* released by the inference layer).
    pub async fn acquire(&self) -> PooledBuffer {
        loop {
            if let Some(b) = self.try_acquire() {
                return b;
            }
            // Register interest *before* re-checking to avoid a lost wakeup.
            let notified = self.inner.notify.notified();
            if let Some(b) = self.try_acquire() {
                return b;
            }
            notified.await;
        }
    }

    /// Wait asynchronously for a free **shadow** buffer. Returns `None`
    /// immediately if the shadow pool is disabled, so callers can
    /// branch deterministically.
    #[allow(dead_code)]
    pub async fn acquire_shadow(&self) -> Option<PooledBuffer> {
        if self.inner.shadow.is_none() {
            return None;
        }
        loop {
            if let Some(b) = self.try_acquire_shadow() {
                return Some(b);
            }
            let notified = self.inner.notify_shadow.notified();
            if let Some(b) = self.try_acquire_shadow() {
                return Some(b);
            }
            notified.await;
        }
    }

    /// Promote a [`PooledBuffer`] previously acquired from the shadow
    /// pool into the **primary** pool, by reassigning the slot tag on
    /// drop. This is the zero-copy hand-off that turns a speculative
    /// prefetch into a confirmed resident: the caller passes the
    /// buffer in, gets the same backing memory back, but on `Drop` it
    /// returns to the primary half (where the LRU's `Arc` lives) so
    /// the resident count is accounted correctly.
    ///
    /// If `buf` was already primary, this is a no-op. If the shadow
    /// half is disabled (capacity 0), promotion is also a no-op
    /// because there is nothing meaningful to swap.
    pub fn promote_shadow(&self, mut buf: PooledBuffer) -> PooledBuffer {
        if buf.slot == Slot::Shadow {
            buf.slot = Slot::Primary;
        }
        buf
    }
}

/// Which free-list a [`PooledBuffer`] returns to on drop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Slot {
    Primary,
    Shadow,
}

/// RAII guard wrapping an [`AlignedBuffer`] borrowed from a [`BufferPool`].
///
/// Dropping the guard returns the buffer to the pool. The buffer is **not**
/// zeroed on return; callers should treat newly-acquired buffers as
/// uninitialised beyond the byte range they explicitly write.
pub struct PooledBuffer {
    buffer: Option<AlignedBuffer>,
    pool: Arc<Inner>,
    slot: Slot,
}

impl PooledBuffer {
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        self.buffer.as_ref().expect("PooledBuffer must hold a buffer until Drop").as_slice()
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.buffer.as_mut().expect("PooledBuffer must hold a buffer until Drop").as_mut_slice()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.buffer.as_ref().expect("PooledBuffer must hold a buffer until Drop").len()
    }

    /// Whether this buffer was acquired from the shadow (speculative)
    /// half. Used by the engine to decide whether a confirmed
    /// prefetch needs `BufferPool::promote_shadow` before becoming
    /// resident.
    #[allow(dead_code)]
    #[inline]
    pub fn is_shadow(&self) -> bool {
        self.slot == Slot::Shadow
    }
}

impl std::ops::Deref for PooledBuffer {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl AsRef<[u8]> for PooledBuffer {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl AsMut<[u8]> for PooledBuffer {
    fn as_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        if let Some(buf) = self.buffer.take() {
            match self.slot {
                Slot::Primary => {
                    self.pool.free.lock().push(buf);
                    self.pool.notify.notify_one();
                }
                Slot::Shadow => {
                    // Shadow only exists when the pool was built with
                    // `new_with_shadow(_, shadow > 0, ...)`. If a buffer
                    // was tagged Shadow but the shadow list is gone we
                    // fall back to primary — safer than leaking the
                    // allocation.
                    if let Some(s) = &self.pool.shadow {
                        s.lock().push(buf);
                        self.pool.notify_shadow.notify_one();
                    } else {
                        self.pool.free.lock().push(buf);
                        self.pool.notify.notify_one();
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn acquire_and_release_cycle() {
        let pool = BufferPool::new(2, 4096, 4096);
        let a = pool.acquire().await;
        let b = pool.acquire().await;
        assert!(pool.try_acquire().is_none());
        drop(a);
        assert!(pool.try_acquire().is_some());
        drop(b);
    }

    #[tokio::test]
    async fn acquire_waits_for_release() {
        let pool = BufferPool::new(1, 4096, 4096);
        let held = pool.acquire().await;
        let pool2 = pool.clone();
        let h = tokio::spawn(async move {
            let _b = pool2.acquire().await;
        });
        // Give the spawned task a moment to park.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        drop(held);
        h.await.unwrap();
    }

    #[test]
    fn raw_iovecs_reports_every_free_buffer() {
        // With nothing acquired, raw_iovecs reports all `slots` buffers.
        let pool = BufferPool::new(3, 4096, 4096);
        let iovecs = pool.raw_iovecs();
        assert_eq!(iovecs.len(), 3);
        for (p, l) in iovecs {
            assert!(!p.is_null());
            assert_eq!(l, 4096);
        }
    }

    // ---------------------- shadow-pool tests --------------------------

    #[tokio::test]
    async fn shadow_pool_independent_of_primary() {
        let pool = BufferPool::new_with_shadow(2, 2, 4096, 4096);
        assert_eq!(pool.capacity(), 2);
        assert_eq!(pool.shadow_capacity(), 2);

        // Drain primary; shadow stays full.
        let _a = pool.acquire().await;
        let _b = pool.acquire().await;
        assert!(pool.try_acquire().is_none());

        // Drain both shadow slots; primary stays exhausted independently.
        let _s1 = pool.try_acquire_shadow().expect("first shadow free");
        let _s2 = pool.try_acquire_shadow().expect("second shadow free");
        assert!(pool.try_acquire_shadow().is_none());
        assert!(pool.try_acquire().is_none());
    }

    #[test]
    fn shadow_disabled_returns_none() {
        let pool = BufferPool::new(2, 4096, 4096);
        assert_eq!(pool.shadow_capacity(), 0);
        assert!(pool.try_acquire_shadow().is_none());
    }

    #[tokio::test]
    async fn promote_shadow_routes_drop_to_primary() {
        let pool = BufferPool::new_with_shadow(1, 1, 4096, 4096);
        // Hold the only primary slot so we can observe where the
        // promoted shadow buffer lands on drop.
        let _hold = pool.acquire().await;
        let s = pool.try_acquire_shadow().expect("shadow free");
        assert!(s.is_shadow());
        let promoted = pool.promote_shadow(s);
        assert!(!promoted.is_shadow());
        // Dropping the promoted buffer must return it to *primary*,
        // not shadow, so the shadow slot is now exhausted.
        drop(promoted);
        // Primary now has one free buffer (the just-dropped, promoted one).
        let _p = pool.try_acquire().expect("primary should have the promoted buffer");
        // Shadow remains empty because we promoted its only slot.
        assert!(pool.try_acquire_shadow().is_none());
    }

    #[test]
    fn raw_iovecs_includes_shadow_buffers() {
        let pool = BufferPool::new_with_shadow(3, 2, 4096, 4096);
        let iovecs = pool.raw_iovecs();
        assert_eq!(iovecs.len(), 5);
    }

    #[test]
    fn allocated_bytes_use_fixed_slot_counts_not_occupancy() {
        let pool = BufferPool::new_with_shadow(3, 2, 8192, 4096);
        assert_eq!(pool.primary_allocated_bytes(), 3 * 8192);
        assert_eq!(pool.shadow_allocated_bytes(), 2 * 8192);
        assert_eq!(pool.allocated_bytes(), 5 * 8192);
        let _primary = pool.try_acquire().unwrap();
        let _shadow = pool.try_acquire_shadow().unwrap();
        assert_eq!(pool.allocated_bytes(), 5 * 8192);
    }
}
