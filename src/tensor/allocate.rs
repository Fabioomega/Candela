use std::alloc::{self, Layout};
use std::marker::PhantomData;
use std::mem;
use std::ptr::NonNull;

/// Owns an aligned, uninitialized allocation for `len` values of type `T`,
/// and frees it on drop.
///
/// Zero-sized allocations (`len == 0`, or `T` being a ZST) never touch the
/// allocator: the buffer holds a dangling-but-correctly-aligned pointer and
/// `drop` does nothing.
pub struct AlignedBuf<T> {
    ptr: NonNull<T>,
    len: usize,
    layout: Layout,
    _owns: PhantomData<T>,
}

impl<T> AlignedBuf<T> {
    /// Allocates uninitialized storage for `len` values of `T`.
    ///
    /// `align` is raised to at least `align_of::<T>()`, so the pointer is
    /// always valid for `T`.
    ///
    /// # Panics
    /// If `align` is not a power of two, or the total size overflows `isize`.
    /// Allocation failure goes through `handle_alloc_error`.
    pub fn new(len: usize, align: usize) -> Self {
        debug_assert!(align.is_power_of_two(), "alignment must be a power of two");

        let align = align.max(mem::align_of::<T>());

        let size = mem::size_of::<T>()
            .checked_mul(len)
            .expect("allocation size overflow");

        let layout = Layout::from_size_align(size, align).expect("invalid layout");

        let ptr = if size == 0 {
            // Nothing to allocate. `align` is a non-zero, suitably aligned
            // address.
            // SAFETY: `align` is a power of two, therefore non-null.
            unsafe { NonNull::new_unchecked(align as *mut T) }
        } else {
            // SAFETY: `layout` has non-zero size.
            let raw = unsafe { alloc::alloc(layout) };
            match NonNull::new(raw.cast::<T>()) {
                Some(p) => p,
                None => alloc::handle_alloc_error(layout),
            }
        };

        Self {
            ptr,
            len,
            layout,
            _owns: PhantomData,
        }
    }

    /// Pointer to the start of the allocation.
    /// Aligned and non-null with uninitialized memory.
    #[inline]
    pub fn as_ptr(&self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Number of `T` slots requested.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// True when nothing was actually allocated (`len == 0` or `T` is a ZST).
    #[inline]
    pub fn is_dangling(&self) -> bool {
        self.layout.size() == 0
    }

    /// # Safety
    /// All `len` elements must have been initialized, and no `&mut` to them
    /// may exist for the lifetime of the returned slice.
    #[inline]
    pub unsafe fn assume_init_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// # Safety
    /// All `len` elements must have been initialized, and no other reference
    /// to them may exist for the lifetime of the returned slice.
    #[inline]
    pub unsafe fn assume_init_slice_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> Drop for AlignedBuf<T> {
    fn drop(&mut self) {
        if self.layout.size() != 0 {
            // SAFETY: this pointer came from `alloc` with
            // exactly this layout, and drop is only called once.
            unsafe { alloc::dealloc(self.ptr.as_ptr().cast::<u8>(), self.layout) }
        }
    }
}

// SAFETY: the buffer only owns storage; sending/sharing it is as safe as
// sending/sharing the `T`s it holds.
unsafe impl<T: Send> Send for AlignedBuf<T> {}
unsafe impl<T: Sync> Sync for AlignedBuf<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn respects_alignment() {
        let buf = AlignedBuf::<f32>::new(1024, 64);
        assert_eq!(buf.as_ptr() as usize % 64, 0);
        assert!(!buf.is_dangling());
    }

    #[test]
    fn zero_length_does_not_allocate() {
        let buf = AlignedBuf::<f32>::new(0, 64);
        assert!(buf.is_dangling());
        assert_eq!(buf.as_ptr() as usize % 64, 0);
        // dropping must not call dealloc
    }

    #[test]
    fn zst_does_not_allocate() {
        let buf = AlignedBuf::<()>::new(1000, 16);
        assert!(buf.is_dangling());
        assert_eq!(buf.len(), 1000);
    }

    #[test]
    fn write_then_read() {
        let mut buf = AlignedBuf::<u64>::new(4, 32);
        for i in 0..buf.len() {
            unsafe { buf.as_ptr().add(i).write(i as u64) };
        }
        assert_eq!(unsafe { buf.assume_init_slice_mut() }, &[0, 1, 2, 3]);
    }
}
