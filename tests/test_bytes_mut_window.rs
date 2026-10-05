use bytes::BytesMut;

#[test]
fn write_inside_window_does_not_overwrite_protected_bytes() {
    let mut buf = BytesMut::with_capacity(8);
    buf.extend_from_slice(b"ab");

    {
        let mut win = buf.window();
        assert_eq!(win.parsed(), b"ab");

        // Record where the protected bytes live; writes through the window
        // must neither move nor overwrite them.
        let addr = win.parsed().as_ptr() as usize;

        // Reading "ab" then writing "cd" appends after the window.
        assert_eq!(win.try_write(b"cd").unwrap(), 2);
        assert_eq!(win.parsed(), b"abcd");
        assert_eq!(&win.parsed()[..2], b"ab");
        assert_eq!(win.parsed().as_ptr() as usize, addr);

        win.commit(2);
        assert_eq!(win.parsed(), b"cd");
    }

    assert_eq!(&buf[..], b"cd");
}

#[test]
fn commit_reclaims_only_the_confirmed_prefix() {
    let mut buf = BytesMut::with_capacity(4);
    buf.extend_from_slice(b"ab");

    {
        let mut win = buf.window();
        win.commit(2);
        // The confirmed prefix is released from the view...
        assert!(win.parsed().is_empty());
        assert_eq!(win.remaining_capacity(), 2);
    }

    // ...so the buffer only exposes the unconfirmed suffix (nothing here)
    // and the released prefix space.
    assert!(buf.is_empty());
    assert_eq!(buf.capacity(), 2);

    // The released prefix is reclaimable for later writes.
    assert!(buf.try_reclaim(4));
    assert_eq!(buf.capacity(), 4);
}

#[test]
fn commit_keeps_unconfirmed_suffix() {
    let mut buf = BytesMut::with_capacity(8);
    buf.extend_from_slice(b"abcdef");

    {
        let mut win = buf.window();
        win.commit(2); // confirm "ab"
        assert_eq!(win.parsed(), b"cdef"); // unconfirmed suffix retained
    }

    assert_eq!(&buf[..], b"cdef");

    // Later writes still append after the retained suffix.
    buf.extend_from_slice(b"gh");
    assert_eq!(&buf[..], b"cdefgh");
}

#[test]
fn drop_without_commit_keeps_all_data() {
    let mut buf = BytesMut::with_capacity(8);
    buf.extend_from_slice(b"ab");

    {
        let mut win = buf.window();
        win.try_write(b"cd").unwrap();
        assert_eq!(win.parsed(), b"abcd");
        // No commit: the window is abandoned.
    }

    // Dropping the window without committing keeps the data complete.
    assert_eq!(&buf[..], b"abcd");
}

#[test]
fn over_capacity_write_is_rejected_before_writing() {
    let mut buf = BytesMut::with_capacity(4);
    buf.extend_from_slice(b"ab");

    {
        let mut win = buf.window();
        assert_eq!(win.remaining_capacity(), 2);

        let err = win.try_write(b"cdef").unwrap_err();
        assert_eq!(err.requested(), 4);
        assert_eq!(err.available(), 2);

        // The rejected write changed nothing.
        assert_eq!(win.parsed(), b"ab");
        assert_eq!(win.remaining_capacity(), 2);
    }

    assert_eq!(&buf[..], b"ab");

    // Growth happens outside of the window's borrow boundary.
    buf.reserve(4);

    {
        let mut win = buf.window();
        assert_eq!(win.try_write(b"cdef").unwrap(), 4);
        assert_eq!(win.parsed(), b"abcdef");
        win.commit(6);
        assert!(win.parsed().is_empty());
    }

    assert!(buf.is_empty());
}

#[test]
fn empty_write_always_fits() {
    let mut buf = BytesMut::from(&b"ab"[..]);

    let mut win = buf.window();
    assert_eq!(win.remaining_capacity(), 0);
    assert_eq!(win.try_write(b"").unwrap(), 0);
    assert_eq!(win.parsed(), b"ab");
}

#[test]
fn six_byte_read_write_commit_example() {
    // Read "ab", write "cd" and "ef" through the window, commit the
    // confirmed prefix: six bytes pass through the protected window.
    let mut buf = BytesMut::with_capacity(8);
    buf.extend_from_slice(b"ab");

    {
        let mut win = buf.window();
        win.try_write(b"cd").unwrap();
        win.try_write(b"ef").unwrap();
        assert_eq!(win.parsed(), b"abcdef");

        win.commit(2);
        assert_eq!(win.parsed(), b"cdef");
    }

    assert_eq!(&buf[..], b"cdef");
}

#[test]
#[should_panic]
fn commit_past_protected_bytes_panics() {
    let mut buf = BytesMut::from(&b"ab"[..]);
    let mut win = buf.window();
    win.commit(3);
}
