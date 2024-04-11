/// This source code was copied from https://github.com/rust-lang/rust.
///
/// The original license of the project having this code under
///     - https://github.com/rust-lang/rust/blob/master/LICENSE-MIT
///     - https://github.com/rust-lang/rust/blob/master/LICENSE-APACHE

use std::borrow::Cow;
use std::fmt;
use std::fmt::{Formatter, Write};
use std::iter::FusedIterator;
use std::slice::from_raw_parts;
use std::str::from_utf8_unchecked;

struct Debug<'a>(&'a [u8]);

impl fmt::Debug for Debug<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_char('"')?;

        for chunk in BytesChunks::new(self.0) {
            {
                let valid = chunk.valid();
                let mut from = 0;
                for (i, c) in valid.char_indices() {
                    let esc = c.escape_debug();
                    if esc.len() != 1 {
                        f.write_str(&valid[from..i])?;
                        for c in esc {
                            f.write_char(c)?;
                        }
                        from = i + c.len_utf8();
                    }
                }
                f.write_str(&valid[from..])?;
            }

            for &b in chunk.invalid() {
                write!(f, "\\x{:02X}", b)?;
            }
        }

        f.write_char('"')
    }
}

struct BytesChunks<'a> {
    source: &'a [u8],
}

impl <'a> BytesChunks<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            source: bytes,
        }
    }

    fn debug(&self) -> Debug<'_> {
        Debug(self.source)
    }
}

impl <'a> Iterator for BytesChunks<'a> {
    type Item = BytesChunk<'a>;

    fn next(&mut self) -> Option<BytesChunk<'a>> {
        if self.source.is_empty() {
            return None;
        }

        const TAG_CONT_U8: u8 = 128;
        fn safe_get(xs: &[u8], i: usize) -> u8 {
            *xs.get(i).unwrap_or(&0)
        }

        let mut i = 0;
        let mut valid_up_to = 0;

        while i < self.source.len() {
            let byte = unsafe { *self.source.get_unchecked(i) };
            i += 1;

            if byte < 128 {
                // This could be a `1 => ...` case in the match below, but for
                // the common case of all-ASCII inputs, we bypass loading the
                // sizeable UTF8_CHAR_WIDTH table into cache.
            }
            else {
                let w = utf8_char_width(byte);

                match w {
                    2 => {
                        if safe_get(self.source, i) & 192 != TAG_CONT_U8 {
                            break;
                        }
                        i += 1;
                    }
                    3 => {
                        match (byte, safe_get(self.source, i)) {
                            (0xE0, 0xA0..=0xBF) => (),
                            (0xE1..=0xEC, 0x80..=0xBF) => (),
                            (0xED, 0x80..=0x9F) => (),
                            (0xEE..=0xEF, 0x80..=0xBF) => (),
                            _ => break,
                        }
                        i += 1;
                        if safe_get(self.source, i) & 192 != TAG_CONT_U8 {
                            break;
                        }
                        i += 1;
                    }
                    4 => {
                        match (byte, safe_get(self.source, i)) {
                            (0xF0, 0x90..=0xBF) => (),
                            (0xF1..=0xF3, 0x80..=0xBF) => (),
                            (0xF4, 0x80..=0x8F) => (),
                            _ => break,
                        }
                        i += 1;
                        if safe_get(self.source, i) & 192 != TAG_CONT_U8 {
                            break;
                        }
                        i += 1;
                        if safe_get(self.source, i) & 192 != TAG_CONT_U8 {
                            break;
                        }
                        i += 1;
                    }
                    _ => break
                }
            }

            valid_up_to = i;
        }

        let (inspected, remaining) = unsafe { split_at_unchecked(self.source, i) };
        self.source = remaining;

        let (valid, invalid) = unsafe { split_at_unchecked(inspected, valid_up_to) };

        Some(BytesChunk {
            valid: unsafe {from_utf8_unchecked(valid)},
            invalid,
        })
    }
}

impl FusedIterator for BytesChunks<'_> {}

impl fmt::Debug for BytesChunks<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("BytesChunks").field("source", &self.debug()).finish()
    }
}

/// Copied from https://github.com/rust-lang/rust/blob/master/library/core/src/str/validations.rs#L245
const UTF8_CHAR_WIDTH: &[u8; 256] = &[
    // 1  2  3  4  5  6  7  8  9  A  B  C  D  E  F
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 0
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 1
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 2
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 3
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 4
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 5
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 6
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 7
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 8
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 9
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // A
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // B
    0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, // C
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, // D
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, // E
    4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // F
];

const fn utf8_char_width(b: u8) -> usize {
    UTF8_CHAR_WIDTH[b as usize] as usize
}

const fn split_at_unchecked<T>(slice: &[T], mid: usize) -> (&[T], &[T]) {
    let len = slice.len();
    let ptr = slice.as_ptr();

    debug_assert!(
        mid <= len,
        "split_at_unchecked requires the index to be within the slice",
    );

    unsafe { (from_raw_parts(ptr, mid), from_raw_parts(ptr.add(mid), len - mid)) }
}

struct BytesChunk<'a> {
    valid: &'a str,
    invalid: &'a [u8],
}

impl <'a> BytesChunk<'a> {
    pub fn valid(&self) -> &'a str {
        self.valid
    }

    pub fn invalid(&self) -> &'a [u8] {
        self.invalid
    }
}

fn from_utf8_or<'a>(v: &'a [u8], replace_bytes: &'a str) -> Cow<'a, str> {
    let mut iter = BytesChunks::new(v);

    let first_valid = if let Some(chunk) = iter.next() {
        let valid = chunk.valid();
        if chunk.invalid().is_empty() {
            debug_assert_eq!(valid.len(), v.len());
            return Cow::Borrowed(valid)
        }
        valid
    } else {
        return Cow::Borrowed("");
    };

    let mut res = String::with_capacity(v.len());
    res.push_str(first_valid);
    res.push_str(replace_bytes);

    for chunk in iter {
        res.push_str(chunk.valid());
        if !chunk.invalid().is_empty() {
            res.push_str(replace_bytes);
        }
    }

    Cow::Owned(res)
}

pub fn from_utf8_ignore(v: &[u8]) -> Cow<str> {
    from_utf8_or(v, "")
}

pub fn from_utf8_backslash(v: &[u8]) -> Cow<str> {
    let mut iter = BytesChunks::new(v);

    let (first_valid, invalid) = if let Some(chunk) = iter.next() {
        let valid = chunk.valid();
        if chunk.invalid().is_empty() {
            debug_assert_eq!(valid.len(), v.len());
            return Cow::Borrowed(valid)
        }
        (valid, chunk.invalid())
    } else {
        return Cow::Borrowed("")
    };

    let push_str_to_string = |bytes: &[u8], res_str: &mut String| {
        for byte in bytes {
            res_str.push_str(format!("\\x{:02X}", byte).as_str())
        }
    };

    let mut res = String::with_capacity(v.len());
    res.push_str(first_valid);
    push_str_to_string(invalid, &mut res);

    for chunk in iter {
        res.push_str(chunk.valid());
        if !chunk.invalid().is_empty() {
            push_str_to_string(chunk.invalid(), &mut res);
        }
    }

    Cow::Owned(res)
}
