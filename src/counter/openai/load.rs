use std::fs::File;
use std::io::Read;
use std::path::Path;
use bstr::ByteSlice;
use sha2::{Digest, Sha256};
use crate::errors::{CounterError, CounterResult};

pub fn read_file(blobpath: &str) -> CounterResult<Vec<u8>> {
    if !blobpath.starts_with("http://") && !blobpath.starts_with("https://") {
        let path = Path::new(blobpath);

        let mut file = match File::open(path) {
            Ok(file) => file,
            Err(e) => return Err(CounterError::IOError(e.to_string())),
        };

        let mut content = Vec::new();

        return match file.read_to_end(&mut content) {
            Ok(_) => Ok(content),
            Err(e) => Err(CounterError::IOError(e.to_string())),
        }
    }

    let resp = reqwest::blocking::get(blobpath)
        .map_err(|e| CounterError::IOError(e.to_string()))?
        .bytes()
        .map_err(|e| CounterError::IOError(e.to_string()))?;

    Ok(resp.as_bytes().to_owned())
}


pub fn check_hash(data: &[u8], expected_hash: &str) -> bool {
    let mut hash = Sha256::new();
    Digest::update(&mut hash, data);
    let actual_result = hash.finalize();

    let mut hex_hash = String::new();

    for byte in &actual_result {
        hex_hash.push_str(&format!("{:02x}", byte));
    }

    if hex_hash == expected_hash {
        true
    } else {
        false
    }
}

#[test]
fn test_check_hash() {
    let text = "test".as_bytes();
    assert!(
        check_hash(
            text,
            "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
        ))
}
