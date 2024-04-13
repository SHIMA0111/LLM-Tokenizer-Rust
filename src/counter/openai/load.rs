use std::collections::HashMap;
use std::env::temp_dir;
use std::fs::{create_dir_all, File, remove_file, rename};
use std::io::{BufRead, Read, Write};
use std::path::Path;
use std::str::from_utf8;
use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use bstr::ByteSlice;
use regex::Regex;
use sha2::{Digest, Sha256};
use uuid::Uuid;
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

    let mut hex_hash = convert_to_hex(actual_result.as_slice());

    if hex_hash == expected_hash {
        true
    } else {
        false
    }
}

pub fn read_cached_file(blobpath: &str, expected_hash: Option<&str>) -> CounterResult<Vec<u8>> {
    let mut user_specified_cache = true;

    let cache_dir = if let Ok(val) = std::env::var("TIKTOKEN_CACHE_DIR") {
        Path::new(val.as_str()).to_path_buf()
    }
    else if let Ok(val) = std::env::var("DATA_GYM_CACHE_DIR") {
        Path::new(val.as_str()).to_path_buf()
    }
    else {
        user_specified_cache = false;
        let dir = temp_dir();
        dir.join(Path::new("data-gym-cache"))
    };

    if cache_dir == Path::new("") {
        return read_file(cache_dir.to_str().unwrap_or(""))
    }

    let mut cache_key_base = sha1::Sha1::new();
    cache_key_base.update(blobpath.as_bytes());
    let cache_key = convert_to_hex(cache_key_base.finalize().as_bytes());

    let cache_path = cache_dir.join(Path::new(&cache_key));

    if cache_path.exists() {
        if let Ok(mut file) = File::open(cache_path.clone()) {
            let mut content = Vec::new();

            if let Ok(_) = file.read_to_end(&mut content) {
                if expected_hash.is_some() {
                    if check_hash(&content, expected_hash.unwrap()) {
                        return Ok(content)
                    }
                }
            }
        }
        remove_file(cache_path.clone()).map_err(|e| CounterError::IOError(e.to_string()))?;
    }

    let contents = read_file(blobpath)?;
    if let Some(hash_value) = expected_hash {
        if !check_hash(&contents, hash_value) {
            return Err(CounterError::ValueError(format!(
                "Hash mismatch for data downloaded from {} (expected {}). \
                This may indicate a corrupted download. Please try again.",
                blobpath, hash_value)));
        }
    }

    create_dir_all(cache_dir).map_err(|e| CounterError::IOError(e.to_string()))?;
    let temp_file_name = cache_path.join(format!(".{}.tmp", Uuid::new_v4().to_string()));
    match File::create(temp_file_name.clone()) {
        Ok(mut file) => {
            file.write_all(&contents).map_err(|e| CounterError::IOError(e.to_string()))?;
        }
        Err(e) => return Err(CounterError::IOError(e.to_string()))
    }

    rename(temp_file_name, cache_path).map_err(|e| CounterError::IOError(e.to_string()))?;

    Ok(contents)

}

pub fn data_gym_to_mergeable_bpe_ranks(vocab_bpe_file: &str,
                                       encoder_json_file: &str,
                                       vocab_bpe_hash: Option<&str>,
                                       encoder_json_hash: Option<&str>
) -> CounterResult<HashMap<Vec<u8>, u32>> {
    let mut rank_to_intbyte = (0..=255).filter(|&b| {
        let c = b as char;
        c.is_ascii_graphic() && c != ' '
    }).collect::<Vec<u8>>();

    let mut data_gym_byte_to_byte = rank_to_intbyte.iter().map(|&byte| {
        (byte as char, byte)
    }).collect::<HashMap<_, _>>();

    let mut n: u8 = 0;

    for b in 0..=255 {
        if !rank_to_intbyte.contains(&b) {
            rank_to_intbyte.push(b);
            data_gym_byte_to_byte.insert((2_u8.pow(8) + n) as char, b);
            n += 1;
        }
    }
    assert_eq!(rank_to_intbyte.len(), 2_usize.pow(8));

    // vocab_bpe contains the merges along with associated ranks
    let vocab_bpe = read_cached_file(vocab_bpe_file, vocab_bpe_hash)?;
    let vocab_bpe_contents = match from_utf8(&vocab_bpe) {
        Ok(str) => str,
        Err(e) => return Err(CounterError::ByteDecodeError(e.to_string())),
    };

    let lines =
        vocab_bpe_contents
            .trim()
            .lines().collect::<Vec<_>>();

    let mut bpe_merges = Vec::new();
    let regex_pat =
        Regex::new(r"\s+").map_err(|e| CounterError::RegexError(e.to_string()))?;
    for line in lines {
        let split_values = regex_pat.split(line).collect::<Vec<_>>();
        if split_values.len() != 2 {
            return Err(CounterError::ValueError("bpe dictionary can't split to pair. Please check input.".to_string()));
        }
        bpe_merges.push((split_values[0], split_values[1]));
    };

    let decode_data_gym = |value: &str| -> CounterResult<Vec<u8>> {
        let mut res = Vec::new();
        for char in value.chars() {
            match data_gym_byte_to_byte.get(&char) {
                Some(value) => res.push(*value),
                None => return Err(CounterError::KeyError(format!("{} not found in byte2byte dict", char))),
            }
        }
        Ok(res)
    };

    // add the single byte tokens
    let mut bpe_ranks =
        rank_to_intbyte
            .iter()
            .enumerate()
            .map(|(idx, byte)| {
                (vec![*byte], idx as u32)
            })
            .collect::<HashMap<Vec<u8>, u32>>();

    // add the merged tokens
    let mut n = bpe_ranks.len() as u32;
    for (first, second) in bpe_merges {
        let mut first_bytes = decode_data_gym(first)?;
        let second_bytes = decode_data_gym(second)?;
        first_bytes.extend(second_bytes);
        bpe_ranks.insert(first_bytes, n);
        n += 1;
    }

    // check that the encoder file matches the merges file
    // this sanity check is important since this code assumes that ranks are ordered the same
    // as merge priority.
    let encoder_data = read_cached_file(encoder_json_file, encoder_json_hash)?;
    let encoder_json = match from_utf8(&encoder_data) {
        Ok(str) => {
            serde_json::from_str::<HashMap<String, u32>>(str)
                .map_err(|e| CounterError::ValueError(e.to_string()))?
        },
        Err(e) => return Err(CounterError::ByteDecodeError(e.to_string())),
    };

    let mut encoder_json_loaded = HashMap::new();

    for (key, value) in encoder_json {
        let decoded_key = decode_data_gym(&key)?;
        encoder_json_loaded.insert(decoded_key, value);
    }

    encoder_json_loaded.remove("<|endoftext|>".as_bytes());
    encoder_json_loaded.remove("<|startoftext|>".as_bytes());

    assert!(
        bpe_ranks
            .iter()
            .all(|(key, value)| encoder_json_loaded.get(key) == Some(&value)) &&
            encoder_json_loaded
                .iter()
                .all(|(key, value)| bpe_ranks.get(key) == Some(&value)));

    Ok(bpe_ranks)
}

pub fn dump_bpe(bpe_ranks: HashMap<Vec<u8>, u32>, bpe_file_path: &str) -> CounterResult<()> {
    let path = Path::new(bpe_file_path);

    let mut file = File::create(path).map_err(|e| CounterError::IOError(e.to_string()))?;

    let mut sorted_bpe_ranks = bpe_ranks.iter().collect::<Vec<(_, _)>>();
    sorted_bpe_ranks.sort_by(|first, second| first.1.cmp(second.1));

    for (bytes, token) in sorted_bpe_ranks {
        let encoded_byte = BASE64_STANDARD.encode(bytes);
        let token_bytes = token.to_string();

        file.write_all(encoded_byte.as_bytes()).map_err(|e| CounterError::IOError(e.to_string()))?;
        file.write_all(" ".as_bytes()).map_err(|e| CounterError::IOError(e.to_string()))?;
        file.write_all(token_bytes.as_bytes()).map_err(|e| CounterError::IOError(e.to_string()))?;
        file.write_all("\n".as_bytes()).map_err(|e| CounterError::IOError(e.to_string()))?;
    }

    Ok(())
}

pub fn load_bpe(bpe_file_path: &str,
                expected_hash: Option<&str>
) -> CounterResult<HashMap<Vec<u8>, u32>> {
    let contents = read_cached_file(bpe_file_path, expected_hash)?;
    let contents_str =
        from_utf8(&contents).map_err(|e| CounterError::ByteDecodeError(e.to_string()))?;

    let mut bpe_dict = HashMap::new();

    let regex_pat =
        Regex::new(r"\s+").map_err(|e| CounterError::RegexError(e.to_string()))?;
    for content in contents_str.lines() {
        let split_value = regex_pat.split(content).collect::<Vec<_>>();
        if split_value.len() != 2 {
            return Err(CounterError::ValueError("bpe dictionary can't split to pair. Please check input.".to_string()));
        }

        let bytes_value =
            BASE64_STANDARD.decode(split_value[0])
                .map_err(|e| CounterError::Base64DecodeError(e.to_string()))?;
        let token_value = match split_value[1].parse::<u32>() {
            Ok(val) => val,
            Err(e) => return Err(CounterError::ValueError(e.to_string())),
        };

        bpe_dict.insert(bytes_value, token_value);
    }

    Ok(bpe_dict)
}

fn convert_to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{:02x}", byte)).collect()
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
