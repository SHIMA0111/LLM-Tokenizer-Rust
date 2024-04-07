/// This source is copied from lib.rs of master branch on https://github.com/openai/tiktoken/
/// at April 7, 2024, following the MIT license of the original project.
/// (The exact file is https://github.com/openai/tiktoken/blob/main/src/lib.rs)
///
/// Please note this source file was edited by this(langcat) project
/// so some parts doesn't match the original library.
///
/// The original license is here: https://github.com/openai/tiktoken/blob/main/LICENSE

use std::collections::HashSet;
use std::num::NonZeroU64;
use std::str::from_utf8;
use std::thread;
use std::thread::ThreadId;

use fancy_regex::Regex;
use rustc_hash::FxHashMap as HashMap;
use crate::errors::{CounterError, CounterResult};

type Rank = u32;

fn _byte_pair_merge(ranks: &HashMap<Vec<u8>, Rank>, piece: &[u8]) -> Vec<(usize, Rank)> {
    // This is a vector of (start, rank).
    // The rank is of the pair starting at position start.
    let mut parts = Vec::with_capacity(piece.len() + 1);

    // Note that we hash bytes when indexing into `ranks`, not token pairs. As long as we train BPE
    // the way we currently do, this is equivalent. An easy way to break this would be to decouple
    // merge priority from token index or to prevent specific token merges.
    let mut min_rank: (Rank, usize) = (Rank::MAX, usize::MAX);
    for i in 0..piece.len() - 1 {
        let rank = *ranks.get(&piece[i..i + 2]).unwrap_or(&Rank::MAX);
        if rank < min_rank.0 {
            min_rank = (rank, i);
        }
        parts.push((i, rank));
    }
    parts.push((piece.len() - 1, Rank::MAX));
    parts.push((piece.len(), Rank::MAX));

    let get_rank = {
        |parts: &Vec<(usize, Rank)>, i: usize| {
            if (i + 3) < parts.len() {
                // Similar to `piece[i..i + 2]` above. The +3 is because we haven't yet deleted
                // parts[i + 1], see comment in the main loop.
                *ranks
                    .get(&piece[parts[i].0..parts[i + 3].0])
                    .unwrap_or(&Rank::MAX)
            }
            else {
                Rank::MAX
            }
        }
    };

    // If you have n parts and m merges, this does 0(mn) work.
    // We could do something with a heap and do 0(m log n) work.
    // n is often very small so considerations like cache-locality outweigh the algorithmic
    // complexity downsides of the `parts` vector.
    while min_rank.0 != Rank::MAX {
        let i = min_rank.1;
        // Update parts[i] and parts[i - 1] before removing parts[i + 1], since
        // `parts.remove(i + 1)` will thrash the cache.
        if i > 0 {
            parts[i - 1].1 = get_rank(&parts, i - 1);
        }
        parts[i].1 = get_rank(&parts, i);
        parts.remove(i + 1);

        min_rank = (Rank::MAX, usize::MAX);
        for (i, &(_, rank)) in parts[..parts.len() - 1].iter().enumerate() {
            if rank < min_rank.0 {
                min_rank = (rank, i);
            }
        }
    }
    parts
}

pub fn bytes_pair_encode(piece: &[u8], ranks: &HashMap<Vec<u8>, Rank>) -> Vec<Rank> {
    assert!(piece.len() > 1);
    _byte_pair_merge(&ranks, &piece)
        .windows(2)
        .map(|part| ranks[&piece[part[0].0..part[1].0]])
        .collect()
}

pub fn bytes_pair_split<'a>(piece: &'a [u8], ranks: &HashMap<Vec<u8>, Rank>) -> Vec<&'a [u8]> {
    assert!(piece.len() > 1);
    _byte_pair_merge(&ranks, &piece)
        .windows(2)
        .map(|part| &piece[part[0].0..part[1].0])
        .collect()
}

pub struct FakeThreadId(NonZeroU64);

fn hash_current_thread() -> usize {
    const _: [u8; 8] = [0; std::mem::size_of::<ThreadId>()];
    const _: [u8; 8] = [0; std::mem::size_of::<FakeThreadId>()];
    let x = unsafe {
        std::mem::transmute::<ThreadId, FakeThreadId>(thread::current().id()).0
    };
    u64::from(x) as usize
}

const MAX_NUM_THREADS: usize = 128;

#[derive(Clone)]
pub(super) struct CoreBPE {
    encoder: HashMap<Vec<u8>, Rank>,
    special_tokens_encoder: HashMap<String, Rank>,
    decoder: HashMap<Rank, Vec<u8>>,
    special_tokens_decoder: HashMap<Rank, Vec<u8>>,
    regex_tls: Vec<Regex>,
    special_regex_tls: Vec<Regex>,
    sorted_token_bytes: Vec<Vec<u8>>,
}

impl CoreBPE {
    pub(super) fn new(
        encoder: HashMap<Vec<u8>, Rank>,
        special_tokens_encoder: HashMap<String, Rank>,
        pattern: &str,
    ) -> CounterResult<Self> {
        let regex = Regex::new(pattern).map_err(|e| CounterError::RegexError(e.to_string()))?;

        let special_regex = {
            let _parts = special_tokens_encoder
                .keys()
                .map(|s| fancy_regex::escape(s))
                .collect::<Vec<_>>();
            Regex::new(&_parts.join("|"))
                .map_err(|e| CounterError::RegexError(e.to_string()))?
        };

        let decoder: HashMap<Rank, Vec<u8>> = encoder
            .iter()
            .map(|(key, value)| (*value, key.clone())).collect();

        assert_eq!(
            encoder.len(), decoder.len(),
            "Encoder and Decoder must have the same length. Please check encoder that may have duplicate tokens."
        );

        let special_tokens_decoder: HashMap<Rank, Vec<u8>> = special_tokens_encoder
            .iter()
            .map(|(key, value)| (*value, key.as_bytes().to_vec()))
            .collect();

        // Clone because I don't know how to tell Rust I'm not going to change the map
        let mut sorted_token_bytes: Vec<Vec<u8>> = encoder.keys().cloned().collect();
        sorted_token_bytes.sort();

        Ok(Self {
            encoder,
            special_tokens_encoder,
            decoder,
            special_tokens_decoder,
            regex_tls: (0..MAX_NUM_THREADS).map(|_| regex.clone()).collect(),
            special_regex_tls: (0..MAX_NUM_THREADS).map(|_| special_regex.clone()).collect(),
            sorted_token_bytes,
        })
    }

    // ===================
    // Encoding
    // ===================

    pub(crate) fn encode_ordinary(&self, text: &str) -> Vec<Rank> {
        let self_clone = self.clone();
        let text = text.to_owned();

        thread::spawn(move || self_clone.encode_ordinary_native(&text)).join().unwrap_or_else(|_| {
            eprintln!("encode failed");
            Vec::new()
        })
    }

    pub(crate) fn encode(&self, text: &str, allowed_special: HashSet<String>) -> Vec<Rank> {
        let self_clone = self.clone();
        let text = text.to_owned();
        let allowed_special = allowed_special.iter().map(|special| special.to_string()).collect::<HashSet<String>>();

        thread::spawn(move || self_clone.encode_native(&text, &allowed_special).0).join().unwrap_or_else(|_| {
            eprintln!("encode failed");
            Vec::new()
        })
    }

    pub(crate) fn encode_with_unstable(
        &self,
        text: &str,
        allowed_special: HashSet<String>,
    ) -> (Vec<Rank>, Vec<Vec<Rank>>) {
        let self_clone = self.clone();
        let text = text.to_owned();
        let allowed_special = allowed_special.iter().map(|special| special.to_string()).collect::<HashSet<String>>();

        let (tokens, completions) = thread::spawn(move || self_clone.encode_unstable_native(&text, &allowed_special)).join().unwrap_or_else(|_| {
            eprintln!();
            (Vec::new(), HashSet::new())
        });

        let completions_vec = Vec::from_iter(completions.iter().map(|seq| seq.to_owned()));
        (tokens, completions_vec)
    }

    pub(crate) fn encode_single_token(&self, piece: &[u8]) -> CounterResult<Rank> {
        if let Some(token) = self.encoder.get(piece).copied() {
            return Ok(token);
        }
        if let Ok(piece_str) = from_utf8(piece) {
            if let Some(token) = self.special_tokens_encoder.get(piece_str).copied() {
                return Ok(token);
            }
        }
        Err(CounterError::KeyError(format!("{:?}", piece)))
    }

    pub(crate) fn encode_single_piece(&self, piece: &[u8]) -> Vec<Rank> {
        if let Some(token) = self.encoder.get(piece) {
            return vec![*token];
        }
        bytes_pair_encode(piece, &self.encoder)
    }

    // ================
    // Decoding
    // ================

    pub(crate) fn decode_bytes(&self, tokens: Vec<Rank>) -> Vec<u8> {
        let self_clone = self.clone();
        thread::spawn(move || self_clone.decode_native(&tokens)).join().unwrap_or_else(|_| {
            eprintln!("Decode failed");
            Vec::new()
        })
    }

    pub(crate) fn decode_single_token_bytes(&self, token: Rank) -> CounterResult<Vec<u8>> {
        if let Some(bytes) = self.decoder.get(&token) {
            return Ok(bytes.to_owned());
        }
        if let Some(bytes) = self.special_tokens_decoder.get(&token) {
            return Ok(bytes.to_owned());
        }

        Err(CounterError::KeyError(token.to_string()))
    }

    // =================
    // Miscellaneous
    // =================

    pub(crate) fn token_byte_values(&self) -> Vec<Vec<u8>> {
        self.sorted_token_bytes
            .iter()
            .cloned()
            .collect()
    }

    fn get_tl_regex(&self) -> &Regex {
        &self.regex_tls[hash_current_thread() % MAX_NUM_THREADS]
    }

    fn get_tl_special_regex(&self) -> &Regex {
        &self.special_regex_tls[hash_current_thread() % MAX_NUM_THREADS]
    }

    fn decode_native(&self, tokens: &[Rank]) -> Vec<u8> {
        let mut ret = Vec::with_capacity(tokens.len() * 2);
        for token in tokens {
            let token_bytes = self
                .decoder
                .get(token)
                .unwrap_or_else(|| &self.special_tokens_decoder[token]);
            ret.extend(token_bytes);
        }
        ret
    }

    fn encode_ordinary_native(&self, text: &str) -> Vec<Rank> {
        let regex = self.get_tl_regex();
        let mut ret = Vec::<Rank>::new();
        for mat in regex.find_iter(text) {
            if let Ok(mat) = mat {
                let piece = mat.as_str().as_bytes();
                match self.encoder.get(piece) {
                    Some(token) => ret.push(*token),
                    None => ret.extend(&bytes_pair_encode(piece, &self.encoder)),
                }
            }
            else {
                panic!()
            }
        }

        ret
    }

    fn encode_native(&self, text: &str, allowed_special: &HashSet<String>) -> (Vec<Rank>, usize) {
        let special_regex = self.get_tl_special_regex();
        let regex = self.get_tl_regex();
        let mut ret = Vec::<Rank>::new();

        let mut start = 0;
        let mut last_piece_token_len = 0;

        loop {
            let mut next_special;
            let mut start_find = start;

            loop {
                next_special = special_regex.find_from_pos(text, start_find).unwrap();
                match next_special {
                    Some(m) => {
                        if allowed_special.contains(&text[m.start()..m.end()]) {
                            break;
                        }
                        start_find = m.start() + 1;
                    }
                    None => break,
                }
            }
            let end = next_special.map_or(text.len(), |m| m.start());

            for mat in regex.find_iter(&text[start..end]) {
                let piece = mat.unwrap().as_str().as_bytes();
                if let Some(token) = self.encoder.get(piece) {
                    last_piece_token_len = 1;
                    ret.push(*token);
                    continue;
                }
                let tokens = bytes_pair_encode(piece, &self.encoder);
                last_piece_token_len = tokens.len();
                ret.extend(&tokens);
            }

            match next_special {
                Some(m) => {
                    let piece = m.as_str();
                    let token = self.special_tokens_encoder[piece];
                    ret.push(token);
                    last_piece_token_len = 0;
                }
                None => break,
            }
        }
        (ret, last_piece_token_len)
    }

    fn increase_last_piece_token_len(
        &self,
        tokens: Vec<Rank>,
        mut last_piece_token_len: usize
    ) -> (Vec<Rank>, usize) {
        {
            let token_is_all_space = |token| {
                self.decoder
                    .get(token)
                    .map(|token_bytes| {
                        token_bytes
                            .iter()
                            .rev()
                            .all(|&b| [b' ', b'\n', b'\t'].contains(&b))
                    })
                    .unwrap_or(false)
            };
            if last_piece_token_len > 0 && token_is_all_space(&tokens[tokens.len() - last_piece_token_len]) {
                while (last_piece_token_len < tokens.len()) && token_is_all_space(&tokens[tokens.len() - last_piece_token_len - 1]) {
                    last_piece_token_len += 1;
                }
            }
        }
        debug_assert!(last_piece_token_len <= tokens.len());

        (tokens, last_piece_token_len)
    }

    fn encode_unstable_native(
        &self,
        text: &str,
        allowed_special: &HashSet<String>
    ) -> (Vec<Rank>, HashSet<Vec<Rank>>) {
        let (tokens, last_piece_token_len) = self.encode_native(text, allowed_special);

        if last_piece_token_len == 0 {
            // If last_piece_token_len is zero, the last token was a special token, and we have no unstable bytes.
            return (tokens, HashSet::new())
        }
        let (mut tokens, last_piece_token_len) = self.increase_last_piece_token_len(tokens, last_piece_token_len);

        let unstable_bytes = self.decode_native(&tokens[(tokens.len() - last_piece_token_len)..]);
        tokens.truncate(tokens.len() - last_piece_token_len);

        // TODO: we should try harder to find additional stable tokens
        // This would reduce the amount of re-tokenize when determining completions
        // Refer to the logic in an older version of this file.

        let mut completions = HashSet::new();
        if unstable_bytes.is_empty() {
            return (tokens, completions);
        }

        let mut point = self.sorted_token_bytes
            .partition_point(|x| x.as_slice() < unstable_bytes.as_slice());

        while point < self.sorted_token_bytes.len() && self.sorted_token_bytes[point].starts_with(&unstable_bytes) {
            completions.insert(vec![
                self.encoder[self.sorted_token_bytes[point].as_slice()]
            ]);
            point += 1;
        }

        // Now apply even more brute force. At every (other) possible position for the straddling token,
        // concatenate additional bytes from that token (if any) to unstable_bytes,
        // and re-tokenize the whole thing and see what we get.

        for i in 1..unstable_bytes.len() {
            let prefix = &unstable_bytes[..i];
            let suffix = &unstable_bytes[i..];
            let mut point = self.sorted_token_bytes
                .partition_point(|x| x.as_slice() < suffix);
            // TODO: Perf optimisation if suffix starts with " "?
            while point < self.sorted_token_bytes.len() && self.sorted_token_bytes[point].starts_with(suffix) {
                let possibility = [prefix, self.sorted_token_bytes[point].as_slice()].concat();
                let encoded = match from_utf8(&possibility) {
                    Ok(s) => self.encode_ordinary_native(s),
                    Err(_) => bytes_pair_encode(&possibility, &self.encoder),
                };
                let mut seq = Vec::new();
                let mut seq_len = 0;
                for token in encoded {
                    seq.push(token);
                    seq_len += self.decoder[&token].len();
                    if seq_len >= unstable_bytes.len() {
                        break;
                    }
                }
                completions.insert(seq);
                point += 1;
            }
        }

        if unstable_bytes.len() > 1 {
            let last_decode = bstr::decode_last_utf8(unstable_bytes.as_slice());
            if unstable_bytes.len() - last_decode.1 > 0 && last_decode.0.map_or(false, |c| c.is_whitespace()) {
                let mut reencoded = bytes_pair_encode(
                    &unstable_bytes[..(unstable_bytes.len() - last_decode.1)],
                    &self.encoder,
                );
                reencoded.extend(bytes_pair_encode(
                    &unstable_bytes[(unstable_bytes.len() - last_decode.1)..],
                    &self.encoder,
                ));
                completions.insert(reencoded);
            }
        }
        (tokens, completions)
    }
}

#[cfg(test)]
mod tests {
    use rustc_hash::FxHashMap as HashMap;

    use crate::counter::token::{bytes_pair_split, Rank};

    fn setup_ranks() -> HashMap<Vec<u8>, Rank> {
        HashMap::from_iter([
            (b"ab".to_vec(), 0),
            (b"cd".to_vec(), 1),
        ])
    }

    #[test]
    fn test_simple_characters() {
        let ranks = setup_ranks();
        let res = bytes_pair_split(b"abcd", &ranks);
        assert_eq!(res, vec![b"ab", b"cd"]);
    }

    #[test]
    fn test_repeated_characters() {
        let ranks = setup_ranks();
        let res = bytes_pair_split(b"abab", &ranks);
        assert_eq!(res, vec![b"ab", b"ab"]);
    }
}
