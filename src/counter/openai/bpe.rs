use std::collections::HashSet;
use std::str::{from_utf8, from_utf8_unchecked};
use regex::Regex;
use rustc_hash::FxHashMap as HashMap;
use crate::errors::{CounterError, CounterResult};

type Rank = u32;

fn byte_pair_merge(ranks: &HashMap<Vec<u8>, Rank>, piece: &[u8]) -> Vec<(usize, Rank)> {
    let mut parts = Vec::with_capacity(piece.len() + 1);

    let mut min_rank: (Rank, usize) = (Rank::MAX, usize::MAX);
    for i in 0..piece.len() - 1 {
        let rank = *ranks.get(&piece[i..=i + 1]).unwrap_or(&Rank::MAX);
        if rank < min_rank.0 {
            min_rank = (rank, i)
        }

        parts.push((i, rank));
    }
    parts.push((piece.len() - 1, Rank::MAX));
    parts.push((piece.len(), Rank::MAX));

    let get_rank = {
        |parts: &Vec<(usize, Rank)>, i: usize| {
            if (i + 3) < parts.len() {
                *ranks.get(&piece[parts[i].0..=parts[i + 2].0])
                    .unwrap_or(&Rank::MAX)
            } else {
                Rank::MAX
            }
        }
    };

    while min_rank.0 != Rank::MAX {
        let i = min_rank.1;
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

fn byte_pair_encode(piece: &[u8], ranks: &HashMap<Vec<u8>, Rank>) -> Vec<Rank> {
    assert!(piece.len() > 1);
    byte_pair_merge(&ranks, piece)
        .windows(2)
        .map(|part| ranks[&piece[part[0].0..part[1].0]])
        .collect()
}

fn byte_pair_split<'a>(piece: &'a [u8], ranks: &HashMap<Vec<u8>, Rank>) -> Vec<&'a [u8]> {
    assert!(piece.len() > 1);
    byte_pair_merge(&ranks, &piece)
        .windows(2)
        .map(|part| &piece[part[0].0..part[1].0])
        .collect()
}

pub(crate) struct CoreBytePairEncoding {
    encoder: HashMap<Vec<u8>, Rank>,
    special_tokens_encoder: HashMap<String, Rank>,
    decoder: HashMap<Rank, Vec<u8>>,
    special_tokens_decoder: HashMap<Rank, Vec<u8>>,
    regex_tls: Regex,
    special_regex_tls: Regex,
    sorted_token_bytes: Vec<Vec<u8>>,
}

impl CoreBytePairEncoding {
    pub(crate) fn new(encoder: HashMap<Vec<u8>, Rank>,
           special_tokens_encoder: HashMap<String, Rank>,
           pattern: &str
    ) -> CounterResult<Self> {
        let regex_obj = Regex::new(pattern)
            .map_err(|e| CounterError::RegexError(e.to_string()))?;

        let special_regex = {
            let escaped_specials = special_tokens_encoder
                .keys()
                .map(|str| regex::escape(str))
                .collect::<Vec<_>>();
            Regex::new(&escaped_specials.join("|"))
                .map_err(|e| CounterError::RegexError(e.to_string()))?
        };

        let decoder = encoder
            .iter()
            .map(|(key, value)| (*value, key.clone()))
            .collect::<HashMap<Rank, Vec<u8>>>();

        assert_eq!(
            encoder.len(), decoder.len(),
            "Encoder and generated decoder from encoder must have equal length. \
            Maybe the encoder has duplicate token indices and generate process of decoder overrides some tokens."
        );

        let special_tokens_decoder = special_tokens_encoder
            .iter()
            .map(|(key, value)| (*value, key.as_bytes().to_vec()))
            .collect::<HashMap<Rank, Vec<u8>>>();

        let mut sorted_token_bytes = encoder.keys().cloned().collect::<Vec<Vec<_>>>();
        // Sort the all bytes vector to ascending order
        sorted_token_bytes.sort();

        Ok(CoreBytePairEncoding {
            encoder,
            special_tokens_encoder,
            decoder,
            special_tokens_decoder,
            regex_tls: regex_obj,
            special_regex_tls: special_regex,
            sorted_token_bytes,
        })
    }

    // =========
    // Encoding
    // =========

    pub(crate) fn encode_ordinary(&self, text: &str) -> Vec<Rank> {
        self.encode_ordinary_native(text)
    }

    pub(crate) fn encode(&self, text: &str, allowed_special: HashSet<&str>) -> Vec<Rank> {
        self.encode_native(text, &allowed_special).0
    }

    fn encode_bytes(&self, bytes: &[u8]) -> Vec<Rank> {
        match from_utf8(bytes) {
            Ok(text) => self.encode_ordinary_native(text),
            Err(e) => {
                let text = unsafe {from_utf8_unchecked(&bytes[..e.valid_up_to()])};
                let (tokens, last_piece_token_len) =
                    self.encode_native(text, &HashSet::new());
                let (mut tokens, last_piece_token_len) =
                    self.increase_last_piece_token_len(tokens, last_piece_token_len);

                if !tokens.is_empty() && last_piece_token_len > 0 {
                    // This method can't say correct when niche case that regex split
                    // the valid UTF-8 and the invalid bytes. So this method should be private.
                    let mut unstable_bytes = self.decode_native(&tokens[tokens.len() - last_piece_token_len..]);
                    unstable_bytes.extend_from_slice(&bytes[e.valid_up_to()..]);

                    tokens.truncate(tokens.len() - last_piece_token_len);
                    match self.encoder.get(&unstable_bytes) {
                        Some(token) => tokens.push(*token),
                        None => tokens.extend(&byte_pair_encode(&unstable_bytes, &self.encoder)),
                    }
                }
                tokens

            }
        }
    }

    pub(crate) fn encode_with_unstable(&self,
                                       text: &str,
                                       allowed_special: HashSet<&str>
    ) -> (Vec<Rank>, Vec<Vec<Rank>>) {
        let (tokens, completions) =
            self.encode_unstable_native(text, &allowed_special);

        let completions = completions.iter().cloned().collect::<Vec<Vec<_>>>();

        (tokens, completions)
    }

    pub(crate) fn encode_single_token(&self, piece: &[u8]) -> CounterResult<u32> {
        if let Some(token) = self.encoder.get(piece) {
            return Ok(*token)
        }
        if let Ok(special_str) = from_utf8(piece) {
            if let Some(special_token) = self.special_tokens_encoder.get(special_str) {
                return Ok(*special_token)
            }
        }

        Err(CounterError::KeyError(format!("{:?}", piece)))
    }

    pub(crate) fn encode_single_piece(&self, piece: &[u8]) -> Vec<Rank> {
        if let Some(token) = self.encoder.get(piece) {
            vec![*token]
        }
        else {
            byte_pair_encode(piece, &self.encoder)
        }
    }

    // =========
    // Decoding
    // =========

    // =========
    // Internal
    // =========

    fn decode_native(&self, tokens: &[Rank]) -> Vec<u8> {
        let mut ret = Vec::with_capacity(tokens.len() * 2);
        for token in tokens {
            let token_bytes = self.decoder
                .get(token)
                .unwrap_or_else(|| &self.special_tokens_decoder[token]);
            ret.extend(token_bytes);
        }
        ret
    }

    fn encode_ordinary_native(&self, text: &str) -> Vec<Rank> {
        let regex = &self.regex_tls;
        let mut ret = vec![];

        for mat in regex.find_iter(text) {
            let piece = mat.as_str().as_bytes();
            match self.encoder.get(piece) {
                Some(token) => ret.push(*token),
                None => ret.extend(&byte_pair_encode(piece, &self.encoder)),
            }
        }
        ret
    }

    fn encode_native(&self, text: &str, allowed_special: &HashSet<&str>) -> (Vec<Rank>, usize) {
        let special_regex = &self.special_regex_tls;
        let regex = &self.regex_tls;
        let mut ret = vec![];

        let mut start = 0;
        let mut last_piece_token_len = 0;

        loop {
            let mut next_special;
            let mut start_find = start;

            loop {
                next_special = special_regex.find_at(text, start_find);
                match next_special {
                    Some(special_pos) => {
                        if allowed_special
                            .contains(&text[special_pos.start()..special_pos.end()]) {
                            break
                        }
                        start_find = special_pos.start() + 1
                    }
                    None => break
                }
            }
            let end = next_special.map_or(text.len(), |special_pos| special_pos.start());

            for mat in regex.find_iter(&text[start..end]) {
                let piece = mat.as_str().as_bytes();
                if let Some(token) = self.encoder.get(piece) {
                    last_piece_token_len = 1;
                    ret.push(*token);
                    continue;
                }
                let tokens = byte_pair_encode(piece, &self.encoder);
                last_piece_token_len = tokens.len();
                ret.extend(&tokens);
            }

            match next_special {
                Some(special_pos) => {
                    let piece = special_pos.as_str();
                    let token = self.special_tokens_encoder[piece];
                    ret.push(token);
                    start = special_pos.end();
                    last_piece_token_len = 0;
                }
                None => break,
            }
        }
        (ret, last_piece_token_len)
    }

    fn increase_last_piece_token_len(&self,
                                     tokens: Vec<Rank>,
                                     mut last_piece_token_len: usize,
    ) -> (Vec<Rank>, usize) {
        {
            let token_is_all_space = |token| {
                self.decoder
                    .get(token)
                    .map(|token_bytes| {
                        token_bytes
                            .iter()
                            .rev()
                            .all(|&byte| [b' ', b'\n', b'\t'].contains(&byte))
                    }).unwrap_or(false)
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

    fn encode_unstable_native(&self,
                              text: &str,
                              allowed_special: &HashSet<&str>
    ) -> (Vec<Rank>, HashSet<Vec<Rank>>) {
        let (tokens, last_piece_token_len) = self.encode_native(text, allowed_special);
        if last_piece_token_len == 0 {
            // If last_piece_token_len is zero, the last token was a special token and we have no unstable bytes
            return (tokens, HashSet::new())
        }
        let (mut tokens, last_piece_token_len) =
            self.increase_last_piece_token_len(tokens, last_piece_token_len);

        let unstable_bytes = self.decode_native(&tokens[(tokens.len() - last_piece_token_len)..]);
        tokens.truncate(tokens.len() - last_piece_token_len);

        let mut completions = HashSet::new();
        if unstable_bytes.is_empty() {
            return (tokens, completions)
        }

        let mut point = self.sorted_token_bytes
            .partition_point(|x| x.as_slice() < unstable_bytes.as_slice());

        while point < self.sorted_token_bytes.len() && self.sorted_token_bytes[point].starts_with(&unstable_bytes) {
            completions.insert(vec![
                self.encoder[self.sorted_token_bytes[point].as_slice()],
            ]);
            point += 1;
        }

        for i in 1..unstable_bytes.len() {
            let prefix = &unstable_bytes[..i];
            let suffix = &unstable_bytes[i..];

            let mut point = self.sorted_token_bytes
                .partition_point(|x| x.as_slice() < suffix);

            while point < self.sorted_token_bytes.len() && self.sorted_token_bytes[point].starts_with(suffix) {
                let possibility = [prefix, self.sorted_token_bytes[point].as_slice()].concat();
                let encoded = match from_utf8(&possibility) {
                    Ok(str) => self.encode_ordinary_native(str),
                    Err(_) => byte_pair_encode(&possibility, &self.encoder),
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
            let last_decoded = bstr::decode_last_utf8(unstable_bytes.as_slice());
            if unstable_bytes.len() - last_decoded.1 > 0 && last_decoded.0.map_or(false, |char| char.is_whitespace()) {
                let mut re_encoded = byte_pair_encode(
                    &unstable_bytes[..unstable_bytes.len() - last_decoded.1],
                    &self.encoder,
                );
                re_encoded.extend(byte_pair_encode(
                    &unstable_bytes[unstable_bytes.len() - last_decoded.1..],
                    &self.encoder,
                ));
                completions.insert(re_encoded);
            }
        }

        (tokens, completions)
    }
}
