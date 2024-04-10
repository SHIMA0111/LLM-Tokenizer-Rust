use std::cmp::max;
use std::collections::{HashMap, HashSet};
use regex::Regex;
use rustc_hash::FxHashMap;
use crate::counter::openai::bpe::CoreBytePairEncoding;
use crate::errors::{CounterError, CounterResult};

pub(super) mod models;
pub(super) mod load;
pub(super) mod bpe;

#[derive(Clone, PartialEq)]
pub enum Specials<'a> {
    All,
    Collection(&'a[&'a str]),
}

pub enum SingleToken<'a> {
    String(&'a str),
    Bytes(&'a[u8]),
}

pub(crate) struct OpenAI<'a> {
    name: &'a str,
    pattern: &'a str,
    merge_ranks: FxHashMap<Vec<u8>, u32>,
    special_token: FxHashMap<String, u32>,
    max_token_value: u32,
    bpe_base: CoreBytePairEncoding,
}

impl <'a> OpenAI<'a> {
    fn new(name: &'a str,
           pattern_str: &'a str,
           merge_ranks: HashMap<Vec<u8>, u32>,
           special_tokens: HashMap<String, u32>,
           explicit_n_vocab: Option<u32>
    ) -> CounterResult<OpenAI<'a>> {
        let fx_ranks = FxHashMap::from_iter(merge_ranks);
        let fx_special_tokens = FxHashMap::from_iter(special_tokens);

        let max_merge_ranks = match fx_ranks.values().max() {
            Some(value) => *value,
            None => return Err(
                CounterError::ValueError(
                    "merge_ranks should have tokens mapping corresponding bytes \
                    but the input has 0 tokens.".to_string()))
        };

        let max_special_tokens = match fx_special_tokens.values().max() {
            Some(value) => *value,
            None => 0,
        };

        let max_token_value = max(max_merge_ranks, max_special_tokens);

        if let Some(explicit_vocabs) = explicit_n_vocab {
            assert_eq!((fx_ranks.len() + fx_special_tokens.len()) as u32, explicit_vocabs);
            assert_eq!(max_token_value, explicit_vocabs - 1);
        }

        let bpe =
            CoreBytePairEncoding::new(fx_ranks.clone(),
                                      fx_special_tokens.clone(),
                                      pattern_str)?;

        Ok(Self {
            name,
            pattern: pattern_str,
            merge_ranks: fx_ranks,
            special_token: fx_special_tokens,
            max_token_value,
            bpe_base: bpe,
        })
    }

    // ===========
    // Encoding
    // ===========

    pub fn encode_ordinary(&self, text: &str) -> Vec<u32> {
        self.bpe_base.encode_ordinary(text)
    }

    pub fn encode(&self,
                  text: &str,
                  allowed_special: Specials,
                  disallowed_special: Specials
    ) -> CounterResult<Vec<u32>> {
        let allowed_special = match allowed_special {
            Specials::All => self.special_tokens_set(),
            Specials::Collection(allowed_specials) => {
                allowed_specials
                    .iter()
                    .map(|special| *special)
                    .collect::<HashSet<_>>()
            }
        };

        let disallowed_special = match disallowed_special {
            Specials::All => {
                self.special_tokens_set()
                    .difference(&allowed_special)
                    .cloned()
                    .collect::<HashSet<_>>()
            }
            Specials::Collection(disallowed_specials) => {
                disallowed_specials
                    .iter()
                    .map(|special| *special)
                    .collect::<HashSet<_>>()
            }
        };

        if !disallowed_special.is_empty() {
            let regex = special_token_regex(disallowed_special)?;
            if let Some(match_value) = regex.find(text) {
                return Err(
                    CounterError::ValueError(
                        format!(
                            "Encountered text corresponding to disallowed special token {}.\n \
                                    If you want this text to be encoded as a special token, \
                                    pass the token as 'allowed_special'. \
                                    If you want to encode this as normal text, \
                                    disable the check for this token by passing \
                                    a disallowed specials set removing this token. \
                                    To disable this check for all tokens, \
                                    `Specials::Collection(&Vec::new())` as `disallowed_special`",
                            match_value.as_str()
                        )
                    ))
            }
        }

        Ok(self.bpe_base.encode(text, allowed_special))
    }

    pub fn encode_ordinary_batch(&self, text: &[&str]) -> Vec<Vec<u32>> {
        text.iter().map(|str| self.bpe_base.encode_ordinary(str)).collect::<Vec<_>>()
    }

    pub fn encode_batch(&self,
                        text: &[&str],
                        allowed_special: Specials,
                        disallowed_special: Specials
    ) -> CounterResult<Vec<Vec<u32>>> {
        let mut tokens = Vec::new();
        for str in text {
            tokens.push(
                self.encode(str, allowed_special.clone(), disallowed_special.clone())?);
        }

        Ok(tokens)
    }

    // ===================
    // Miscellaneous
    // ===================

    /// Returns the list of all token byte values.
    pub fn token_bytes_values(&self) -> Vec<Vec<u8>> {
        self.bpe_base.token_byte_values()
    }

    pub fn end_of_text_token(&self) -> u32 {
        self.special_token["<|endoftext|>"]
    }

    pub fn special_tokens_set(&self) -> HashSet<&str> {
        self.special_token.keys().map(|key| key.as_str()).collect::<HashSet<_>>()
    }

    /// For backwards compatibility.
    pub fn n_vocab(&self) -> u32 {
        self.max_token_value + 1
    }
}

fn special_token_regex(tokens: HashSet<&str>) -> CounterResult<Regex> {
    let regex_text = tokens
        .iter()
        .map(|token| token.to_string())
        .collect::<Vec<_>>()
        .join("|");

    Regex::new(regex_text.as_str()).map_err(|e| CounterError::RegexError(e.to_string()))
}
