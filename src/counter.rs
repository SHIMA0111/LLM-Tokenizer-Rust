use std::cmp::max;
use std::collections::{HashMap, HashSet};
use std::fmt::{Display, Formatter};
use std::sync::Arc;
use rustc_hash::FxHashMap;
use rayon::prelude::*;
use regex::Regex;
use crate::counter::openai::bpe::CoreBytePairEncoding;
use crate::errors::CounterResult;

mod openai;
mod utils;

pub trait Counter {
    fn count(&self) -> usize;
}

struct TokenEncoding<'a> {
    name: &'a str,
    pat_str: &'a str,
    mergeable_ranks: FxHashMap<Vec<u8>, u32>,
    special_tokens: FxHashMap<String, u32>,
    max_token_value: u32,
    core_bpe: Arc<CoreBytePairEncoding>,
}

#[derive(Clone, PartialEq)]
pub enum Specials<'a> {
    All,
    Collection(Vec<&'a str>),
}

pub enum SingleToken<'a> {
    String(&'a str),
    Bytes(Vec<u8>),
}

impl <'a> TokenEncoding<'a> {
    fn new(name: &'a str,
           pat_str: &'a str,
           mergeable_ranks: HashMap<Vec<u8>, u32>,
           special_tokens: HashMap<String, u32>,
           explicit_n_vocab: Option<u32>) -> CounterResult<TokenEncoding<'a>> {

        let fx_margeable_ranks = FxHashMap::from_iter(mergeable_ranks);
        let fx_special_token = FxHashMap::from_iter(special_tokens);

        let mergeable_ranks_max = if let Some(max_value) = fx_margeable_ranks.values().max() {
            *max_value
        } else {
            0
        };

        let special_tokens_max = if let Some(max_value) = fx_special_token.values().max() {
            *max_value
        } else {
            0
        };

        let max_token_value = max(mergeable_ranks_max, special_tokens_max);

        if let Some(explicit_vocab_num) = explicit_n_vocab {
            assert_eq!((fx_margeable_ranks.len() + fx_special_token.len()) as u32, explicit_vocab_num);
            assert_eq!(max_token_value, explicit_vocab_num - 1);
        }

        let core_bpe = 
            CoreBytePairEncoding::new(fx_margeable_ranks.clone(),
                         fx_special_token.clone(), 
                         pat_str)?;

        Ok(Self {
            name,
            pat_str,
            mergeable_ranks: fx_margeable_ranks,
            special_tokens: fx_special_token,
            max_token_value,
            core_bpe: Arc::new(core_bpe),
        })
    }

    // ================
    // Encoding
    // ================

    pub fn encode_ordinary(&self, text: &str) -> Vec<u32> {
        todo!()
    }

    pub fn encode(&self,
                  text: &str,
                  allowed_special: Specials,
                  disallowed_special: Specials
    ) -> CounterResult<Vec<u32>> {
        todo!()
    }

    pub fn encode_ordinary_batch(&self, text: &[&str]) -> Vec<Vec<u32>> {
        todo!()
    }

    pub fn encode_batch(&self,
                        text: &[&str],
                        allowed_special: Specials,
                        disallowed_special: Specials
    ) -> CounterResult<Vec<Vec<u32>>> {
        todo!()
    }

    pub fn encode_with_unstable(&self,
                                text: &str,
                                allowed_special: Specials,
                                disallowed_special: Specials
    ) -> CounterResult<(Vec<u32>, Vec<Vec<u32>>)> {
        todo!()
    }

    pub fn encode_single_token(self, text_or_bytes: SingleToken) -> CounterResult<u32> {
        todo!()
    }

    // ===========
    // Decoding
    // ===========

    pub fn special_tokens_set(&self) -> HashSet<String> {
        todo!()
    }

    fn special_token_regex(tokens: HashSet<String>) -> Regex {
        todo!()
    }

    fn parse_input_values(&self,
                          text: &str,
                          allowed_special: Specials,
                          disallowed_special: Specials
    ) -> CounterResult<HashSet<String>> {
        todo!()
    }
}

impl Display for TokenEncoding<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "<Encoding {}>", self.name)
    }
}
