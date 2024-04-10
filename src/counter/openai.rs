use std::cmp::max;
use std::collections::HashMap;
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
        todo!()
    }
}
