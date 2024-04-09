use std::cmp::max;
use std::collections::{HashMap, HashSet};
use std::fmt::{Display, Formatter};
use std::sync::Arc;
use rustc_hash::FxHashMap;
use rayon::prelude::*;
use crate::errors::{CounterError, CounterResult};

mod openai;

struct TokenEncoding<'a> {
    name: &'a str,
    pat_str: &'a str,
    mergeable_ranks: FxHashMap<Vec<u8>, u32>,
    special_tokens: FxHashMap<String, u32>,
    max_token_value: u32,
    core_bpe: Arc<CoreB>,
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
            CoreBPE::new(fx_margeable_ranks.clone(), 
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
        self.encode_ordinary(text)
    }

    pub fn encode(&self,
                  text: &str,
                  allowed_special: Specials,
                  disallowed_special: Specials
    ) -> CounterResult<Vec<u32>> {
        let allowed_special =
            self.parse_input_values(text, allowed_special, disallowed_special)?;

        Ok(self.core_bpe.encode(text, allowed_special))
    }

    pub fn encode_ordinary_batch(&self, text: &[&str]) -> Vec<Vec<u32>> {
        text.par_iter()
            .map(|str| self.encode_ordinary(str))
            .collect()
    }

    pub fn encode_batch(&self,
                        text: &[&str],
                        allowed_special: Specials,
                        disallowed_special: Specials
    ) -> CounterResult<Vec<Vec<u32>>> {
        let outputs = text.par_iter()
            .map(|str| self.encode(str, allowed_special.clone(), disallowed_special.clone()))
            .collect::<Vec<CounterResult<Vec<u32>>>>();

        let mut out_vec = Vec::new();

        for output in outputs {
            match output {
                Ok(vec) => out_vec.push(vec),
                Err(e) => return Err(e),
            }
        }

        Ok(out_vec)
    }

    pub fn encode_with_unstable(&self,
                                text: &str,
                                allowed_special: Specials,
                                disallowed_special: Specials
    ) -> CounterResult<(Vec<u32>, Vec<Vec<u32>>)> {
        let allowed_special =
            self.parse_input_values(text, allowed_special, disallowed_special)?;

        Ok(self.core_bpe.encode_with_unstable(text, allowed_special))
    }

    pub fn encode_single_token(self, text_or_bytes: SingleToken) -> u32 {
        match text_or_bytes {
            SingleToken::String(str) => {
                let bytes = str.as_bytes();
                self.core_bpe.encode_single_token(bytes)?
            },
            SingleToken::Bytes(byte) => self.core_bpe.encode_single_token(&byte)?,
        }
    }

    // ===========
    // Decoding
    // ===========

    pub fn special_tokens_set(&self) -> HashSet<String> {
        self.special_tokens.keys().into_iter().map(|token| token.to_owned()).collect::<HashSet<String>>()
    }

    fn special_token_regex(tokens: HashSet<String>) -> Regex {
        let inner = tokens.iter().map(|token| regex::escape(token)).collect::<Vec<String>>().join("|");
        Regex::new(&inner).unwrap()
    }

    fn parse_input_values(&self,
                          text: &str,
                          allowed_special: Specials,
                          disallowed_special: Specials
    ) -> CounterResult<HashSet<String>> {
        let allowed_special = match allowed_special {
            Specials::All => self.special_tokens_set(),
            Specials::Collection(vec) => vec.iter().map(|str| str.to_string()).collect::<HashSet<String>>(),
        };
        let disallowed_special = match disallowed_special {
            Specials::All => self.special_tokens_set().difference(&allowed_special).cloned().collect::<HashSet<_>>(),
            Specials::Collection(vec) => vec.iter().map(|str| str.to_string()).collect::<HashSet<String>>(),
        };

        if !disallowed_special.is_empty() {
            if let Ok(match_res) = Self::special_token_regex(disallowed_special.clone()).find(text) {
                if let Some(matches) = match_res {
                    return Err(CounterError::ValueError(format!(
                        "Encountered text corresponding to disallowed special token {}.\n \
                        If you want this text to be encoded as a special token, \
                        pass it to `allowed_special`, e.g. `Collection(vec![{}, ...])` as allowed_special.\n \
                        If you want this text to be encoded as normal text, pass \
                        `special_tokens_set.remove({})`.\n \
                        To disable this check for all special tokens, pass `Collection(Vec::new())` as disabled_special. \n",
                        matches.as_str(), matches.as_str(), matches.as_str()
                    )))
                }
            }
        }

        Ok(allowed_special)
    }
}

impl Display for TokenEncoding<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "<Encoding {}>", self.name)
    }
}
