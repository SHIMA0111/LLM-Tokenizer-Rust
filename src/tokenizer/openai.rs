use std::cmp::max;
use std::collections::{HashMap, HashSet};
use std::str::from_utf8;
use regex::Regex;
use rustc_hash::FxHashMap;
use crate::tokenizer::openai::bpe::CoreBytePairEncoding;
use crate::tokenizer::utils::{from_utf8_backslash, from_utf8_ignore};
use crate::errors::{CounterError, CounterResult};

pub(super) mod models;
pub(super) mod load;
pub(super) mod bpe;
mod openai_sets;

/// When encode text, you can specify special characters as allowed or disallowed.
/// In the OpenAI encode methods, `allowed_special` is preferred so both of allowed and disallowed
/// is specified as `All`, all specials inputted as dictionary assign to `allowed_special` and
/// none of them to `disallowed_special`.
#[derive(Clone, PartialEq)]
pub enum Specials<'a> {
    All,
    Collection(&'a[&'a str]),
}

/// When you want to get single token, you can through `&str` or `&[u8](Bytes)`.
pub enum SingleInput<'a> {
    String(&'a str),
    Bytes(&'a[u8]),
}

/// Handle method of failing decode the bytes.
/// When use 'Strict', the process raise error when encounter the decoding error.
/// For 'Replace', the invalid bytes will be replaced with "\u{FFFD}", about 'Ignore',
/// the invalid bytes will be ignored.
/// If you select 'BackSlashReplace', the invalid each byte convert to escape sequence like '\xNN'.
#[derive(Copy, Clone)]
pub enum DecodeErrorHandler {
    Strict,
    Replace,
    Ignore,
    BackSlashReplace,
}

#[derive(Clone)]
pub(crate) struct OpenAIInput {
    name: String,
    pattern: String,
    merge_able_ranks: HashMap<Vec<u8>, u32>,
    special_tokens: HashMap<String, u32>,
    explicit_n_vocab: Option<u32>,
}

/// OpenAI API tokenizer struct based on BPE(Byte Pair Encoding)
/// This code based on the tiktoken (https://github.com/openai/tiktoken)
/// But current implementation doesn't support parallel execution.
pub(crate) struct OpenAI {
    name: String,
    pattern: String,
    merge_able_ranks: FxHashMap<Vec<u8>, u32>,
    special_token: FxHashMap<String, u32>,
    max_token_value: u32,
    bpe_base: CoreBytePairEncoding,
}

impl <'a> OpenAI {
    /// Constructs a new instance of `OpenAI` tokenizer.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the tokenizer.
    /// * `pattern_str` - The pattern string used for tokenization.
    /// * `merge_able_ranks` - A `HashMap` mapping byte sequences to merge ranks.
    /// * `special_tokens` - A `HashMap` mapping special tokens to ids.
    /// * `explicit_n_vocab` - An optional explicit number of vocabulary tokens.
    ///
    /// # Returns
    ///
    /// Returns a `CounterResult` that resolves to an `OpenAI` tokenizer on success.
    /// Otherwise, returns a `CounterError`.
    pub fn new(name: String,
               pattern_str: String,
               merge_able_ranks: HashMap<Vec<u8>, u32>,
               special_tokens: HashMap<String, u32>,
               explicit_n_vocab: Option<u32>
    ) -> CounterResult<Self> {
        let fx_ranks = FxHashMap::from_iter(merge_able_ranks);
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
                                      &pattern_str)?;

        Ok(Self {
            name,
            pattern: pattern_str,
            merge_able_ranks: fx_ranks,
            special_token: fx_special_tokens,
            max_token_value,
            bpe_base: bpe,
        })
    }

    // ===================
    // Encoding
    // ===================

    /// Encodes ordinary text into a sequence of tokens.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to be encoded.
    ///
    /// # Returns
    ///
    /// A vector of `u32` values representing encoded tokens.
    pub fn encode_ordinary(&self, text: &str) -> Vec<u32> {
        self.bpe_base.encode_ordinary(text)
    }

    /// Encodes the given text using the specified allowed and disallowed special characters.
    ///
    /// Returns a `Result` that contains a vector of encoded values on success, or an error message on failure.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to be encoded.
    /// * `allowed_special` - The allowed special characters to include in the encoding.
    /// * `disallowed_special` - The disallowed special characters to exclude from the encoding.
    ///
    /// # Returns
    ///
    /// A `Result` that contains a vector of encoded tokens on success,
    /// or an error message on failure on `CounterError`.
    pub fn encode(&self,
                  text: &str,
                  allowed_special: Specials<'_>,
                  disallowed_special: Specials<'_>
    ) -> CounterResult<Vec<u32>> {
        let allowed_special =
            self.validation_specials(text,
                                     allowed_special.clone(),
                                     disallowed_special)?;

        Ok(self.bpe_base.encode(text, allowed_special))
    }

    /// Encodes a batch of ordinary text into a Vec of tokens vector.
    ///
    /// # Arguments
    ///
    /// * `text` - A slice of string references representing the text to be encoded.
    ///
    /// # Returns
    ///
    /// A Vec constructed from Vec of u32, where each inner Vec corresponds
    /// to the encoded token of a text element in `text`.
    pub fn encode_ordinary_batch(&self, text: &[&str]) -> Vec<Vec<u32>> {
        text.iter().map(|str| self.bpe_base.encode_ordinary(str)).collect::<Vec<_>>()
    }

    /// Encodes a batch of text into a vector of encoded tokens.
    ///
    /// # Arguments
    ///
    /// * `text` - A slice of strings representing the text to be encoded.
    /// * `allowed_special` - An instance of `Specials` representing the special characters allowed during encoding.
    /// * `disallowed_special` - An instance of `Specials` representing the special characters disallowed during encoding.
    ///
    /// # Returns
    ///
    /// A `CounterResult` containing the vector of encoded tokens vec,
    /// or an error if encoding fails for any string in `text`.
    pub fn encode_batch(&self,
                        text: &[&str],
                        allowed_special: Specials<'a>,
                        disallowed_special: Specials<'a>
    ) -> CounterResult<Vec<Vec<u32>>> {
        let mut tokens = Vec::new();
        for str in text {
            tokens.push(
                self.encode(str, allowed_special.clone(), disallowed_special.clone())?);
        }

        Ok(tokens)
    }

    /// Encodes the given `text` using the unstable method.
    ///
    /// Special characters allowed in the encoding can be specified using the `allowed_special`
    /// parameter. Special characters that must not be included in the encoding can be specified
    /// using the `disallowed_special` parameter.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to be encoded.
    /// * `allowed_special` - `Specials` having the allowed special characters.
    /// * `disallowed_special` - `Specials` having the disallowed special characters.
    ///
    /// # Returns
    ///
    /// The result of the encoding, which consists of two vectors:
    /// * A vector of `u32` values representing the encoded text.
    /// * A vector of vectors of `u32` values representing completion candidate texts tokens.
    ///
    /// # Errors
    ///
    /// Returns a `CounterResult` with an error variant if the encoding fails.
    /// Possible error types include:
    /// * Invalid input text.
    /// * Presence of disallowed special characters, etc.
    pub fn encode_with_unstable(&self,
                                text: &str,
                                allowed_special: Specials<'a>,
                                disallowed_special: Specials<'a>
    ) -> CounterResult<(Vec<u32>, Vec<Vec<u32>>)> {
        let allowed_special =
            self.validation_specials(text,
                                     allowed_special.clone(),
                                     disallowed_special)?;

        Ok(self.bpe_base.encode_with_unstable(text, allowed_special))
    }

    /// Encodes a single input into a token.
    ///
    /// # Arguments
    ///
    /// * `text_or_bytes` - The single input represent encodable string or bytes to be encoded.
    ///
    /// # Returns
    ///
    /// Returns a `CounterResult` which contains the encoded token as a `u32`.
    pub fn encode_single_token(&self, text_or_bytes: SingleInput) -> CounterResult<u32> {
        match text_or_bytes {
            SingleInput::String(str) => {
                let bytes = str.as_bytes();
                self.bpe_base.encode_single_token(bytes)
            }
            SingleInput::Bytes(bytes) => {
                self.bpe_base.encode_single_token(bytes)
            }
        }
    }

    // ===================
    // Decoding
    // ===================

    /// Decode to vector of bytes from a given token.
    ///
    /// # Arguments
    ///
    /// * `token` - A slice of u32 values representing the tokens.
    ///
    /// # Returns
    ///
    /// Returns a bytes as vector of u8.
    pub fn decode_bytes(&self, token: &[u32]) -> Vec<u8> {
        self.bpe_base.decode_bytes(token)
    }

    /// Decodes a token into a string.
    ///
    /// # Arguments
    ///
    /// * `token` - The token to decode, represented as a slice of `u32` values.
    /// * `errors` - The error handling strategy when decoding fails.
    ///
    /// # Errors
    ///
    /// Returns a `CounterError` if decoding fails and the error handling strategy is set to `Strict`.
    ///
    /// # Returns
    ///
    /// Returns a `CounterResult` containing the decoded string on success,
    /// or the decoded replacement string based on the error handling strategy.
    pub fn decode(&self, token: &[u32], errors: DecodeErrorHandler) -> CounterResult<String> {
        let bytes = self.bpe_base.decode_bytes(token);

        let decoded_str = match from_utf8(&bytes) {
            Ok(decoded_str) => decoded_str.to_string(),
            Err(e) => {
                match errors {
                    DecodeErrorHandler::Strict => return Err(CounterError::ByteDecodeError(e.to_string())),
                    DecodeErrorHandler::Replace => {
                        String::from_utf8_lossy(&bytes).to_string()
                    }
                    DecodeErrorHandler::Ignore => from_utf8_ignore(&bytes).to_string(),
                    DecodeErrorHandler::BackSlashReplace => from_utf8_backslash(&bytes).to_string(),
                }
            }
        };
        Ok(decoded_str)
    }

    /// Decodes a single token into a vector of bytes.
    ///
    /// # Arguments
    ///
    /// * `token` - The token to decode.
    ///
    /// # Returns
    ///
    /// A `CounterResult` containing the decoded vector of bytes.
    pub fn decode_single_tokens_bytes(self, token: u32) -> CounterResult<Vec<u8>> {
        self.bpe_base.decode_single_token_bytes(token)
    }

    /// Decode a slice of tokens into a vector of byte vectors.
    ///
    /// # Arguments
    ///
    /// * `tokens` - A slice of u32 tokens representing the encoded tokens.
    ///
    /// # Returns
    ///
    /// A `CounterResult` containing vector of the decoded byte vectors,
    /// or an error if the decoding fails.
    pub fn decode_tokens_bytes(self, tokens: &[u32]) -> CounterResult<Vec<Vec<u8>>> {
        let mut res = Vec::new();

        for token in tokens {
            res.push(self.bpe_base.decode_single_token_bytes(*token)?);
        }

        Ok(res)
    }

    /// Decode the given tokens into text and offsets.
    ///
    /// This method takes an array of encoded tokens and returns the decoded text along with the
    /// corresponding offset positions of each token.
    ///
    /// # Arguments
    ///
    /// * `tokens` - An slice of u32 representing the encoded tokens.
    ///
    /// # Returns
    ///
    /// A `CounterResult` containing a tuple `(String, Vec<usize>)` representing the decoded text and
    /// the offset positions.
    ///
    /// # Errors
    ///
    /// Returns a `CounterError::ByteDecodeError` if the byte decoding fails.
    pub fn decode_with_offsets(self, tokens: &[u32]) -> CounterResult<(String, Vec<usize>)> {
        let token_bytes = self.decode_tokens_bytes(tokens)?;

        let mut text_len = 0;
        let mut offset: Vec<usize> = Vec::new();

        for token in &token_bytes {
            offset.push(max(0, text_len - (
                match 0x80 <= token[0] && token[0] < 0xC0 {
                    true => 1,
                    false => 0,
            })));
            text_len += token.iter().filter(|byte| !(0x80 <= **byte && **byte < 0xC0)).count();
        }

        let bytes = token_bytes.iter().flatten().cloned().collect::<Vec<_>>();
        let text = from_utf8(&bytes).map_err(|e| CounterError::ByteDecodeError(e.to_string()))?;

        Ok((text.to_string(), offset))
    }

    /// Decodes a batch of tokens into a vector of strings.
    ///
    /// # Arguments
    ///
    /// * `batch` - A slice of vectors containing the tokens to decode.
    /// * `errors` - The error handling strategy of decode failure.
    ///
    /// # Returns
    ///
    /// Returns the decoded strings as a `CounterResult<Vec<String>>`. If successful, the `Ok` variant
    /// contains the decoded strings. If an error occurs during decoding, the `Err` variant contains an
    /// error message.
    pub fn decode_batch(&self,
                        batch: &[Vec<u32>],
                        errors: DecodeErrorHandler
    ) -> CounterResult<Vec<String>> {
        let mut res_str = Vec::new();

        for token in batch {
            res_str.push(self.decode(token, errors)?);
        }

        Ok(res_str)
    }

    /// Decodes a slice of tokens vectors into corresponding bytes vector.
    ///
    /// # Arguments
    ///
    /// * `batch` - A slice of tokens vectors.
    ///
    /// # Returns
    ///
    /// A vector of bytes, where each byte sequence represents the decoded version of a tokenized sequence.
    pub fn decode_bytes_batch(&self, batch: &[Vec<u32>]) -> Vec<Vec<u8>> {
        let mut res_bytes = Vec::new();

        for token in batch {
            res_bytes.push(self.decode_bytes(token));
        }

        res_bytes
    }

    // ===================
    // Miscellaneous
    // ===================

    /// Returns the list of all token byte values.
    pub fn token_bytes_values(&self) -> Vec<Vec<u8>> {
        self.bpe_base.token_byte_values()
    }

    /// Returns the end-of-text token.
    ///
    /// # Returns
    ///
    /// The end-of-text token as an unsigned 32-bit integer.
    pub fn end_of_text_token(&self) -> u32 {
        self.special_token["<|endoftext|>"]
    }

    /// All special tokens set
    pub fn special_tokens_set(&self) -> HashSet<&str> {
        self.special_token.keys().map(|key| key.as_str()).collect::<HashSet<_>>()
    }

    /// For backwards compatibility.
    pub fn n_vocab(&self) -> u32 {
        self.max_token_value + 1
    }

    fn validation_specials(&'a self,
                           text: &str,
                           allowed_special: Specials<'a>,
                           disallowed_special: Specials<'a>
    ) -> CounterResult<HashSet<&'a str>> {
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
        Ok(allowed_special)
    }
}

impl <'a> TryFrom<OpenAIInput> for OpenAI {
    type Error = CounterError;

    fn try_from(value: OpenAIInput) -> Result<OpenAI, Self::Error> {
        Self::new(
            value.name,
            value.pattern,
            value.merge_able_ranks,
            value.special_tokens,
            value.explicit_n_vocab,
        )
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
