use std::collections::HashMap;
use crate::counter::openai::load::{data_gym_to_mergeable_bpe_ranks, load_bpe};
use crate::counter::openai::OpenAIInput;
use crate::errors::{CounterError, CounterResult};

const ENDOFTEXT: &str = "<|endoftext|>";
const FIM_PREFIX: &str = "<|fim_prefix|>";
const FIM_MIDDLE: &str = "<|fim_middle|>";
const FIM_SUFFIX: &str = "<|fim_middle|>";
const ENDOFPROMPT: &str = "<|endofprompt|>";


#[derive(Copy, Clone)]
pub enum Models {
    GPT2,
    R50KBase,
    P50KBase,
    P50KEdit,
    CL100KBase,
}

impl Models {
    pub fn get_input(&self) -> CounterResult<OpenAIInput<'_>> {
        match self {
            Self::GPT2 => {
                let merge_able_ranks = data_gym_to_mergeable_bpe_ranks(
                    "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe",
                    "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json",
                    Some("1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5"),
                    Some("196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783"),
                )?;

                Ok(OpenAIInput {
                    name: "gpt2",
                    pattern: r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
                    merge_able_ranks,
                    special_tokens: [(ENDOFTEXT.to_string(), 50256)].iter().cloned().collect(),
                    explicit_n_vocab: Some(50257),
                })
            }
            Self::R50KBase => {
                let merge_able_ranks = load_bpe(
                    "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken",
                    Some("306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930"),
                )?;

                Ok(OpenAIInput {
                    name: "r50k_base",
                    pattern: r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
                    merge_able_ranks,
                    special_tokens: [(ENDOFTEXT.to_string(), 50256)].iter().cloned().collect(),
                    explicit_n_vocab: Some(50257),
                })
            }
            Self::P50KBase => {
                let merge_able_ranks = load_bpe(
                    "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken",
                    Some("94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069"),
                )?;

                Ok(OpenAIInput {
                    name: "p50k_base",
                    pattern: r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
                    merge_able_ranks,
                    special_tokens: [(ENDOFTEXT.to_string(), 50256)].iter().cloned().collect(),
                    explicit_n_vocab: Some(50281),
                })
            }
            Self::P50KEdit => {
                let merge_able_ranks = load_bpe(
                    "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken",
                    Some("94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069"),
                )?;

                let special_tokens = [
                    (ENDOFTEXT.to_string(), 50256),
                    (FIM_PREFIX.to_string(), 50281),
                    (FIM_MIDDLE.to_string(), 50282),
                    (FIM_SUFFIX.to_string(), 50283),
                ].iter().cloned().collect::<HashMap<_, u32>>();

                Ok(OpenAIInput {
                    name: "p50k_edit",
                    pattern: r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
                    merge_able_ranks,
                    special_tokens,
                    explicit_n_vocab: None,
                })
            }
            Self::CL100KBase => {
                let merge_able_ranks = load_bpe(
                    "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
                    Some("223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"),
                )?;

                let special_tokens = [
                    (ENDOFTEXT.to_string(), 100257),
                    (FIM_PREFIX.to_string(), 100258),
                    (FIM_MIDDLE.to_string(), 100259),
                    (FIM_SUFFIX.to_string(), 100260),
                    (ENDOFPROMPT.to_string(), 100276)
                ].iter().cloned().collect::<HashMap<_, u32>>();

                Ok(OpenAIInput {
                    name: "cl100k_base",
                    pattern: r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+",
                    merge_able_ranks,
                    special_tokens,
                    explicit_n_vocab: None,
                })
            }
        }
    }
}

impl TryFrom<String> for Models {
    type Error = CounterError;

    fn try_from(value: String) -> Result<Self, Self::Error>   {
        let model =  match value.as_str() {
            "gpt2" => Self::GPT2,
            "r50k_base" => Self::R50KBase,
            "p50k_base" => Self::P50KBase,
            "p50k_edit" => Self::P50KEdit,
            "cl100k_base" => Self::CL100KBase,
            _ => return Err(CounterError::ValueError(format!("'{}' model not found from the openai tokenizers.", value))),
        };

        Ok(model)
    }
}
