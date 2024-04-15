use crate::tokenizer::openai::OpenAI;
use crate::tokenizer::openai::openai_sets::Models;
use crate::errors::{CounterError, CounterResult};

const MODEL_PREFIX_TO_CL100K_BASE: [&str; 7] = [
    "gpt-4-",
    "gpt-3.5-turbo-",
    "gpt-35-turbo-",
    "ft:gpt-4",
    "ft:gpt-3.5",
    "ft:davinci-002",
    "ft:babbage-002"];

const CL100K_BASE: [&str; 9] = [
    // chat
    "gpt-4",
    "gpt-3.5-turbo",
    "gpt-3.5",
    // chat from azure deployment
    "gpt-35-turbo",
    // base
    "davinci-002",
    "babbage-002",
    // embeddings
    "text-embedding-ada-002",
    "text-embedding-3-small",
    "text-embedding-3-large",
];

const P50K_BASE: [&str; 8] = [
    // text (DEPRECATED)
    "text-davinci-003",
    "text_davinci-002",
    // code (DEPRECATED)
    "code-davinci-002",
    "code-davinci-001",
    "code-cushman-002",
    "code-cushman-001",
    "davinci-codex",
    "cushman-codex",
];

const R50K_BASE: [&str; 18] = [
    // text (DEPRECATED)
    "text-davinci-001",
    "text-curie-001",
    "text-babbage-001",
    "text-ada-001",
    "davinci",
    "curie",
    "babbage",
    "ada",
    // old embeddings (DEPRECATED)
    "text-similarity-davinci-001",
    "text-similarity-curie-001",
    "text-similarity-babbage-001",
    "text-similarity-ada-001",
    "text-search-davinci-doc-001",
    "text-search-curie-doc-001",
    "text-search-babbage-doc-001",
    "text-search-ada-doc-001",
    "text-search-babbage-code-001",
    "text-search-ada-code-001",
];

const P50K_EDIT: [&str; 2] = [
    // edit (DEPRECATED)
    "text-davinci-edit-001",
    "code-davinci-edit-001",
];

const GPT2: [&str; 2] = [
    // open source
    "gpt2",
    "gpt-2",
];

/// Returns the name of the encoding used by a model user
pub fn encoding_name_for_model(model_name: &str) -> CounterResult<String> {
    let encoding_name = if CL100K_BASE.contains(&model_name) || MODEL_PREFIX_TO_CL100K_BASE.iter().any(|candidate_model| model_name.starts_with(candidate_model)) {
        "cl100k_base"
    }
    else if P50K_BASE.contains(&model_name) {
        "p50k_base"
    }
    else if R50K_BASE.contains(&model_name) {
        "r50k_base"
    }
    else if P50K_EDIT.contains(&model_name) {
        "p50k_edit"
    }
    else if GPT2.contains(&model_name) {
        "gpt2"
    }
    else {
        return Err(CounterError::ModelNotFound(model_name.to_string()))
    };

    Ok(encoding_name.to_string())
}

pub fn encoding_for_model(model_name: &str) -> CounterResult<OpenAI> {
    let encoding_name = encoding_name_for_model(model_name)?;
    let model = Models::try_from(encoding_name)?;
    let input = model.get_input()?;

    OpenAI::try_from(input)
}

mod test {
    use crate::tokenizer::openai::models::encoding_for_model;
    use crate::tokenizer::openai::Specials;

    #[test]
    fn test_encoding() {

        let text = "GMOアドマーケティング";
        let encoder = encoding_for_model("gpt-4").unwrap();
        let tokens = encoder.encode(text, Specials::Collection(&[]), Specials::All).unwrap();
        let token_count = tokens.len();

        eprintln!("{:?}", tokens);
        assert_eq!(token_count, 10)
    }
}
