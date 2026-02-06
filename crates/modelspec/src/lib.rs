use serde::{Deserialize, Serialize};

pub const CLASSIC_MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";
pub const CLASSIC_REVISION: &str = "e4ce9877abf3edfe10b0d82785e83bdcb973e22e";
pub const CLASSIC_BASE_URL: &str = "https://huggingface.co";
pub const HASH_MODEL_SAFETENSORS: &str =
    "8087e9bf97c265f8435ed268733ecf3791825ad24850fd5d84d89e32ee3a589a";
pub const HASH_TOKENIZER_JSON: &str =
    "82483bb4f0bdb81779f295ecc5a93285d2156834e994a2169f9800e4c8f250c1";
pub const HASH_CONFIG_JSON: &str =
    "02ba870d29dc00b373fe71bd273baca30586d6577def4130456e756c7b286890";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelFileSpec {
    pub path: String,
    pub blake3: String,
    pub size_bytes: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelSpec {
    pub model_id: String,
    pub revision: String,
    pub base_url: String,
    pub required_files: Vec<ModelFileSpec>,
}

impl ModelSpec {
    pub fn classic() -> Self {
        Self {
            model_id: CLASSIC_MODEL_ID.to_string(),
            revision: CLASSIC_REVISION.to_string(),
            base_url: CLASSIC_BASE_URL.to_string(),
            required_files: vec![
                ModelFileSpec {
                    path: "model.safetensors".to_string(),
                    blake3: HASH_MODEL_SAFETENSORS.to_string(),
                    size_bytes: None,
                },
                ModelFileSpec {
                    path: "tokenizer.json".to_string(),
                    blake3: HASH_TOKENIZER_JSON.to_string(),
                    size_bytes: None,
                },
                ModelFileSpec {
                    path: "config.json".to_string(),
                    blake3: HASH_CONFIG_JSON.to_string(),
                    size_bytes: None,
                },
            ],
        }
    }

    pub fn download_url(&self, file: &ModelFileSpec) -> String {
        format!(
            "{}/{}/resolve/{}/{}",
            self.base_url.trim_end_matches('/'),
            self.model_id,
            self.revision,
            file.path
        )
    }
}
