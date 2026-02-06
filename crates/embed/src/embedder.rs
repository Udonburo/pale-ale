use crate::{EmbedError, ModelManager};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use std::fs;
use std::path::Path;
use tokenizers::tokenizer::{
    PaddingParams, PaddingStrategy, Tokenizer, TruncationParams, TruncationStrategy,
};

const MAX_SEQ_LEN: usize = 512;
const EMBED_DIM: usize = 384;

pub struct Embedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl Embedder {
    pub fn new(manager: &ModelManager) -> Result<Self, EmbedError> {
        let resolved_dir = manager.resolved_dir();
        let config_path = resolved_dir.join("config.json");
        let model_path = resolved_dir.join("model.safetensors");
        let tokenizer_path = resolved_dir.join("tokenizer.json");

        ensure_model_file_exists(&config_path)?;
        ensure_model_file_exists(&model_path)?;
        ensure_model_file_exists(&tokenizer_path)?;

        let config = load_config(&config_path)?;
        let device = Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&model_path),
                DType::F32,
                &device,
            )
        }
        .map_err(|err| EmbedError::ModelLoad {
            path: model_path.clone(),
            message: err.to_string(),
        })?;
        let model = BertModel::load(vb, &config).map_err(|err| EmbedError::ModelLoad {
            path: model_path,
            message: err.to_string(),
        })?;
        let tokenizer = load_tokenizer(&tokenizer_path)?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let encoding =
            self.tokenizer
                .encode(text, true)
                .map_err(|err| EmbedError::Tokenization {
                    message: err.to_string(),
                })?;
        if encoding.len() != MAX_SEQ_LEN {
            return Err(EmbedError::Tokenization {
                message: format!(
                    "unexpected token length {}; expected {}",
                    encoding.len(),
                    MAX_SEQ_LEN
                ),
            });
        }

        let input_ids = Tensor::new(encoding.get_ids(), &self.device)
            .and_then(|t| t.reshape((1, MAX_SEQ_LEN)))
            .map_err(|err| EmbedError::Tensor {
                message: err.to_string(),
            })?;

        let token_type_ids_vec: Vec<u32> = if encoding.get_type_ids().is_empty() {
            vec![0; MAX_SEQ_LEN]
        } else {
            encoding.get_type_ids().to_vec()
        };
        let token_type_ids = Tensor::new(token_type_ids_vec, &self.device)
            .and_then(|t| t.reshape((1, MAX_SEQ_LEN)))
            .map_err(|err| EmbedError::Tensor {
                message: err.to_string(),
            })?;

        let attention_mask_vec: Vec<f32> = encoding
            .get_attention_mask()
            .iter()
            .map(|v| *v as f32)
            .collect();
        let attention_mask = Tensor::new(attention_mask_vec, &self.device)
            .and_then(|t| t.reshape((1, MAX_SEQ_LEN)))
            .map_err(|err| EmbedError::Tensor {
                message: err.to_string(),
            })?;

        let hidden_state = self
            .model
            .forward(&input_ids, &token_type_ids)
            .map_err(|err| EmbedError::Inference {
                message: err.to_string(),
            })?;

        let pooled = masked_mean_pooling_v1(&hidden_state, &attention_mask)?;
        let normalized = l2_normalize(&pooled)?;
        let vector = normalized
            .squeeze(0)
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|err| EmbedError::Inference {
                message: err.to_string(),
            })?;
        if vector.len() != EMBED_DIM {
            return Err(EmbedError::Inference {
                message: format!(
                    "unexpected embedding dimension {}; expected {}",
                    vector.len(),
                    EMBED_DIM
                ),
            });
        }
        Ok(vector)
    }
}

fn ensure_model_file_exists(path: &Path) -> Result<(), EmbedError> {
    if path.exists() {
        return Ok(());
    }
    Err(EmbedError::ModelFileMissing {
        path: path.to_path_buf(),
        details: Vec::new(),
    })
}

fn load_config(path: &Path) -> Result<Config, EmbedError> {
    let bytes = fs::read(path).map_err(|err| EmbedError::Io {
        path: Some(path.to_path_buf()),
        message: err.to_string(),
    })?;
    serde_json::from_slice::<Config>(&bytes).map_err(|err| EmbedError::ConfigLoad {
        path: path.to_path_buf(),
        message: err.to_string(),
    })
}

fn load_tokenizer(path: &Path) -> Result<Tokenizer, EmbedError> {
    let mut tokenizer = Tokenizer::from_file(path).map_err(|err| EmbedError::TokenizerLoad {
        path: path.to_path_buf(),
        message: err.to_string(),
    })?;

    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::Fixed(MAX_SEQ_LEN),
        ..Default::default()
    }));

    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: MAX_SEQ_LEN,
            strategy: TruncationStrategy::LongestFirst,
            ..Default::default()
        }))
        .map_err(|err| EmbedError::TokenizerLoad {
            path: path.to_path_buf(),
            message: err.to_string(),
        })?;
    Ok(tokenizer)
}

fn masked_mean_pooling_v1(
    hidden_state: &Tensor,
    attention_mask: &Tensor,
) -> Result<Tensor, EmbedError> {
    let (batch, seq_len, hidden_dim) = hidden_state.dims3().map_err(|err| EmbedError::Tensor {
        message: err.to_string(),
    })?;
    let (mask_batch, mask_seq_len) = attention_mask.dims2().map_err(|err| EmbedError::Tensor {
        message: err.to_string(),
    })?;
    if batch != mask_batch || seq_len != mask_seq_len {
        return Err(EmbedError::Inference {
            message: format!(
                "attention mask shape [{mask_batch}, {mask_seq_len}] mismatches hidden state [{batch}, {seq_len}, {hidden_dim}]"
            ),
        });
    }

    let mask_expanded = attention_mask
        .unsqueeze(2)
        .map_err(|err| EmbedError::Tensor {
            message: err.to_string(),
        })?;
    let mask_broadcast = mask_expanded
        .broadcast_as((batch, seq_len, hidden_dim))
        .map_err(|err| EmbedError::Tensor {
            message: err.to_string(),
        })?;
    debug_assert_eq!(hidden_state.dims(), mask_broadcast.dims());

    let masked_embeddings = (hidden_state * &mask_broadcast).map_err(|err| EmbedError::Tensor {
        message: err.to_string(),
    })?;
    let sum_embeddings = masked_embeddings.sum(1).map_err(|err| EmbedError::Tensor {
        message: err.to_string(),
    })?;
    let sum_mask = mask_expanded.sum(1).map_err(|err| EmbedError::Tensor {
        message: err.to_string(),
    })?;
    let clamped_mask = sum_mask
        .clamp(1e-9, f32::MAX)
        .map_err(|err| EmbedError::Tensor {
            message: err.to_string(),
        })?;

    sum_embeddings
        .broadcast_div(&clamped_mask)
        .map_err(|err| EmbedError::Tensor {
            message: err.to_string(),
        })
}

fn l2_normalize(input: &Tensor) -> Result<Tensor, EmbedError> {
    let norm = input
        .sqr()
        .and_then(|t| t.sum_keepdim(1))
        .and_then(|t| t.sqrt())
        .and_then(|t| t.clamp(1e-12, f32::MAX))
        .map_err(|err| EmbedError::Tensor {
            message: err.to_string(),
        })?;
    input
        .broadcast_div(&norm)
        .map_err(|err| EmbedError::Tensor {
            message: err.to_string(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_masked_mean_pooling() {
        let device = Device::Cpu;
        let hidden = Tensor::from_slice(
            &[1.0_f32, 2.0, 3.0, 4.0, 100.0, 200.0, 300.0, 400.0],
            (1, 2, 4),
            &device,
        )
        .expect("hidden tensor");
        let mask = Tensor::from_slice(&[1.0_f32, 0.0], (1, 2), &device).expect("mask tensor");

        let pooled = masked_mean_pooling_v1(&hidden, &mask).expect("pooling");
        let values = pooled
            .squeeze(0)
            .and_then(|t| t.to_vec1::<f32>())
            .expect("values");
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
