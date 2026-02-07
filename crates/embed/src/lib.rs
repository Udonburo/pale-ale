mod embedder;
mod model;

pub use embedder::{EmbedOutput, Embedder, TokenizeMeta};
pub use model::{
    EmbedError, ModelManager, PrintHashesReport, StatusReport, VerifyDetail, VerifyFile,
    VerifyReport, VerifyState,
};
pub use pale_ale_modelspec::{ModelFileSpec, ModelSpec};
