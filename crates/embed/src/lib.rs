mod embedder;
mod model;

pub use embedder::Embedder;
pub use model::{
    EmbedError, ModelManager, PrintHashesReport, StatusReport, VerifyDetail, VerifyFile,
    VerifyReport, VerifyState,
};
pub use pale_ale_modelspec::{ModelFileSpec, ModelSpec};
