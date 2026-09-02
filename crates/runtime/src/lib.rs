mod chat_template;
mod gated_delta;
pub(crate) mod gguf;
mod gguf_tokenizer;
mod model_owned;
mod model_resolver;
mod pinned_model;
mod qwen38_plan;
mod qwen3_5;
#[cfg(any(feature = "dflash2", test))]
mod qwen3_5_dflash;
mod qwen3_5_mtp;
mod qwen3_5_weights;
mod qwen3_next;
mod qwen3_vl_vision;
mod qwen_mrope;
mod qwen_mrope_state;
mod qwen_vision_rope;
mod qwen_vl;
mod qwen_vl_merge;
mod qwen_vl_position;
mod qwen_vl_processor;
mod sha256;
#[cfg(any(feature = "specprefill", test))]
mod specprefill;

mod portable_snapshot;
pub mod provider;

pub use mlxcel_core::cache::KVCacheMode;
pub use pinned_model::{
    PINNED_ARTIFACTS, PINNED_MTP, PINNED_REPOSITORY, PINNED_REVISION, PINNED_TARGET,
    PinnedArtifact, PinnedArtifactRole, io_verify_artifact_file, pinned_model_dir,
    resolve_pinned_model_dir, verify_artifact_file, verify_mtp_file, verify_target_file,
};
pub use portable_snapshot::{
    PortableArray, PortableModelState, PortablePage, PortablePagedTensor, PortablePromptSnapshot,
};
pub use provider::{
    BaselineGeneration, ChatContentPart, ChatContentRef, ChatCustomToolCall, ChatFile,
    ChatImageUrl, ChatInputAudio, ChatMessage, ChatMessageContent, ChatPromptCacheBreakpoint,
    ChatTool, ChatToolCall, ChatToolCallFunction, ChatToolFunction, GenerationOutput,
    GenerationRequest, MtpPrefixReuse, MtpPromptSnapshot, PreparedMultimodalPrefill,
    PromptSnapshot, Qwen35Provider,
};
#[cfg(any(feature = "dflash2", test))]
pub use provider::{Dflash2GenerationStats, Dflash2PrefixReuse, Dflash2PromptSnapshot};
pub use qwen_vl::{ExpandedImageTokens, insert_qwen_vl_image_tokens};
pub use qwen_vl_processor::{PreparedImage, QwenVLProcessor};
pub use sha256::sha256_file;
#[cfg(any(feature = "specprefill", test))]
pub use specprefill::{
    PrefillMode, SPECPREFILL_DRAFT_MODEL_IDENTIFIER, SpecPrefillConfig, SpecPrefillStats,
};
