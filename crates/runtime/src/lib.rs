mod chat_template;
mod gated_delta;
mod model_owned;
mod model_resolver;
mod qwen3_5;
#[cfg(any(feature = "dflash2", test))]
mod qwen3_5_dflash;
mod qwen3_5_mtp;
mod qwen3_next;
mod qwen3_vl_vision;
mod qwen_mrope;
mod qwen_mrope_state;
mod qwen_vision_rope;
mod qwen_vl;
mod qwen_vl_merge;
mod qwen_vl_position;
mod qwen_vl_processor;
mod specprefill;

pub mod provider;
mod portable_snapshot;

pub use mlxcel_core::cache::KVCacheMode;
pub use provider::{
    BaselineGeneration, ChatContentPart, ChatContentRef, ChatCustomToolCall, ChatFile,
    ChatImageUrl, ChatInputAudio, ChatMessage, ChatMessageContent, ChatPromptCacheBreakpoint,
    ChatTool, ChatToolCall, ChatToolCallFunction, ChatToolFunction,
    GenerationOutput, GenerationRequest, MtpPrefixReuse, MtpPromptSnapshot,
    PreparedMultimodalPrefill, PromptSnapshot, Qwen35Provider,
};
#[cfg(any(feature = "dflash2", test))]
pub use provider::Dflash2GenerationStats;
pub use model_resolver::{
    DEFAULT_MODEL_IDENTIFIER, DEFAULT_SPECPREFILL_DRAFT_MODEL_IDENTIFIER, model_cache_path,
    resolve_model_dir, resolve_model_path, resolve_specprefill_draft_path, validate_identifier,
};
pub use specprefill::{
    PrefillMode, SPECPREFILL_DRAFT_MODEL_IDENTIFIER, SpecPrefillConfig, SpecPrefillStats,
};
pub use portable_snapshot::{PortableArray, PortableModelState, PortablePromptSnapshot};
pub use qwen_vl::{ExpandedImageTokens, insert_qwen_vl_image_tokens};
pub use qwen_vl_processor::{PreparedImage, QwenVLProcessor};
