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
#[cfg(any(feature = "specprefill", test))]
mod specprefill;

mod portable_snapshot;
pub mod provider;

pub use mlxcel_core::cache::KVCacheMode;
#[cfg(any(feature = "dflash2", test))]
pub use model_resolver::{DEFAULT_DFLASH2_DRAFT_MODEL_IDENTIFIER, resolve_dflash2_draft_path};
pub use model_resolver::{
    DEFAULT_MODEL_IDENTIFIER, model_cache_path, resolve_model_dir, resolve_model_path,
    validate_identifier,
};
#[cfg(any(feature = "specprefill", test))]
pub use model_resolver::{
    DEFAULT_SPECPREFILL_DRAFT_MODEL_IDENTIFIER, resolve_specprefill_draft_path,
};
pub use portable_snapshot::{
    PortableArray, PortableModelState, PortablePage, PortablePagedTensor, PortablePromptSnapshot,
};
pub use provider::{
    BaselineGeneration, ChatContentPart, ChatContentRef, ChatCustomToolCall, ChatFile,
    ChatImageUrl, ChatInputAudio, ChatMessage, ChatMessageContent, ChatPromptCacheBreakpoint,
    ChatTool, ChatToolCall, ChatToolCallFunction, ChatToolFunction, GenerationOutput,
    GenerationRequest, MtpGenerationStats, MtpPrefixReuse, MtpPromptSnapshot,
    PreparedMultimodalPrefill, PromptSnapshot, Qwen35GenerationMode, Qwen35Provider,
    SamplingOptions, select_qwen35_decoder,
};
#[cfg(any(feature = "dflash2", test))]
pub use provider::{Dflash2GenerationStats, Dflash2PrefixReuse, Dflash2PromptSnapshot};
pub use qwen_vl::{ExpandedImageTokens, insert_qwen_vl_image_tokens};
pub use qwen_vl_processor::{PreparedImage, QwenVLProcessor};
#[cfg(any(feature = "specprefill", test))]
pub use specprefill::{
    PrefillMode, SPECPREFILL_DRAFT_MODEL_IDENTIFIER, SpecPrefillConfig, SpecPrefillStats,
};
