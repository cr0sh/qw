#[derive(Debug, PartialEq, Eq)]
pub(crate) struct GenerationElements {
    pub(crate) prefill: u64,
    pub(crate) decode: u64,
}

pub(crate) fn generation_elements(
    batch_size: usize,
    prompt_tokens: usize,
    completion_tokens: usize,
) -> GenerationElements {
    GenerationElements {
        prefill: (batch_size * prompt_tokens) as u64,
        // Prefill produces the logits for the first completion token. Only the
        // remaining completion tokens require autoregressive decode steps.
        decode: (batch_size * completion_tokens.saturating_sub(1)) as u64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_one_generation_iteration_to_phase_token_counts() {
        assert_eq!(
            generation_elements(1, 241, 32),
            GenerationElements {
                prefill: 241,
                decode: 31,
            }
        );
    }

    #[test]
    fn includes_every_sequence_in_a_batch() {
        assert_eq!(
            generation_elements(4, 16, 8),
            GenerationElements {
                prefill: 64,
                decode: 28,
            }
        );
    }
}
