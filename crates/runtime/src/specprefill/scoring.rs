// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 oMLX contributors
//
// Adapted from https://github.com/jundot/omlx/tree/0436bdf2792edf74e8f077beea20782bd0f4a298/omlx/specprefill
// and omlx/patches/specprefill.py.

use anyhow::{Result, ensure};
use mlxcel_core::generate::SamplingConfig;
use mlxcel_core::generation_policy::seed_rng_if_needed;
use mlxcel_core::sampling::sample_token_optimized;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::qwen3_5::Qwen35Model;

/// Number of decoded draft lookahead query rows.
const LOOKAHEAD_TOKENS: usize = 8;
/// Average-pool kernel width over scored prompt positions.
const POOL_WIDTH: i32 = 13;

/// The fixed seed reseeds MLX's global RNG, including subsequent target sampling.
fn draft_sampling_config() -> SamplingConfig {
    SamplingConfig {
        temperature: 0.6,
        top_p: 0.95,
        seed: Some(0),
        ..SamplingConfig::default()
    }
}

pub(crate) fn score_tokens(draft: &Qwen35Model, prompt_ids: &[i32]) -> Result<Vec<f32>> {
    let mut logits = draft.specprefill_draft_prefill(prompt_ids)?;
    let sampling = draft_sampling_config();
    seed_rng_if_needed(&sampling);
    let mut captures: Vec<Vec<UniquePtr<MlxArray>>> = Vec::new();
    let mut history = prompt_ids.to_vec();
    let mut token = sample_token_optimized(&logits, &sampling, &history).0;
    mlxcel_core::eval(&token);

    for _ in 0..LOOKAHEAD_TOKENS {
        let token_id = mlxcel_core::item_i32(&token);
        history.push(token_id);
        let (next_logits, queries) = draft.specprefill_draft_lookahead(token_id);
        if captures.is_empty() {
            ensure!(
                !queries.is_empty(),
                "SpecPrefill draft captured no full-attention queries"
            );
            captures = queries
                .into_iter()
                .map(|query| {
                    let mut layer = Vec::with_capacity(LOOKAHEAD_TOKENS);
                    layer.push(query);
                    layer
                })
                .collect();
        } else {
            ensure!(
                queries.len() == captures.len(),
                "SpecPrefill draft attention capture topology changed"
            );
            for (layer, query) in captures.iter_mut().zip(queries) {
                layer.push(query);
            }
        }
        logits = next_logits;
        token = sample_token_optimized(&logits, &sampling, &history).0;
        mlxcel_core::eval(&token);
    }

    let keys = draft.specprefill_draft_prompt_keys(prompt_ids.len());
    compute_importance(&captures, &keys, prompt_ids.len())
}

fn compute_importance(
    captures: &[Vec<UniquePtr<MlxArray>>],
    prompt_keys: &[UniquePtr<MlxArray>],
    prompt_len: usize,
) -> Result<Vec<f32>> {
    let mut all_scores: Option<UniquePtr<MlxArray>> = None;
    for (queries, keys) in captures.iter().zip(prompt_keys) {
        ensure!(!queries.is_empty(), "SpecPrefill captured no draft queries");
        let stacked = stack_queries(queries);
        let weights = layer_attention_weights(&stacked, keys, prompt_len)?;
        all_scores = Some(match all_scores {
            Some(existing) => mlxcel_core::concatenate(&existing, &weights, 0),
            None => weights,
        });
    }
    let combined = all_scores.expect("pinned draft contains full-attention layers");
    importance_from_weights(&combined, prompt_len)
}

fn stack_queries(queries: &[UniquePtr<MlxArray>]) -> UniquePtr<MlxArray> {
    let query_refs = queries
        .iter()
        .map(|query| query.as_ref().expect("captured query is non-null"))
        .collect::<Vec<_>>();
    let mut stacked = mlxcel_core::copy(query_refs[0]);
    for query in &query_refs[1..] {
        stacked = mlxcel_core::concatenate(&stacked, query, 2);
    }
    stacked
}

fn layer_attention_weights(
    stacked: &MlxArray,
    keys: &MlxArray,
    prompt_len: usize,
) -> Result<UniquePtr<MlxArray>> {
    let key_shape = mlxcel_core::array_shape(keys);
    let query_shape = mlxcel_core::array_shape(stacked);
    ensure!(
        key_shape[2] >= prompt_len as i32,
        "SpecPrefill draft cache does not span the scored prompt"
    );
    ensure!(
        query_shape[1] % key_shape[1] == 0,
        "SpecPrefill draft query/KV head ratio is not integral"
    );
    let query_groups = query_shape[1] / key_shape[1];
    let grouped_queries = mlxcel_core::reshape(
        stacked,
        &[
            query_shape[0],
            key_shape[1],
            query_groups,
            query_shape[2],
            query_shape[3],
        ],
    );
    let transposed_keys = mlxcel_core::transpose_axes(keys, &[0, 1, 3, 2]);
    let grouped_keys = mlxcel_core::expand_dims(&transposed_keys, 2);
    let scores = mlxcel_core::matmul(&grouped_queries, &grouped_keys);
    let scale = mlxcel_core::full_f32(
        &[1],
        1.0 / (key_shape[3] as f32).sqrt(),
        mlxcel_core::dtype::FLOAT32,
    );
    let scores = mlxcel_core::multiply(
        &mlxcel_core::astype(&scores, mlxcel_core::dtype::FLOAT32),
        &scale,
    );
    let weights = mlxcel_core::softmax_precise(&scores, -1);
    let weights = mlxcel_core::reshape(&weights, &[-1, LOOKAHEAD_TOKENS as i32, prompt_len as i32]);
    Ok(weights)
}

fn importance_from_weights(combined: &MlxArray, prompt_len: usize) -> Result<Vec<f32>> {
    let pooled = average_pool_1d(combined, POOL_WIDTH);
    let max_over_layers_and_heads = mlxcel_core::max_axis(&pooled, 0, false);
    let importance = mlxcel_core::mean_axis(&max_over_layers_and_heads, 0, false);
    let importance = mlxcel_core::astype(&importance, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&importance);
    let bytes = mlxcel_core::array_to_raw_bytes(&importance);
    ensure!(
        bytes.len() == prompt_len * std::mem::size_of::<f32>(),
        "SpecPrefill importance vector has an unexpected byte length"
    );
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_ne_bytes(*bytes))
        .collect())
}

fn average_pool_1d(values: &MlxArray, width: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(values);
    let pad = width / 2;
    let padded = mlxcel_core::pad(values, &[0, 0, 0, 0, pad, pad], 0.0);
    let prefix = mlxcel_core::cumsum(&padded, -1, false, true);
    let zero = mlxcel_core::zeros(&[shape[0], shape[1], 1], mlxcel_core::array_dtype(values));
    let prefix = mlxcel_core::concatenate(&zero, &prefix, -1);
    let prefix_shape = mlxcel_core::array_shape(&prefix);
    let right = mlxcel_core::slice(
        &prefix,
        &[0, 0, width],
        &[prefix_shape[0], prefix_shape[1], prefix_shape[2]],
    );
    let left = mlxcel_core::slice(
        &prefix,
        &[0, 0, 0],
        &[prefix_shape[0], prefix_shape[1], prefix_shape[2] - width],
    );
    let divisor = mlxcel_core::full_f32(&[1], width as f32, mlxcel_core::array_dtype(values));
    mlxcel_core::divide(&mlxcel_core::subtract(&right, &left), &divisor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::specprefill::{SpecPrefillConfig, select_target_indices};

    // Golden bits recorded from the pre-refactor implementation with FP16 captures/keys.
    #[test]
    fn synthetic_fp16_importance_golden() {
        let prompt_len = 40;
        let captures = (0..2)
            .map(|layer| {
                (0..LOOKAHEAD_TOKENS)
                    .map(|row| {
                        let values = (0..16)
                            .map(|i| ((i * 7 + row * 3 + layer * 11) % 23) as f32 / 19.0 - 0.5)
                            .collect::<Vec<_>>();
                        mlxcel_core::astype(
                            &mlxcel_core::from_slice_f32(&values, &[1, 4, 1, 4]),
                            mlxcel_core::dtype::FLOAT16,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let keys = (0..2)
            .map(|layer| {
                let values = (0..2 * prompt_len * 4)
                    .map(|i| ((i * 13 + layer * 5) % 31) as f32 / 29.0 - 0.5)
                    .collect::<Vec<_>>();
                mlxcel_core::astype(
                    &mlxcel_core::from_slice_f32(&values, &[1, 2, prompt_len as i32, 4]),
                    mlxcel_core::dtype::FLOAT16,
                )
            })
            .collect::<Vec<_>>();
        let actual = compute_importance(&captures, &keys, prompt_len).unwrap();
        assert_eq!(
            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            [
                1013394133, 1015175198, 1016177234, 1017277246, 1018311351, 1019339299, 1020449912,
                1020400107, 1020448902, 1020386788, 1020401863, 1020531504, 1020475035, 1020597359,
                1020599688, 1020544883, 1020551626, 1020549574, 1020500759, 1020564103, 1020648978,
                1020495688, 1020650282, 1020559252, 1020511530, 1020514517, 1020520430, 1020489426,
                1020486949, 1020554430, 1020442156, 1020439143, 1020335424, 1020363470, 1019341506,
                1018357880, 1017301551, 1016177232, 1015192470, 1013219259
            ]
        );
        let config = SpecPrefillConfig {
            keep_rate: 0.5,
            keep_first_tokens: 2,
            keep_last_tokens: 2,
            ..Default::default()
        };
        assert_eq!(
            select_target_indices(&actual, 7, 47, config),
            (7..39).chain(45..47).collect::<Vec<_>>()
        );
    }

    #[test]
    fn synthetic_importance_and_selection_golden() {
        let prompt_len = 40;
        let captures = (0..2)
            .map(|layer| {
                (0..LOOKAHEAD_TOKENS)
                    .map(|row| {
                        let values = (0..16)
                            .map(|i| ((i * 7 + row * 3 + layer * 11) % 23) as f32 / 16.0 - 0.5)
                            .collect::<Vec<_>>();
                        mlxcel_core::from_slice_f32(&values, &[1, 4, 1, 4])
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let keys = (0..2)
            .map(|layer| {
                let values = (0..2 * prompt_len * 4)
                    .map(|i| ((i * 13 + layer * 5) % 31) as f32 / 32.0 - 0.5)
                    .collect::<Vec<_>>();
                mlxcel_core::from_slice_f32(&values, &[1, 2, prompt_len as i32, 4])
            })
            .collect::<Vec<_>>();
        let importance = compute_importance(&captures, &keys, prompt_len).unwrap();
        assert_eq!(
            importance.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            [
                1013425948, 1015178647, 1016193843, 1017291542, 1018312022, 1019368750, 1020476772,
                1020440848, 1020499998, 1020402491, 1020432094, 1020584228, 1020491640, 1020634622,
                1020669885, 1020589334, 1020585118, 1020603384, 1020525738, 1020601220, 1020697765,
                1020550344, 1020704616, 1020607070, 1020559768, 1020540383, 1020564202, 1020525362,
                1020503617, 1020601091, 1020498499, 1020467297, 1020362922, 1020403245, 1019362182,
                1018390988, 1017331880, 1016193843, 1015236765, 1013284446
            ]
        );
        let config = SpecPrefillConfig {
            keep_rate: 0.5,
            keep_first_tokens: 2,
            keep_last_tokens: 2,
            ..Default::default()
        };
        assert_eq!(
            select_target_indices(&importance, 7, 47, config),
            (7..39).chain(45..47).collect::<Vec<_>>()
        );

        let values = (0..40).map(|i| (i % 7) as f32 / 8.0).collect::<Vec<_>>();
        let input = mlxcel_core::from_slice_f32(&values, &[1, 1, 40]);
        let pooled = average_pool_1d(&input, POOL_WIDTH);
        mlxcel_core::eval(&pooled);
        let bytes = mlxcel_core::array_to_raw_bytes(&pooled);
        let pooled = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_ne_bytes(*b))
            .collect::<Vec<_>>();
        assert_eq!(
            pooled.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            [
                1045349612, 1045349612, 1045994890, 1047285445, 1048898639, 1050189194, 1051802388,
                1053738220, 1053415582, 1053092943, 1052770304, 1052447665, 1052125026, 1051802388,
                1053738220, 1053415582, 1053092943, 1052770304, 1052447665, 1052125026, 1051802388,
                1053738220, 1053415582, 1053092943, 1052770304, 1052447665, 1052125026, 1051802388,
                1053738220, 1053415582, 1053092943, 1052770304, 1052447665, 1052125026, 1050189194,
                1050189194, 1049866555, 1049221278, 1047930722, 1045349612
            ]
        );
        for (i, actual) in pooled.iter().enumerate() {
            let expected = values[i.saturating_sub(6)..(i + 7).min(values.len())]
                .iter()
                .sum::<f32>()
                / 13.0;
            assert!((actual - expected).abs() < 1e-7);
        }
    }
}
