use anyhow::{Result, ensure};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpandedImageTokens {
    pub image_blocks: usize,
    pub total_image_tokens: usize,
}

pub fn insert_qwen_vl_image_tokens(
    prompt_tokens: &mut Vec<i32>,
    grid_thw: &[(i32, i32, i32)],
    spatial_merge_size: usize,
    vision_start_token_id: i32,
    image_token_id: i32,
) -> Result<ExpandedImageTokens> {
    ensure!(!prompt_tokens.is_empty(), "image prompt token sequence is empty");
    ensure!(!grid_thw.is_empty(), "image grid list is empty");
    ensure!(spatial_merge_size > 0, "spatial merge size must be nonzero");
    let merge = i32::try_from(spatial_merge_size)
        .map_err(|_| anyhow::anyhow!("spatial merge size exceeds i32"))?;
    let mut counts = Vec::with_capacity(grid_thw.len());
    for (index, &(temporal, height, width)) in grid_thw.iter().enumerate() {
        ensure!(
            temporal > 0 && height > 0 && width > 0,
            "image grid {index} must contain positive dimensions"
        );
        ensure!(
            height % merge == 0 && width % merge == 0,
            "image grid {index} is not divisible by spatial merge size"
        );
        counts.push(
            usize::try_from(temporal * (height / merge) * (width / merge))
                .map_err(|_| anyhow::anyhow!("image grid {index} token count overflow"))?,
        );
    }

    let placeholders = prompt_tokens
        .iter()
        .enumerate()
        .filter_map(|(index, &token)| (token == image_token_id).then_some(index))
        .collect::<Vec<_>>();
    ensure!(
        placeholders.len() == grid_thw.len(),
        "rendered image placeholder count {} does not match image count {}",
        placeholders.len(),
        grid_thw.len()
    );
    let vision_end_token_id = vision_start_token_id
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("vision end token identifier overflow"))?;
    for (image_index, &placeholder) in placeholders.iter().enumerate() {
        ensure!(
            placeholder > 0
                && placeholder + 1 < prompt_tokens.len()
                && prompt_tokens[placeholder - 1] == vision_start_token_id
                && prompt_tokens[placeholder + 1] == vision_end_token_id,
            "rendered image placeholder {image_index} is not framed by vision markers"
        );
    }

    let total_image_tokens: usize = counts.iter().sum();
    let mut expanded = Vec::with_capacity(
        prompt_tokens.len() + total_image_tokens.saturating_sub(grid_thw.len()),
    );
    let mut image_index = 0;
    for &token in prompt_tokens.iter() {
        if token == image_token_id {
            expanded.extend(std::iter::repeat_n(image_token_id, counts[image_index]));
            image_index += 1;
        } else {
            expanded.push(token);
        }
    }
    *prompt_tokens = expanded;
    Ok(ExpandedImageTokens {
        image_blocks: grid_thw.len(),
        total_image_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_one_placeholder_to_the_merged_grid_count() {
        let mut tokens = vec![1, 100, 103, 101, 42];
        let stats = insert_qwen_vl_image_tokens(&mut tokens, &[(1, 4, 4)], 2, 100, 103)
            .expect("expand image tokens");
        assert_eq!(stats.total_image_tokens, 4);
        assert_eq!(tokens, vec![1, 100, 103, 103, 103, 103, 101, 42]);
    }

    #[test]
    fn expands_multiple_images_in_declared_order() {
        let mut tokens = vec![100, 103, 101, 9, 100, 103, 101];
        let stats = insert_qwen_vl_image_tokens(
            &mut tokens,
            &[(1, 4, 4), (1, 2, 4)],
            2,
            100,
            103,
        )
        .expect("expand image tokens");
        assert_eq!(stats.total_image_tokens, 6);
        assert_eq!(tokens.iter().filter(|&&token| token == 103).count(), 6);
        assert_eq!(&tokens[..6], &[100, 103, 103, 103, 103, 101]);
        assert_eq!(&tokens[7..], &[100, 103, 103, 101]);
    }

    #[test]
    fn rejects_missing_extra_and_already_expanded_placeholders() {
        let grids = [(1, 4, 4)];
        for mut tokens in [
            vec![1, 2, 3],
            vec![100, 103, 101, 100, 103, 101],
            vec![100, 103, 103, 101],
        ] {
            assert!(insert_qwen_vl_image_tokens(&mut tokens, &grids, 2, 100, 103).is_err());
        }
    }

    #[test]
    fn rejects_unframed_and_nondivisible_image_grids() {
        let mut unframed = vec![1, 103, 2];
        assert!(insert_qwen_vl_image_tokens(&mut unframed, &[(1, 4, 4)], 2, 100, 103)
            .is_err());
        let mut nondivisible = vec![100, 103, 101];
        assert!(insert_qwen_vl_image_tokens(
            &mut nondivisible,
            &[(1, 3, 4)],
            2,
            100,
            103,
        )
        .is_err());
    }
}
