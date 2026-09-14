use anyhow::{Result, ensure};
use mlxcel_core::{MlxArray, UniquePtr};

pub(crate) fn merge_llava(
    image_token_id: i32,
    image_features: &MlxArray,
    input_embeddings: &MlxArray,
    input_ids: &MlxArray,
) -> Result<UniquePtr<MlxArray>> {
    let feature_shape = mlxcel_core::array_shape(image_features);
    ensure!(
        feature_shape.len() >= 2,
        "vision features must have at least two dimensions"
    );
    let embedding_shape = mlxcel_core::array_shape(input_embeddings);
    ensure!(
        embedding_shape.len() == 3,
        "text embeddings must have shape [batch, sequence, hidden]"
    );
    let hidden = *feature_shape.last().expect("feature rank checked");
    ensure!(
        embedding_shape[2] == hidden,
        "vision feature width does not match text embedding width"
    );
    let feature_rows = feature_shape[..feature_shape.len() - 1]
        .iter()
        .product::<i32>();
    let flat_features = mlxcel_core::reshape(image_features, &[feature_rows, hidden]);
    let flat_features =
        mlxcel_core::astype(&flat_features, mlxcel_core::array_dtype(input_embeddings));

    let image_token = mlxcel_core::full_f32(&[1], image_token_id as f32, mlxcel_core::dtype::INT32);
    let image_token = mlxcel_core::astype(&image_token, mlxcel_core::dtype::INT32);
    let is_image = mlxcel_core::equal(input_ids, &image_token);
    let image_count =
        mlxcel_core::sum_all(&mlxcel_core::astype(&is_image, mlxcel_core::dtype::INT32));
    mlxcel_core::eval(&image_count);
    ensure!(
        mlxcel_core::item_i32(&image_count) == feature_rows,
        "merged vision feature count does not match expanded image token count"
    );

    let image_mask = mlxcel_core::expand_dims(&is_image, -1);
    let image_mask = mlxcel_core::repeat(&image_mask, hidden, -1);
    Ok(masked_scatter(
        input_embeddings,
        &image_mask,
        &flat_features,
    ))
}

fn masked_scatter(
    input_embeddings: &MlxArray,
    expanded_mask: &MlxArray,
    image_features: &MlxArray,
) -> UniquePtr<MlxArray> {
    let output_shape = mlxcel_core::array_shape(input_embeddings);
    let features = mlxcel_core::flatten(image_features);
    let embeddings = mlxcel_core::flatten(input_embeddings);
    let mask = mlxcel_core::flatten(expanded_mask);
    let mask_i32 = mlxcel_core::astype(&mask, mlxcel_core::dtype::INT32);
    let cumulative = mlxcel_core::cumsum(&mask_i32, 0, false, true);
    let one = mlxcel_core::full_f32(&[1], 1.0, mlxcel_core::dtype::INT32);
    let one = mlxcel_core::astype(&one, mlxcel_core::dtype::INT32);
    let indices = mlxcel_core::subtract(&cumulative, &one);
    let zero = mlxcel_core::full_f32(&[1], 0.0, mlxcel_core::dtype::INT32);
    let zero = mlxcel_core::astype(&zero, mlxcel_core::dtype::INT32);
    let indices = mlxcel_core::maximum(&indices, &zero);
    let gathered = mlxcel_core::take(&features, &indices, 0);
    let output = mlxcel_core::where_cond(&mask, &gathered, &embeddings);
    mlxcel_core::reshape(&output, &output_shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(array: &MlxArray) -> Vec<f32> {
        mlxcel_core::eval(array);
        mlxcel_core::array_to_raw_bytes(array)
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().expect("f32 byte width")))
            .collect()
    }

    #[test]
    fn scatters_multiple_image_features_in_token_order() {
        let ids = mlxcel_core::from_slice_i32(&[5, 9, 6, 9], &[1, 4]);
        let text = mlxcel_core::from_slice_f32(&[0.0; 8], &[1, 4, 2]);
        let vision = mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let merged = merge_llava(9, &vision, &text, &ids).expect("merge features");
        assert_eq!(
            values(&merged),
            vec![0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0]
        );
    }

    #[test]
    fn rejects_feature_and_image_token_cardinality_mismatch() {
        let ids = mlxcel_core::from_slice_i32(&[9], &[1, 1]);
        let text = mlxcel_core::from_slice_f32(&[0.0, 0.0], &[1, 1, 2]);
        let vision = mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert!(merge_llava(9, &vision, &text, &ids).is_err());
    }
}
