use mlxcel_core::{MlxArray, UniquePtr};

pub(crate) fn decode_rope_positions(
    cache_offset: i32,
    sequence_length: i32,
    rope_delta: i32,
) -> UniquePtr<MlxArray> {
    let positions = mlxcel_core::arange_i32(
        cache_offset + rope_delta,
        cache_offset + rope_delta + sequence_length,
        1,
    );
    mlxcel_core::broadcast_to(
        &mlxcel_core::reshape(&positions, &[1, 1, sequence_length]),
        &[3, 1, sequence_length],
    )
}

#[cfg(test)]
mod tests {
    use super::decode_rope_positions;

    #[test]
    fn decode_continues_from_cache_offset_plus_delta() {
        let positions = decode_rope_positions(10, 2, -3);
        mlxcel_core::eval(&positions);
        assert_eq!(mlxcel_core::array_shape(&positions), vec![3, 1, 2]);
        assert_eq!(
            mlxcel_core::item_i32(&mlxcel_core::slice(&positions, &[0, 0, 0], &[1, 1, 1])),
            7
        );
    }
}
