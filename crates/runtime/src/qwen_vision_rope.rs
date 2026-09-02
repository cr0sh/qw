use mlxcel_core::{MlxArray, UniquePtr};

pub(crate) fn concat_many(arrays: &[UniquePtr<MlxArray>], axis: i32) -> UniquePtr<MlxArray> {
    assert!(!arrays.is_empty());
    let mut result = mlxcel_core::copy(
        arrays[0]
            .as_ref()
            .expect("vision concatenate input must not be null"),
    );
    for array in &arrays[1..] {
        result = mlxcel_core::concatenate(
            result
                .as_ref()
                .expect("vision concatenate result must not be null"),
            array
                .as_ref()
                .expect("vision concatenate input must not be null"),
            axis,
        );
    }
    result
}

pub(crate) struct VisionRotaryEmbedding {
    dim: usize,
    theta: f32,
}

impl VisionRotaryEmbedding {
    #[cfg(any(feature = "specprefill", test))]
    pub(crate) fn new(dim: usize) -> Self {
        Self {
            dim,
            theta: 10_000.0,
        }
    }

    pub(crate) fn forward(&self, sequence_length: i32) -> UniquePtr<MlxArray> {
        let half_dim = self.dim / 2;
        let dim = self.dim as f32;
        let frequencies = (0..half_dim)
            .map(|index| 1.0 / self.theta.powf((2 * index) as f32 / dim))
            .collect::<Vec<_>>();
        let frequencies = mlxcel_core::from_slice_f32(&frequencies, &[half_dim as i32]);
        let sequence = mlxcel_core::arange_i32(0, sequence_length, 1);
        let sequence = mlxcel_core::astype(&sequence, mlxcel_core::dtype::FLOAT32);
        mlxcel_core::outer(&sequence, &frequencies)
    }
}

pub(crate) fn apply_rotary_pos_emb_vision(
    tensor: &MlxArray,
    frequencies: &MlxArray,
) -> UniquePtr<MlxArray> {
    let original_dtype = mlxcel_core::array_dtype(tensor);
    let tensor = mlxcel_core::astype(tensor, mlxcel_core::dtype::FLOAT32);
    let cosine = mlxcel_core::tile(
        &mlxcel_core::expand_dims(&mlxcel_core::cos(frequencies), 1),
        &[1, 1, 2],
    );
    let sine = mlxcel_core::tile(
        &mlxcel_core::expand_dims(&mlxcel_core::sin(frequencies), 1),
        &[1, 1, 2],
    );
    let rotated = rotate_half(&tensor);
    let output = mlxcel_core::add(
        &mlxcel_core::multiply(&tensor, &cosine),
        &mlxcel_core::multiply(&rotated, &sine),
    );
    mlxcel_core::astype(&output, original_dtype)
}

fn rotate_half(array: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(array);
    let half = shape[shape.len() - 1] / 2;
    let rank = shape.len();
    let starts = vec![0; rank];
    let mut stops = shape.clone();
    stops[rank - 1] = half;
    let first = mlxcel_core::slice(array, &starts, &stops);
    let mut starts = starts;
    starts[rank - 1] = half;
    stops[rank - 1] = shape[rank - 1];
    let second = mlxcel_core::slice(array, &starts, &stops);
    mlxcel_core::concatenate(&mlxcel_core::negative(&second), &first, rank as i32 - 1)
}
