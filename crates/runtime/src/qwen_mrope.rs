use mlxcel_core::{MlxArray, UniquePtr};

pub(crate) struct InterleavedMRoPE {
    inverse_frequency: Vec<f32>,
    sections: Vec<i32>,
}

impl InterleavedMRoPE {
    pub(crate) fn new(dim: usize, base: f32, sections: Vec<i32>) -> Self {
        let inverse_frequency = (0..dim)
            .step_by(2)
            .map(|index| 1.0 / base.powf(index as f32 / dim as f32))
            .collect();
        Self {
            inverse_frequency,
            sections,
        }
    }

    pub(crate) fn forward(
        &self,
        position_ids: &MlxArray,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let shape = mlxcel_core::array_shape(position_ids);
        let positions = if shape.len() == 2 {
            mlxcel_core::broadcast_to(
                &mlxcel_core::expand_dims(position_ids, 0),
                &[3, shape[0], shape[1]],
            )
        } else {
            mlxcel_core::copy(position_ids)
        };
        let shape = mlxcel_core::array_shape(&positions);
        let batch = shape[1];
        let sequence = shape[2];
        let half_dim = self.inverse_frequency.len() as i32;
        let inverse = mlxcel_core::from_slice_f32(&self.inverse_frequency, &[half_dim]);
        let inverse = mlxcel_core::reshape(&inverse, &[1, 1, half_dim, 1]);
        let inverse = mlxcel_core::broadcast_to(&inverse, &[3, batch, half_dim, 1]);
        let positions = mlxcel_core::astype(
            &mlxcel_core::reshape(&positions, &[3, batch, 1, sequence]),
            mlxcel_core::dtype::FLOAT32,
        );
        let frequencies =
            mlxcel_core::transpose_axes(&mlxcel_core::matmul(&inverse, &positions), &[0, 1, 3, 2]);
        let frequencies = self.interleave(&frequencies);
        let embedding = mlxcel_core::concatenate(&frequencies, &frequencies, -1);
        (mlxcel_core::cos(&embedding), mlxcel_core::sin(&embedding))
    }

    fn interleave(&self, frequencies: &MlxArray) -> UniquePtr<MlxArray> {
        let half_dim = mlxcel_core::array_shape(frequencies)[3];
        let mut dimensions = vec![0; half_dim as usize];
        for (index, &section_length) in self.sections.iter().skip(1).enumerate() {
            let source = index as i32 + 1;
            let mut column = source;
            while column < section_length * 3 {
                if let Some(dimension) = dimensions.get_mut(column as usize) {
                    *dimension = source;
                }
                column += 3;
            }
        }
        let indices = mlxcel_core::from_slice_i32(&dimensions, &[1, 1, 1, half_dim]);
        mlxcel_core::squeeze_axis(&mlxcel_core::take_along_axis(frequencies, &indices, 0), 0)
    }
}

pub(crate) fn apply_multimodal_rotary_pos_emb(
    queries: &MlxArray,
    keys: &MlxArray,
    cosine: &MlxArray,
    sine: &MlxArray,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let cosine = mlxcel_core::expand_dims(cosine, 1);
    let sine = mlxcel_core::expand_dims(sine, 1);
    let rotate = |array: &MlxArray| {
        mlxcel_core::add(
            &mlxcel_core::multiply(array, &cosine),
            &mlxcel_core::multiply(&rotate_half(array), &sine),
        )
    };
    (rotate(queries), rotate(keys))
}

fn rotate_half(array: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(array);
    let rank = shape.len();
    let half = shape[rank - 1] / 2;
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
