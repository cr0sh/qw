use anyhow::{Result, ensure};
use image::RgbImage;
use image::imageops::{FilterType, resize};
use mlxcel_core::{MlxArray, UniquePtr};

#[derive(Debug, Clone, PartialEq)]
pub struct PreparedImage {
    pub patches: Vec<f32>,
    pub grid_thw: (i32, i32, i32),
    pub row_width: usize,
}

impl PreparedImage {
    pub(crate) fn to_mlx(&self) -> UniquePtr<MlxArray> {
        let rows = self.patches.len() / self.row_width;
        mlxcel_core::from_slice_f32(&self.patches, &[rows as i32, self.row_width as i32])
    }
}

#[derive(Debug, Clone)]
pub struct QwenVLProcessor {
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
}

impl QwenVLProcessor {
    pub fn new(
        patch_size: usize,
        temporal_patch_size: usize,
        spatial_merge_size: usize,
        min_pixels: usize,
        max_pixels: usize,
    ) -> Result<Self> {
        ensure!(patch_size > 0, "vision patch_size must be nonzero");
        ensure!(
            temporal_patch_size > 0,
            "vision temporal_patch_size must be nonzero"
        );
        ensure!(
            spatial_merge_size > 0,
            "vision spatial_merge_size must be nonzero"
        );
        ensure!(min_pixels > 0, "vision min_pixels must be nonzero");
        ensure!(
            max_pixels >= min_pixels,
            "vision max_pixels must be at least min_pixels"
        );
        Ok(Self {
            patch_size,
            temporal_patch_size,
            spatial_merge_size,
            min_pixels,
            max_pixels,
        })
    }

    pub fn smart_resize(&self, orig_h: u32, orig_w: u32) -> (u32, u32) {
        let factor = (self.patch_size * self.spatial_merge_size) as u32;
        let mut height = ((orig_h as f64 / factor as f64).round() as u32).max(1) * factor;
        let mut width = ((orig_w as f64 / factor as f64).round() as u32).max(1) * factor;

        let pixels = height as usize * width as usize;
        if pixels > self.max_pixels {
            let scale = (self.max_pixels as f64 / pixels as f64).sqrt();
            height = ((height as f64 * scale / factor as f64).round() as u32).max(1) * factor;
            width = ((width as f64 * scale / factor as f64).round() as u32).max(1) * factor;
        }
        if height as usize * width as usize > self.max_pixels {
            let scale = (self.max_pixels as f64 / (height as usize * width as usize) as f64).sqrt();
            height = ((height as f64 * scale / factor as f64).floor() as u32).max(1) * factor;
            width = ((width as f64 * scale / factor as f64).floor() as u32).max(1) * factor;
        }
        let pixels = height as usize * width as usize;
        if pixels < self.min_pixels {
            let scale = (self.min_pixels as f64 / pixels as f64).sqrt();
            height = ((height as f64 * scale / factor as f64).ceil() as u32).max(1) * factor;
            width = ((width as f64 * scale / factor as f64).ceil() as u32).max(1) * factor;
        }
        (height, width)
    }

    pub fn prepare_rgb(&self, image: &RgbImage) -> PreparedImage {
        let (target_h, target_w) = self.smart_resize(image.height(), image.width());
        let resized = resize(image, target_w, target_h, FilterType::Lanczos3);
        self.patchify(&resized)
    }

    pub fn prepare_rgb_bytes(
        &self,
        width: u32,
        height: u32,
        rgb: Vec<u8>,
    ) -> Result<PreparedImage> {
        let image = RgbImage::from_raw(width, height, rgb).ok_or_else(|| {
            anyhow::anyhow!("decoded RGB image dimensions do not match its buffer")
        })?;
        Ok(self.prepare_rgb(&image))
    }

    fn patchify(&self, image: &RgbImage) -> PreparedImage {
        let height = image.height() as usize;
        let width = image.width() as usize;
        let h_patches = height / self.patch_size;
        let w_patches = width / self.patch_size;
        let row_width = 3 * self.patch_size * self.patch_size;
        let mut normalized = vec![0.0f32; 3 * height * width];
        for y in 0..height {
            for x in 0..width {
                let pixel = image.get_pixel(x as u32, y as u32);
                for channel in 0..3 {
                    normalized[channel * height * width + y * width + x] =
                        (pixel[channel] as f32 / 255.0 - 0.5) / 0.5;
                }
            }
        }

        let mut patches =
            Vec::with_capacity(h_patches * w_patches * self.temporal_patch_size * row_width);
        for block_y in 0..h_patches / self.spatial_merge_size {
            for block_x in 0..w_patches / self.spatial_merge_size {
                for inner_y in 0..self.spatial_merge_size {
                    for inner_x in 0..self.spatial_merge_size {
                        let patch_y = block_y * self.spatial_merge_size + inner_y;
                        let patch_x = block_x * self.spatial_merge_size + inner_x;
                        let y_start = patch_y * self.patch_size;
                        let x_start = patch_x * self.patch_size;
                        for _ in 0..self.temporal_patch_size {
                            for channel in 0..3 {
                                for dy in 0..self.patch_size {
                                    for dx in 0..self.patch_size {
                                        let y = y_start + dy;
                                        let x = x_start + dx;
                                        patches.push(
                                            normalized[channel * height * width + y * width + x],
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        PreparedImage {
            patches,
            grid_thw: (1, h_patches as i32, w_patches as i32),
            row_width,
        }
    }
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};

    use super::*;

    fn processor() -> QwenVLProcessor {
        QwenVLProcessor::new(14, 2, 2, 4 * 28 * 28, 16_384 * 28 * 28).expect("processor")
    }

    #[test]
    fn smart_resize_honors_exact_factor_and_pixel_boundaries() {
        let processor = QwenVLProcessor::new(14, 2, 2, 28 * 28, 56 * 56).expect("processor");
        assert_eq!(processor.smart_resize(1, 1), (28, 28));
        assert_eq!(processor.smart_resize(56, 56), (56, 56));
        let (height, width) = processor.smart_resize(500, 500);
        assert_eq!((height, width), (56, 56));
    }

    #[test]
    fn merger_grouped_patch_order_and_grid_are_deterministic() {
        let processor = processor();
        let mut image = RgbImage::new(56, 56);
        for patch_y in 0..4u32 {
            for patch_x in 0..4u32 {
                let value = (patch_y * 4 + patch_x) as u8 * 10;
                for y in patch_y * 14..(patch_y + 1) * 14 {
                    for x in patch_x * 14..(patch_x + 1) * 14 {
                        image.put_pixel(x, y, Rgb([value, 0, 0]));
                    }
                }
            }
        }
        let prepared = processor.prepare_rgb(&image);
        assert_eq!(prepared.grid_thw, (1, 4, 4));
        assert_eq!(prepared.row_width, 3 * 14 * 14);
        assert_eq!(prepared.patches.len(), 32 * prepared.row_width);
        let grouped_patch_ids = [0u8, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15];
        for (row, patch_id) in grouped_patch_ids.into_iter().enumerate() {
            let actual = prepared.patches[row * 2 * prepared.row_width];
            let expected = patch_id as f32 * 10.0 / 255.0 * 2.0 - 1.0;
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn multiple_images_keep_declared_order() {
        let processor = processor();
        let black = RgbImage::from_pixel(56, 56, Rgb([0, 0, 0]));
        let white = RgbImage::from_pixel(56, 56, Rgb([255, 255, 255]));
        let prepared = [processor.prepare_rgb(&black), processor.prepare_rgb(&white)];
        assert_eq!(prepared[0].patches[0], -1.0);
        assert_eq!(prepared[1].patches[0], 1.0);
    }

    #[test]
    fn owned_values_convert_to_the_same_mlx_shape_and_order() {
        let prepared = processor().prepare_rgb(&RgbImage::new(56, 56));
        let mlx = prepared.to_mlx();
        assert_eq!(
            mlxcel_core::array_shape(&mlx),
            vec![32, prepared.row_width as i32]
        );
        mlxcel_core::eval(&mlx);
        let roundtrip = mlxcel_core::array_to_raw_bytes(&mlx)
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().expect("f32 byte width")))
            .collect::<Vec<_>>();
        assert_eq!(roundtrip, prepared.patches);
    }
}
