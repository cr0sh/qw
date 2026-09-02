use std::io::Cursor;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use image::{GenericImageView, ImageFormat, ImageReader, Limits};

use crate::protocol::{CompletionRequest, RequestError};

pub const MAX_SOURCE_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_IMAGE_DIMENSION: u32 = 16_384;
pub const MAX_DECODER_ALLOCATION: u64 = 512 * 1024 * 1024;
const MAX_ENCODED_BYTES: usize = MAX_SOURCE_BYTES.div_ceil(3) * 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodedImageFormat {
    Png,
    Jpeg,
    WebP,
}

impl DecodedImageFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpeg",
            Self::WebP => "webp",
        }
    }

    fn image_format(self) -> ImageFormat {
        match self {
            Self::Png => ImageFormat::Png,
            Self::Jpeg => ImageFormat::Jpeg,
            Self::WebP => ImageFormat::WebP,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedImage {
    pub format: DecodedImageFormat,
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

pub fn decode_request_images(request: &mut CompletionRequest) -> Result<(), RequestError> {
    let image_urls = request
        .messages
        .iter()
        .flat_map(|message| message.image_urls())
        .map(|image| image.url.as_str())
        .collect::<Vec<_>>();
    if image_urls.len() != request.image_params.len() {
        return Err(RequestError::new(
            "internal image input cardinality mismatch",
            None,
        ));
    }

    let mut decoded = Vec::with_capacity(image_urls.len());
    for (url, param) in image_urls.into_iter().zip(&request.image_params) {
        decoded.push(decode_data_image(url, param)?);
    }
    request.decoded_images = decoded;
    Ok(())
}

pub fn decode_data_image(url: &str, param: &str) -> Result<DecodedImage, RequestError> {
    let (format, payload) = if let Some(payload) = url.strip_prefix("data:image/png;base64,") {
        (DecodedImageFormat::Png, payload)
    } else if let Some(payload) = url.strip_prefix("data:image/jpeg;base64,") {
        (DecodedImageFormat::Jpeg, payload)
    } else if let Some(payload) = url.strip_prefix("data:image/webp;base64,") {
        (DecodedImageFormat::WebP, payload)
    } else {
        return Err(RequestError::at(
            "image_url must be a base64 data URI for PNG, JPEG, or WebP",
            param,
        ));
    };

    if payload.is_empty() {
        return Err(RequestError::at(
            "image data URI payload must not be empty",
            param,
        ));
    }
    if payload.len() > MAX_ENCODED_BYTES {
        return Err(RequestError::at(
            format!("decoded image source must not exceed {MAX_SOURCE_BYTES} bytes"),
            param,
        ));
    }

    let bytes = STANDARD
        .decode(payload.as_bytes())
        .map_err(|_| RequestError::at("image data URI contains malformed base64", param))?;
    if bytes.len() > MAX_SOURCE_BYTES {
        return Err(RequestError::at(
            format!("decoded image source must not exceed {MAX_SOURCE_BYTES} bytes"),
            param,
        ));
    }

    let detected = image::guess_format(&bytes)
        .map_err(|_| RequestError::at("image data does not contain a supported image", param))?;
    if detected != format.image_format() {
        return Err(RequestError::at(
            "image MIME type does not match the decoded image format",
            param,
        ));
    }

    let mut limits = Limits::default();
    limits.max_image_width = Some(MAX_IMAGE_DIMENSION);
    limits.max_image_height = Some(MAX_IMAGE_DIMENSION);
    limits.max_alloc = Some(MAX_DECODER_ALLOCATION);
    let mut reader = ImageReader::with_format(Cursor::new(&bytes), detected);
    reader.limits(limits);
    let image = reader
        .decode()
        .map_err(|error| RequestError::at(format!("failed to decode image: {error}"), param))?;
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 || width > MAX_IMAGE_DIMENSION || height > MAX_IMAGE_DIMENSION {
        return Err(RequestError::at(
            format!("image dimensions must be between 1 and {MAX_IMAGE_DIMENSION} pixels per axis"),
            param,
        ));
    }
    let rgb_allocation = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| RequestError::at("image allocation size overflow", param))?;
    if rgb_allocation > MAX_DECODER_ALLOCATION {
        return Err(RequestError::at(
            format!("decoded image allocation must not exceed {MAX_DECODER_ALLOCATION} bytes"),
            param,
        ));
    }
    let rgb = image.into_rgb8().into_raw();
    Ok(DecodedImage {
        format,
        width,
        height,
        rgb,
    })
}

#[cfg(test)]
mod tests {
    use image::{DynamicImage, ImageFormat, Rgb, RgbImage};

    use super::*;

    fn data_uri(format: ImageFormat, mime: &str, image: RgbImage) -> String {
        let mut encoded = Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(image)
            .write_to(&mut encoded, format)
            .expect("encode fixture");
        format!(
            "data:image/{mime};base64,{}",
            STANDARD.encode(encoded.into_inner())
        )
    }

    #[test]
    fn decodes_png_jpeg_and_webp_at_one_pixel() {
        for (format, mime, expected) in [
            (ImageFormat::Png, "png", DecodedImageFormat::Png),
            (ImageFormat::Jpeg, "jpeg", DecodedImageFormat::Jpeg),
            (ImageFormat::WebP, "webp", DecodedImageFormat::WebP),
        ] {
            let decoded = decode_data_image(
                &data_uri(format, mime, RgbImage::from_pixel(1, 1, Rgb([1, 2, 3]))),
                "image",
            )
            .expect("decode fixture");
            assert_eq!(decoded.format, expected);
            assert_eq!((decoded.width, decoded.height), (1, 1));
            assert_eq!(decoded.rgb.len(), 3);
        }
    }

    #[test]
    fn decodes_fifty_six_pixel_inputs_without_reordering_pixels() {
        let decoded = decode_data_image(
            &data_uri(
                ImageFormat::Png,
                "png",
                RgbImage::from_pixel(56, 56, Rgb([7, 8, 9])),
            ),
            "image",
        )
        .expect("decode fixture");
        assert_eq!((decoded.width, decoded.height), (56, 56));
        assert!(decoded.rgb.chunks_exact(3).all(|pixel| pixel == [7, 8, 9]));
    }

    #[test]
    fn rejects_malformed_base64_image_bytes_and_mime_mismatch() {
        let malformed = decode_data_image("data:image/png;base64,***", "p").unwrap_err();
        assert_eq!(malformed.param.as_deref(), Some("p"));
        let bytes = STANDARD.encode(b"not an image");
        assert!(decode_data_image(&format!("data:image/png;base64,{bytes}"), "p").is_err());
        let jpeg = data_uri(
            ImageFormat::Jpeg,
            "jpeg",
            RgbImage::from_pixel(1, 1, Rgb([0, 0, 0])),
        );
        let mismatch = jpeg.replacen("image/jpeg", "image/png", 1);
        assert!(
            decode_data_image(&mismatch, "p")
                .unwrap_err()
                .message
                .contains("MIME")
        );
    }

    #[test]
    fn enforces_encoded_source_and_dimension_boundaries() {
        assert_eq!(MAX_ENCODED_BYTES, MAX_SOURCE_BYTES.div_ceil(3) * 4);
        let oversized = format!(
            "data:image/png;base64,{}",
            "A".repeat(MAX_ENCODED_BYTES + 1)
        );
        assert!(
            decode_data_image(&oversized, "p")
                .unwrap_err()
                .message
                .contains("must not exceed")
        );

        let at_boundary = data_uri(
            ImageFormat::Png,
            "png",
            RgbImage::from_pixel(MAX_IMAGE_DIMENSION, 1, Rgb([0, 0, 0])),
        );
        assert_eq!(
            decode_data_image(&at_boundary, "p")
                .expect("dimension boundary")
                .width,
            MAX_IMAGE_DIMENSION
        );
        let beyond = data_uri(
            ImageFormat::Png,
            "png",
            RgbImage::from_pixel(MAX_IMAGE_DIMENSION + 1, 1, Rgb([0, 0, 0])),
        );
        assert!(decode_data_image(&beyond, "p").is_err());
    }

    #[test]
    fn decoder_allocation_limit_rejects_decompression_bombs() {
        fn crc32(bytes: &[u8]) -> u32 {
            let mut crc = u32::MAX;
            for byte in bytes {
                crc ^= u32::from(*byte);
                for _ in 0..8 {
                    crc = (crc >> 1) ^ (0xedb8_8320 & (0u32.wrapping_sub(crc & 1)));
                }
            }
            !crc
        }

        let uri = data_uri(
            ImageFormat::Png,
            "png",
            RgbImage::from_pixel(1, 1, Rgb([0, 0, 0])),
        );
        let payload = uri.split_once(',').expect("data URI separator").1;
        let mut png = STANDARD.decode(payload).expect("fixture base64");
        png[16..20].copy_from_slice(&14_000u32.to_be_bytes());
        png[20..24].copy_from_slice(&14_000u32.to_be_bytes());
        let ihdr_crc = crc32(&png[12..29]);
        png[29..33].copy_from_slice(&ihdr_crc.to_be_bytes());
        let bomb = format!("data:image/png;base64,{}", STANDARD.encode(png));
        assert!(decode_data_image(&bomb, "p").is_err());
    }
}
