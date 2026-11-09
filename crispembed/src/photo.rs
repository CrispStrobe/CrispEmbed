//! Explicit phone-photo geometry, with checked buffers and no process-global gates.
//!
//! These additive APIs require the native library built from the same source.
//! A zero angle is inconclusive (already aligned or no reliable estimate), not
//! a confidence score. Keep the original page for preview and undo.

use std::ptr;

const MAX_PIXELS: usize = 64 * 1024 * 1024;

/// Model-free photo preparation. Calls are independent and do not mutate inputs.
pub struct CrispScanPhoto;

fn image_len(width: i32, height: i32, channels: i32) -> Result<usize, String> {
    if !(3..=32768).contains(&width)
        || !(3..=32768).contains(&height)
        || ![1, 3, 4].contains(&channels)
    {
        return Err("Photo dimensions or channel count are invalid".into());
    }
    let count = (width as usize)
        .checked_mul(height as usize)
        .filter(|count| *count <= MAX_PIXELS)
        .ok_or("Photo exceeds the 64 megapixel limit")?;
    count
        .checked_mul(channels as usize)
        .ok_or_else(|| "Photo byte count overflows".into())
}

fn validate(pixels: &[u8], width: i32, height: i32, channels: i32) -> Result<(), String> {
    if pixels.len() != image_len(width, height, channels)? {
        return Err("Photo byte count does not match its dimensions".into());
    }
    Ok(())
}

struct NativeImage(*mut u8);

impl Drop for NativeImage {
    fn drop(&mut self) {
        unsafe { crispembed_sys::crispembed_scan_photo_free_image(self.0) };
    }
}

fn take_image(
    status: i32,
    pixels: *mut u8,
    width: i32,
    height: i32,
    channels: i32,
) -> Result<(Vec<u8>, i32, i32), String> {
    let owned = NativeImage(pixels);
    if status != 0 || owned.0.is_null() {
        return Err("Photo transform rejected its geometry or could not allocate an image".into());
    }
    // An output can be smaller than the accepted source (e.g. a 2px crop).
    if width < 1 || height < 1 || width > 32768 || height > 32768 {
        return Err("Photo transform returned invalid dimensions".into());
    }
    let count = (width as usize)
        .checked_mul(height as usize)
        .filter(|count| *count <= MAX_PIXELS)
        .and_then(|count| count.checked_mul(channels as usize))
        .ok_or("Photo transform returned an oversized image")?;
    let copy = unsafe { std::slice::from_raw_parts(owned.0, count) }.to_vec();
    Ok((copy, width, height))
}

impl CrispScanPhoto {
    /// Recommended clockwise correction in degrees, measured on a bounded image.
    /// `max_angle_degrees` must be finite in `(0,20]`. Zero means aligned or unknown.
    pub fn correction_angle(
        pixels: &[u8],
        width: i32,
        height: i32,
        channels: i32,
        max_angle_degrees: f32,
    ) -> Result<f32, String> {
        validate(pixels, width, height, channels)?;
        if !max_angle_degrees.is_finite() || max_angle_degrees <= 0.0 || max_angle_degrees > 20.0 {
            return Err("Photo angle limit must be between 0 and 20 degrees".into());
        }
        let mut angle = 0.0;
        let status = unsafe {
            crispembed_sys::crispembed_scan_photo_angle(
                pixels.as_ptr(), pixels.len(), width, height, channels, max_angle_degrees, &mut angle,
            )
        };
        if status != 0 || !angle.is_finite() || angle.abs() > max_angle_degrees {
            return Err("Photo angle analysis failed".into());
        }
        Ok(angle)
    }

    /// Rotate clockwise by a finite angle in `[-20,20]`; keeps all channels.
    /// The canvas expands, with white fill. Zero preserves the exact pixel bytes.
    pub fn rotate(
        pixels: &[u8],
        width: i32,
        height: i32,
        channels: i32,
        clockwise_degrees: f32,
    ) -> Result<(Vec<u8>, i32, i32), String> {
        validate(pixels, width, height, channels)?;
        if !clockwise_degrees.is_finite() || clockwise_degrees.abs() > 20.0 {
            return Err("Photo rotation must be between -20 and 20 degrees".into());
        }
        let (mut output, mut ow, mut oh) = (ptr::null_mut(), 0, 0);
        let status = unsafe {
            crispembed_sys::crispembed_scan_photo_rotate(
                pixels.as_ptr(), pixels.len(), width, height, channels, clockwise_degrees,
                &mut output, &mut ow, &mut oh,
            )
        };
        take_image(status, output, ow, oh, channels)
    }

    /// Rectify four ordered `[x,y]` pixel centres: TL, TR, BR, BL. Corners must
    /// form a convex polygon inside the image. The labels preserve orientation.
    pub fn perspective(
        pixels: &[u8],
        width: i32,
        height: i32,
        channels: i32,
        corners: &[[f32; 2]; 4],
    ) -> Result<(Vec<u8>, i32, i32), String> {
        validate(pixels, width, height, channels)?;
        for [x, y] in corners {
            if !x.is_finite() || !y.is_finite() || *x < 0.0 || *y < 0.0
                || *x > (width - 1) as f32 || *y > (height - 1) as f32
            {
                return Err("Photo corners must be finite and inside the source image".into());
            }
        }
        let (mut output, mut ow, mut oh) = (ptr::null_mut(), 0, 0);
        let status = unsafe {
            crispembed_sys::crispembed_scan_photo_perspective(
                pixels.as_ptr(), pixels.len(), width, height, channels, corners.as_ptr().cast(),
                &mut output, &mut ow, &mut oh,
            )
        };
        take_image(status, output, ow, oh, channels)
    }
}
