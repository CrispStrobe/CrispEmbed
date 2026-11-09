// Compile the actual public FFI declarations and safe photo module against the
// production geometry objects. No mock FFI, inference engine, model or network.
extern crate self as crispembed_sys;
#[path = "../crispembed-sys/src/lib.rs"]
mod ffi;
pub use ffi::*;
#[path = "../crispembed/src/photo.rs"]
mod photo;

use photo::CrispScanPhoto;

#[test]
fn full_frame_and_zero_rotation_retain_rgb() {
    let pixels: Vec<u8> = (0..7 * 5 * 3).map(|i| (i % 251) as u8).collect();
    let corners = [[0., 0.], [6., 0.], [6., 4.], [0., 4.]];
    assert_eq!(CrispScanPhoto::rotate(&pixels, 7, 5, 3, 0.).unwrap(), (pixels.clone(), 7, 5));
    assert_eq!(CrispScanPhoto::perspective(&pixels, 7, 5, 3, &corners).unwrap(), (pixels, 7, 5));
}

#[test]
fn invalid_buffers_angles_and_corners_are_errors() {
    let pixels = vec![255; 7 * 5 * 3];
    assert!(CrispScanPhoto::rotate(&pixels[..5], 7, 5, 3, 2.).is_err());
    assert!(CrispScanPhoto::rotate(&pixels, i32::MAX, 5, 3, 2.).is_err());
    assert!(CrispScanPhoto::rotate(&pixels, 7, 5, 3, f32::NAN).is_err());
    assert!(CrispScanPhoto::rotate(&pixels, 7, 5, 3, 21.).is_err());
    assert!(CrispScanPhoto::perspective(&pixels, 7, 5, 3, &[[0., 0.], [6., 4.], [6., 0.], [0., 4.]]).is_err());
    assert!(CrispScanPhoto::correction_angle(&pixels, 7, 5, 3, 0.).is_err());
}

#[test]
fn blank_angle_is_inconclusive_and_rotation_expands() {
    let pixels = vec![255; 100 * 140 * 3];
    assert_eq!(CrispScanPhoto::correction_angle(&pixels, 100, 140, 3, 15.).unwrap(), 0.);
    let (output, width, height) = CrispScanPhoto::rotate(&pixels, 100, 140, 3, -8.).unwrap();
    assert!(width > 100 && height > 140);
    assert_eq!(output.len(), width as usize * height as usize * 3);
    assert!(output.iter().all(|byte| *byte == 255));
}
