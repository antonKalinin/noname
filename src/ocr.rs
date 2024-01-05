use std::error::Error;

use image::RgbaImage;
use ocrs::{OcrEngine, OcrEngineParams};
use rten::Model;
use rten_tensor::{NdTensor, NdTensorView};

// Converts an image into a CHW tensor.
fn image_to_tensor(image: &RgbaImage) -> Result<NdTensor<f32, 3>, Box<dyn Error>> {
    let (width, height) = image.dimensions();
    let layout = image.sample_layout();

    let chw_tensor = NdTensorView::from_slice(
        image.as_raw().as_slice(),
        [height as usize, width as usize, 3],
        Some([
            layout.height_stride,
            layout.width_stride,
            layout.channel_stride,
        ]),
    )?
    .permuted([2, 0, 1]) // HWC => CHW
    .to_tensor() // Make tensor contiguous, which makes `map` faster
    .map(|x| *x as f32 / 255.); // Rescale from [0, 255] to [0, 1]

    Ok(chw_tensor)
}

pub fn recognize(image: &RgbaImage) -> Result<(), Box<dyn Error>> {
    let detection_model_data = include_bytes!("../models/text-detection.rten");
    let recognition_model_data = include_bytes!("../models/text-recognition.rten");

    println!("✅ Models are read");

    let detection_model = Model::load(detection_model_data)?;
    let recognition_model = Model::load(recognition_model_data)?;

    println!("✅ Models are loaded");

    let engine = OcrEngine::new(OcrEngineParams {
        detection_model: Some(detection_model),
        recognition_model: Some(recognition_model),
        ..Default::default()
    })?;

    println!("✅ Engine is created");

    let image_tensor = image_to_tensor(image)?;

    println!("✅ Image converted to tensor");

    // Apply standard image pre-processing expected by this library (convert
    // to grayscale, map range to [-0.5, 0.5]).
    let ocr_input = engine.prepare_input(image_tensor.view())?;

    println!("✅ Image is prepared");

    // Phase 1: Detect text words
    let word_rects = engine.detect_words(&ocr_input)?;

    println!("✅ Words are detected");

    // Phase 2: Perform layout analysis
    let line_rects = engine.find_text_lines(&ocr_input, &word_rects);

    println!("✅ Line rects are found");

    // Phase 3: Recognize text
    let line_texts = engine.recognize_text(&ocr_input, &line_rects)?;

    println!("✅ Text is recognized");

    for line in line_texts
        .iter()
        .flatten()
        // Filter likely spurious detections. With future model improvements
        // this should become unnecessary.
        .filter(|l| l.to_string().len() > 1)
    {
        println!("{}", line);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    fn test_recognize() {
        let input_img = image::open("test.png").unwrap();
        let input_img = input_img.into_rgba8();

        recognize(&input_img).unwrap();
        assert_eq!(1, 1);
    }
}
