use std::error::Error;

use arboard::Clipboard;
use image::ImageBuffer;
use noname::ocr;

fn main() -> Result<(), Box<dyn Error>> {
    let mut clipboard = Clipboard::new().unwrap();
    let image_data = match clipboard.get_image() {
        Ok(data) => data,
        Err(_) => {
            println!("No image in clipboard");
            return Ok(());
        }
    };

    println!("Image width: {:?}", image_data.width);
    println!("Image height: {:?}", image_data.height);

    let image = ImageBuffer::from_raw(
        image_data.width as u32,
        image_data.height as u32,
        image_data.bytes.into_owned(),
    )
    .unwrap();

    ocr::recognize(&image)?;

    Ok(())
}
