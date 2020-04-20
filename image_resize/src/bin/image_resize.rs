extern crate image;
extern crate imageproc;
extern crate getopts;
extern crate rand;
use image::GenericImageView;
use image::GenericImage;
use std::env;
use std::path::PathBuf;
use std::ffi;

fn modify(img: &mut image::DynamicImage,  opt_fill: Option<u32>) -> Result<(), image::ImageError>{
    let (width, height) = img.dimensions();
    for x in 0..width {
        for y in 0..height {
            let pixel = img.get_pixel(x,y);
            let image::Rgba(data) = pixel;
            let rgb = [data[0], data[1], data[2]];
            if rgb == [0xe7, 0xb2, 0x34] || rgb == [0x3c, 0xff, 0x00] {
                img.put_pixel(x,y, image::Rgba([0x00,0x00,0x08,0xff]));
            }
        }
    }
    if let Some(fill) = opt_fill {
        let mut blur = img.filter3x3(&[
                1./16., 0.125, 0.125,
            0.125,0.125,0.125,
            0.125,0.125,1./16.,
        ]);
        for _ in 1..fill {
            blur = blur.filter3x3(&[
                1./16., 0.125, 0.125,
                0.125,0.125,0.125,
                0.125,0.125,1./16.,
            ]);
        }
        let mut bkgimg = img.clone();
        let grid_res = 16;
        let num_polygons_wide = (width + grid_res - 1) / grid_res;
        let mut grid = vec![imageproc::drawing::Point::new(0,0); num_polygons_wide as usize * ((height + grid_res - 1)/grid_res) as usize];
        for (index, item) in grid.iter_mut().enumerate() {
            let x_index = index as u32 % num_polygons_wide;
            let y_index = index as u32 / num_polygons_wide;
            let mut x = x_index * grid_res;
            let mut y = y_index * grid_res;
            if x_index != 0 && y_index != 0 && x_index + 1 != num_polygons_wide && y_index + 1 != num_polygons_wide {
                let jitterx = rand::random::<u32>() % grid_res;
                let jittery = rand::random::<u32>() % grid_res;
                x -= jitterx;
                y -= jittery;
            }
            *item = imageproc::drawing::Point::new(x as i32,y as i32);
        }
        for x in 0..(num_polygons_wide as usize - 1) {
            for y in 0..(grid.len()/num_polygons_wide as usize -1) {
                let points = [grid[x as usize + y as usize *num_polygons_wide as usize],
                          grid[x + 1 + y*num_polygons_wide as usize],
                          grid[x + (y + 1)*num_polygons_wide as usize],
                          grid[(x + 1) + (y + 1)*num_polygons_wide as usize],
                ];
                let color_a = rand::random::<u32>();
                let color_b = rand::random::<u32>();
                imageproc::drawing::draw_convex_polygon_mut(
                    &mut bkgimg,
                    &points[..3],
                    image::Rgba([color_a as u8,(color_a>>8) as u8, (color_a>>16) as u8, 0xff]),
                );
                imageproc::drawing::draw_convex_polygon_mut(
                    &mut bkgimg,
                    &points[1..4],
                    image::Rgba([color_b as u8,(color_b>>8) as u8, (color_b>>16) as u8, 0xff]),
                );
            }
        }
        for x in 0..width {
            for y in 0..height {
                if blur.get_pixel(x,y) == image::Rgba([0x00,0x00,0x08,0xff]) {
                    img.put_pixel(x,y, bkgimg.get_pixel(x,y))
                }
            }
        }
    }
    Ok(())
}
fn shrink_nearest(img: &image::DynamicImage) -> Result<image::RgbaImage, image::ImageError>{
    let (width, height) = img.dimensions();
    let small_img = image::ImageBuffer::from_fn(width>>1, height>>1, |x, y| {
        img.get_pixel(x<<1, y<<1)
    });
    Ok(small_img)
}
fn process(input_path: &str, opt_crop: Option<Rect>, opt_fill: Option<u32>) -> Result<(), image::ImageError>{
    let path = PathBuf::from(input_path.clone());
    let mut img = image::open(input_path)?;
    if let Some(crop) = opt_crop {
        img = img.crop(crop.x, crop.y, crop.w, crop.h);
    }
    let ext = if let Some(iext) = path.extension().clone() {
        Some(iext.clone())
    } else {
        None
    };
    let path_nox = path.with_extension("");
    if let Some(filename) = path_nox.file_name() {
        let output_filename = filename.to_string_lossy().into_owned() + "_hi";
        let output_path_nox = path_nox.with_file_name(output_filename);
        let output_path = if let Some(uext) = ext {
            output_path_nox.with_extension(uext)
        } else {
            output_path_nox
        };
        modify(&mut img, opt_fill)?;
        img.save(output_path)?;
        let output2_filename = filename.to_string_lossy().into_owned() + "_lo";
        let output2_path_nox = path_nox.with_file_name(output2_filename);
        let output2_path = if let Some(uext) = ext {
            output2_path_nox.with_extension(uext)
        } else {
            output2_path_nox
        };
        shrink_nearest(&img)?.save(output2_path)?
    }
    Ok(())
}

#[derive(Clone)]
struct Rect {
    x:u32,
    y:u32,
    w:u32,
    h:u32,
}
impl Rect {
    pub fn try_new(x:Option<u32>, y:Option<u32>, w_opt:Option<u32>, h_opt:Option<u32>) -> Option<Rect> {
        if let Some(w) = w_opt {
            if let Some(h) = h_opt {
                if w != 0 && h != 0 {
                    return Some(Rect {
                        x:x.unwrap_or(0),
                        y:y.unwrap_or(0),
                        w:w,
                        h:h,
                    })
                }
            }
        }
        None
    }
}
fn main() {
    let args : Vec<ffi::OsString> = env::args_os().collect();
    let mut opts = getopts::Options::new();
    opts.optopt("x","cropx","set crop x", "0");
    opts.optopt("y","cropy","set crop y", "0");
    opts.optopt("w", "cropw", "set  crop width/wideness", "0");
    opts.optopt("t", "croph", "set  crop height/tallness", "0");
    opts.optopt("f", "fill", "set blackened areas to colors: specify num pixels", "");
    opts.optflag("h", "help", "print help");
    let matches = match opts.parse(&args[1..]) {
        Ok(m) => {m}
        Err(f)  => { panic!(f.to_string())}
    };
    if matches.opt_present("h") {
        print!("{}", opts.usage(&""));
        return;
    }
    let crop = Rect::try_new(
        matches.opt_get::<u32>("cropx").unwrap(),
        matches.opt_get::<u32>("cropy").unwrap(),
        matches.opt_get::<u32>("cropw").unwrap(),
        matches.opt_get::<u32>("croph").unwrap(),
    );
    let fill = matches.opt_get::<u32>("fill").unwrap();
    for argument in matches.free {
        process(&argument, crop.clone(), fill).unwrap();
    }
}
