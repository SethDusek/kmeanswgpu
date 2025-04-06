use std::{env::Args, sync::Arc};

use image::{ImageBuffer, ImageReader, Rgba};
use pollster::FutureExt;
use renderer::Renderer;
use winit::{
    application::ApplicationHandler,
    dpi::Size,
    event_loop::EventLoop,
    window::{Window, WindowAttributes},
};

mod renderer;

fn get_image(args: Args) -> anyhow::Result<Image> {
    let path = args.skip(1).next().unwrap();
    Ok(ImageReader::open(path)?.decode()?.to_rgba8())
}

pub type Image = ImageBuffer<Rgba<u8>, Vec<u8>>;

struct App<'a> {
    image: Image,
    window: Option<Arc<Window>>,
    renderer: Option<Renderer<'a>>,
}

impl<'a> ApplicationHandler for App<'a> {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = event_loop
            .create_window(WindowAttributes::default().with_inner_size(Size::Physical(
                winit::dpi::PhysicalSize {
                    width: self.image.width(),
                    height: self.image.height(),
                },
            )))
            .unwrap();
        let window = Arc::new(window);
        let renderer = Renderer::new(window.clone(), self.image.clone())
            .block_on()
            .unwrap();
        self.renderer = Some(renderer);
        self.window = Some(window.into());
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            winit::event::WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            winit::event::WindowEvent::RedrawRequested => {
                self.renderer.as_mut().map(|r| r.render());
            }
            _ => {}
        }
    }
}
fn main() -> anyhow::Result<()> {
    let image = get_image(std::env::args())?;
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Wait);
    let mut app = App {
        image,
        window: None,
        renderer: None,
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}
