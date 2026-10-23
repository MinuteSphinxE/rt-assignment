//! RTXON, second take: a GPU path tracer built on hardware ray tracing.
//!
//! * WGSL compute shader using inline ray queries (`enable wgpu_ray_query`)
//! * wgpu 30 on Vulkan, so the rays are traced by the RT cores through a
//!   bottom/top level acceleration structure
//! * egui front-end with a live progressive renderer
//!
//! `--out shot.png` runs the same renderer headless and writes a PNG, which is
//! also how the reference images for this project were produced.

mod app;
mod renderer;
mod scene;

use std::path::Path;
use std::sync::Arc;

use eframe::wgpu;

use renderer::{PathTracer, Settings};

fn device_descriptor(adapter: &wgpu::Adapter) -> wgpu::DeviceDescriptor<'static> {
    // Ray queries are still an experimental wgpu feature: the bit must be in
    // `required_features` *and* the experimental-features token must be opted
    // into. `EXPERIMENTAL_RAY_QUERY` is Vulkan-only, which is exactly what we
    // want (Windows + NVIDIA driver).
    let mut features = wgpu::Features::empty();
    if adapter
        .features()
        .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
    {
        features |= wgpu::Features::EXPERIMENTAL_RAY_QUERY;
    }

    wgpu::DeviceDescriptor {
        label: Some("rtxon-rt device"),
        required_features: features,
        required_limits: adapter.limits(),
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    let mut out: Option<String> = None;
    let mut width: u32 = 1280;
    let mut height: u32 = 720;
    let mut samples: u32 = 128;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--out" | "-o" => {
                out = Some(args.get(i + 1).cloned().ok_or("--out needs a path")?);
                i += 2;
            }
            "--width" => {
                width = args.get(i + 1).ok_or("--width needs a number")?.parse()?;
                i += 2;
            }
            "--height" => {
                height = args.get(i + 1).ok_or("--height needs a number")?.parse()?;
                i += 2;
            }
            "--samples" => {
                samples = args.get(i + 1).ok_or("--samples needs a number")?.parse()?;
                i += 2;
            }
            "--help" | "-h" => {
                println!(
                    "rtxon-rt ? WGSL ray-query path tracer\n\n\
                     USAGE:\n  rtxon-rt                     open the egui viewer\n  \
                     rtxon-rt --out shot.png      headless render to PNG\n\n\
                     OPTIONS:\n  --width <px>     headless width  (default 1280)\n  \
                     --height <px>    headless height (default 720)\n  \
                     --samples <n>    samples per pixel (default 128)"
                );
                return Ok(());
            }
            other => {
                eprintln!("unknown argument: {other} (try --help)");
                std::process::exit(2);
            }
        }
    }

    match out {
        Some(path) => run_headless(&path, width, height, samples),
        None => run_gui(),
    }
}

fn run_headless(path: &str, width: u32, height: u32, samples: u32) -> Result<(), Box<dyn std::error::Error>> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))?;

    let info = adapter.get_info();
    if !adapter.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY) {
        return Err(format!(
            "adapter '{}' ({:?}) does not expose experimental ray queries",
            info.name, info.backend
        )
        .into());
    }

    let (device, queue) = pollster::block_on(adapter.request_device(&device_descriptor(&adapter)))?;
    device.on_uncaptured_error(std::sync::Arc::new(|e: wgpu::Error| eprintln!("wgpu error: {e}")));


    let mut tracer = PathTracer::new(
        device,
        queue,
        format!("{} ({:?})", info.name, info.backend),
        width,
        height,
    );
    println!(
        "adapter: {}\ntriangles: {}\nBLAS+TLAS build: {:.1} ms",
        tracer.adapter_name, tracer.triangle_count, tracer.blas_build_ms
    );

    let settings = Settings::default();
    let t0 = std::time::Instant::now();
    for _ in 0..samples {
        tracer.render(&settings, 1);
    }
    // Finish the last submission before reading back.
    tracer.device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    })?;
    let elapsed = t0.elapsed().as_secs_f32();
    tracer.save_png(Path::new(path))?;

    println!(
        "rendered {}x{} @ {} spp in {:.2} s ({:.1} Msamples/s) -> {}",
        width,
        height,
        samples,
        elapsed,
        (width as f32 * height as f32 * samples as f32) / elapsed / 1e6,
        path
    );
    Ok(())
}

fn run_gui() -> Result<(), Box<dyn std::error::Error>> {
    use eframe::egui_wgpu;

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1440.0, 900.0])
            .with_title("RTXON ? WGSL ray-query path tracer"),
        wgpu_options: egui_wgpu::WgpuConfiguration {
            wgpu_setup: egui_wgpu::WgpuSetup::CreateNew(egui_wgpu::WgpuSetupCreateNew {
                device_descriptor: Arc::new(|adapter| device_descriptor(adapter)),
                ..egui_wgpu::WgpuSetupCreateNew::without_display_handle()
            }),
            ..Default::default()
        },
        ..Default::default()
    };

    eframe::run_native(
        "rtxon-rt",
        options,
        Box::new(|cc| Ok(Box::new(app::RtApp::new(cc)))),
    )?;
    Ok(())
}






