//! Minimal egui front-end: a viewer for the ray-query path tracer.

use std::time::Instant;

use eframe::egui;
use eframe::wgpu;

use crate::renderer::{PathTracer, Settings};

pub struct RtApp {
    tracer: PathTracer,
    settings: Settings,
    tex_id: egui::TextureId,
    paused: bool,
    fps: f32,
    last_frame: Instant,
    status: String,
    tex_size: (u32, u32),
}

impl RtApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let rs = cc
            .wgpu_render_state
            .as_ref()
            .expect("this app needs the wgpu backend");
        let info = rs.adapter.get_info();
        let tracer = PathTracer::new(
            rs.device.clone(),
            rs.queue.clone(),
            format!("{} ({:?})", info.name, info.backend),
            1024,
            640,
        );
        let tex_id = rs.renderer.write().register_native_texture(
            &rs.device,
            &tracer.display_view,
            wgpu::FilterMode::Linear,
        );
        let mut app = Self {
            tracer,
            settings: Settings::default(),
            tex_id,
            paused: false,
            fps: 0.0,
            last_frame: Instant::now(),
            status: String::new(),
            tex_size: (1024, 640),
        };
        app.status = "space: pause / drag: orbit / wheel: zoom".to_owned();
        app
    }

    fn reset(&mut self) {
        self.tracer.reset();
    }
}

impl eframe::App for RtApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();

        // ---- keyboard / timing ---------------------------------------
        if ctx.input(|i| i.key_pressed(egui::Key::Space)) {
            self.paused = !self.paused;
        }
        let dt = self.last_frame.elapsed().as_secs_f32();
        self.last_frame = Instant::now();
        if dt > 0.0 {
            self.fps = self.fps * 0.9 + (1.0 / dt) * 0.1;
        }

        // ---- side panel ------------------------------------------------
        egui::Panel::left("controls")
            .resizable(false)
            .default_size(310.0)
            .show(ui, |ui| {
                ui.heading("RTXON - Ray Query");
                ui.label("WGSL path tracer, RT cores, egui");
                ui.separator();

                ui.label(format!("GPU: {}", self.tracer.adapter_name));
                ui.label(format!(
                    "{} tris ? BLAS {:.1} ms",
                    self.tracer.triangle_count, self.tracer.blas_build_ms
                ));
                ui.label(format!("{} spp ? {:.0} fps", self.tracer.sample_index, self.fps));
                ui.separator();

                let mut changed = false;
                changed |= ui
                    .add(egui::Slider::new(&mut self.settings.samples_per_frame, 1..=8).text("samples / frame"))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut self.settings.max_bounces, 1..=16).text("max bounces"))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut self.settings.exposure, 0.1..=4.0).text("exposure"))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut self.settings.roughness_scale, 0.0..=2.0).text("roughness"))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut self.settings.light_intensity, 0.0..=3.0).text("light"))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut self.settings.camera.distance, 3.0..=30.0).text("distance"))
                    .changed();
                let mut sky = self.settings.sky.to_array();
                if ui.color_edit_button_rgb(&mut sky).changed() {
                    self.settings.sky = glam::Vec3::from_array(sky);
                    changed = true;
                }
                if changed {
                    self.reset();
                }

                ui.separator();
                ui.horizontal(|ui| {
                    if ui
                        .button(if self.paused { "resume" } else { "pause" })
                        .clicked()
                    {
                        self.paused = !self.paused;
                    }
                    if ui.button("reset").clicked() {
                        self.reset();
                    }
                    if ui.button("save png").clicked() {
                        let path = std::path::Path::new("rtxon_rt.png");
                        match self.tracer.save_png(path) {
                            Ok(()) => self.status = format!("saved {}", path.display()),
                            Err(e) => self.status = format!("save failed: {e}"),
                        }
                    }
                });
                ui.add_space(6.0);
                ui.label(egui::RichText::new(&self.status).small().weak());
            });

        // ---- viewport ----------------------------------------------------
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ui, |ui| {
                let avail = ui.available_size();
                let ppp = ctx.pixels_per_point();
                let w = (avail.x * ppp).max(64.0) as u32;
                let h = (avail.y * ppp).max(64.0) as u32;
                if (w, h) != self.tex_size {
                    self.tracer.resize(w, h);
                    if let Some(rs) = frame.wgpu_render_state() {
                        rs.renderer.write().update_egui_texture_from_wgpu_texture(
                            &rs.device,
                            &self.tracer.display_view,
                            wgpu::FilterMode::Linear,
                            self.tex_id,
                        );
                    }
                    self.tex_size = (w, h);
                }

                if !self.paused {
                    let n = self.settings.samples_per_frame;
                    self.tracer.render(&self.settings, n);
                }

                let image = egui::Image::new(egui::load::SizedTexture::new(self.tex_id, avail))
                    .sense(egui::Sense::drag());
                let response = ui.add(image);
                if response.dragged() {
                    let d = response.drag_delta();
                    self.settings.camera.orbit(d.x, d.y);
                    self.reset();
                }
                if response.hovered() {
                    let scroll = ctx.input(|i| i.smooth_scroll_delta.y);
                    if scroll.abs() > 0.0 {
                        self.settings.camera.distance =
                            (self.settings.camera.distance * (1.0 - scroll * 0.004)).clamp(3.0, 30.0);
                        self.reset();
                    }
                }
            });

        ctx.request_repaint();
    }
}


