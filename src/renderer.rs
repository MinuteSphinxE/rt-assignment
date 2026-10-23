//! wgpu renderer: builds the acceleration structures, runs the ray-query path
//! tracer into an accumulation buffer and tonemaps it into a texture that egui
//! (or the headless screenshot mode) can display.

use std::sync::Arc;

use glam::Vec3;
use wgpu::util::DeviceExt;

use crate::scene::{Camera, Scene};

pub const DISPLAY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// Mirrors the `Uniforms` struct in shader.wgsl (11 * vec4 = 176 bytes).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    pub cam_pos: [f32; 4],
    pub cam_fwd: [f32; 4],
    pub cam_right: [f32; 4],
    pub cam_up: [f32; 4],
    pub resolution: [f32; 4],
    pub params: [f32; 4],
    pub sky: [f32; 4],
    pub light_origin: [f32; 4],
    pub light_u: [f32; 4],
    pub light_v: [f32; 4],
    pub light_emission: [f32; 4],
}

#[derive(Clone, Debug)]
pub struct Settings {
    pub camera: Camera,
    pub max_bounces: u32,
    pub exposure: f32,
    pub roughness_scale: f32,
    pub sky: Vec3,
    pub light_intensity: f32,
    pub samples_per_frame: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            camera: Camera::default(),
            max_bounces: 8,
            exposure: 1.0,
            roughness_scale: 1.0,
            sky: Vec3::new(0.22, 0.15, 0.27),
            light_intensity: 1.0,
            samples_per_frame: 2,
        }
    }
}

pub struct PathTracer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,

    pub width: u32,
    pub height: u32,
    pub sample_index: u32,

    pub adapter_name: String,
    pub triangle_count: usize,
    pub blas_build_ms: f32,

    scene: Scene,
    uniforms: wgpu::Buffer,
    accum: wgpu::Buffer,
    positions_buf: wgpu::Buffer,
    normals_buf: wgpu::Buffer,
    indices_buf: wgpu::Buffer,
    tri_mat_buf: wgpu::Buffer,
    materials_buf: wgpu::Buffer,

    compute_bgl: wgpu::BindGroupLayout,
    compute_pipeline: wgpu::ComputePipeline,
    compute_bind_group: wgpu::BindGroup,

    display_pipeline: wgpu::RenderPipeline,
    display_bgl: wgpu::BindGroupLayout,
    display_bind_group: wgpu::BindGroup,

    display_texture: wgpu::Texture,
    pub display_view: wgpu::TextureView,

    // Kept alive so the TLAS stays valid.
    _blas: wgpu::Blas,
    tlas: wgpu::Tlas,
}

impl PathTracer {
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        adapter_name: String,
        width: u32,
        height: u32,
    ) -> Self {
        let scene = Scene::build();

        // ---- vertex / index buffers -------------------------------------
        let positions_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("positions"),
            contents: bytemuck::cast_slice(&scene.positions),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::BLAS_INPUT | wgpu::BufferUsages::COPY_DST,
        });
        let normals_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("normals"),
            contents: bytemuck::cast_slice(&scene.normals),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let indices_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("indices"),
            contents: bytemuck::cast_slice(&scene.indices),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDEX | wgpu::BufferUsages::BLAS_INPUT,
        });
        let tri_mat_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tri_material"),
            contents: bytemuck::cast_slice(&scene.tri_material),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let materials_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("materials"),
            contents: bytemuck::cast_slice(&scene.materials),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // ---- acceleration structures -------------------------------------
        let t0 = std::time::Instant::now();
        let blas = device.create_blas(
            &wgpu::CreateBlasDescriptor {
                label: Some("scene blas"),
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            },
            wgpu::BlasGeometrySizeDescriptors::Triangles {
                descriptors: vec![wgpu::BlasTriangleGeometrySizeDescriptor {
                    vertex_format: wgpu::VertexFormat::Float32x3,
                    vertex_count: scene.positions.len() as u32,
                    index_format: Some(wgpu::IndexFormat::Uint32),
                    index_count: Some(scene.indices.len() as u32),
                    flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
                }],
            },
        );

        let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
            label: Some("scene tlas"),
            max_instances: 1,
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        });

        // Identity instance transform (the geometry is already in world space).
        *tlas.get_mut_single(0).unwrap() = Some(wgpu::TlasInstance::new(
            &blas,
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            0,
            0xFF,
        ));

        let size_desc = wgpu::BlasTriangleGeometrySizeDescriptor {
            vertex_format: wgpu::VertexFormat::Float32x3,
            vertex_count: scene.positions.len() as u32,
            index_format: Some(wgpu::IndexFormat::Uint32),
            index_count: Some(scene.indices.len() as u32),
            flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
        };
        let build_entry = wgpu::BlasBuildEntry {
            blas: &blas,
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                size: &size_desc,
                vertex_buffer: &positions_buf,
                first_vertex: 0,
                vertex_stride: 16, // vec4<f32> stride, xyz used
                index_buffer: Some(&indices_buf),
                first_index: Some(0),
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        };

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("as build"),
        });
        encoder.build_acceleration_structures([&build_entry], [&tlas]);
        queue.submit([encoder.finish()]);
        device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        }).ok();
        let blas_build_ms = t0.elapsed().as_secs_f32() * 1000.0;

        // ---- uniform + accumulation buffers ------------------------------
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let accum = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("accumulation"),
            size: (width as u64) * (height as u64) * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---- shader -------------------------------------------------------
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("path tracer"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        let compute_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compute bgl"),
            entries: &[
                uniform_entry(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::AccelerationStructure { vertex_return: false },
                    count: None,
                },
                storage_entry(2),
                storage_entry(3),
                storage_entry(4),
                storage_entry(5),
                storage_entry(6),
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compute bg"),
            layout: &compute_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: tlas.as_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: positions_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: normals_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: indices_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: tri_mat_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: materials_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: accum.as_entire_binding() },
            ],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("path trace pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("compute layout"),
                bind_group_layouts: &[Some(&compute_bgl)],
                immediate_size: 0,
            })),
            module: &shader,
            entry_point: Some("path_trace"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ---- display pipeline ---------------------------------------------
        let display_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("display bgl"),
            entries: &[uniform_entry(0), storage_entry(1)],
        });

        let display_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tonemap pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("display layout"),
                bind_group_layouts: &[Some(&display_bgl)],
                immediate_size: 0,
            })),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_fullscreen"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_tonemap"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: DISPLAY_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let (display_texture, display_view) = create_display_texture(&device, width, height);
        let display_bind_group = make_display_bind_group(&device, &display_bgl, &uniforms, &accum);

        Self {
            device,
            queue,
            width,
            height,
            sample_index: 0,
            adapter_name,
            triangle_count: scene.indices.len() / 3,
            blas_build_ms,
            scene,
            uniforms,
            accum,
            positions_buf,
            normals_buf,
            indices_buf,
            tri_mat_buf,
            materials_buf,
            compute_bgl,
            compute_pipeline,
            compute_bind_group,
            display_pipeline,
            display_bgl,
            display_bind_group,
            display_texture,
            display_view,
            _blas: blas,
            tlas,
        }
    }

    pub fn scene(&self) -> &Scene {
        &self.scene
    }

    pub fn reset(&mut self) {
        // Zero the accumulation buffer so the next frame starts from scratch.
        self.queue.write_buffer(&self.accum, 0, &vec![0u8; (self.width as usize) * (self.height as usize) * 16]);
        self.sample_index = 0;
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == self.width && height == self.height {
            return;
        }
        self.width = width;
        self.height = height;
        self.accum = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("accumulation"),
            size: (width as u64) * (height as u64) * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (tex, view) = create_display_texture(&self.device, width, height);
        self.display_texture = tex;
        self.display_view = view;
        self.display_bind_group =
            make_display_bind_group(&self.device, &self.display_bgl, &self.uniforms, &self.accum);
        // The compute pass writes into `accum`, so its bind group has to point at
        // the new buffer as well - otherwise the tonemap samples a buffer that is
        // never written and the image stays black.
        self.compute_bind_group = self.make_compute_bind_group();
        self.sample_index = 0;
    }

    /// Renders `samples` additional samples and returns the tonemapped texture.
    pub fn render(&mut self, settings: &Settings, samples: u32) -> &wgpu::TextureView {
        for _ in 0..samples.max(1) {
            self.write_uniforms(settings);
            self.encode_and_submit();
            self.sample_index += 1;
        }
        &self.display_view
    }

    fn write_uniforms(&self, s: &Settings) {
        let (fwd, right, up) = s.camera.basis();
        let eye = s.camera.eye();
        let u = Uniforms {
            cam_pos: [eye.x, eye.y, eye.z, 1.0],
            cam_fwd: [fwd.x, fwd.y, fwd.z, (s.camera.fov_y * 0.5).tan()],
            cam_right: [right.x, right.y, right.z, 0.0],
            cam_up: [up.x, up.y, up.z, 0.0],
            resolution: [
                self.width as f32,
                self.height as f32,
                self.sample_index as f32,
                s.max_bounces as f32,
            ],
            params: [s.roughness_scale, s.exposure, 0.0, s.light_intensity],
            sky: [s.sky.x, s.sky.y, s.sky.z, 0.0],
            light_origin: [
                self.scene.light_origin.x,
                self.scene.light_origin.y,
                self.scene.light_origin.z,
                0.0,
            ],
            light_u: [self.scene.light_u.x, self.scene.light_u.y, self.scene.light_u.z, 0.0],
            light_v: [self.scene.light_v.x, self.scene.light_v.y, self.scene.light_v.z, 0.0],
            light_emission: [
                self.scene.light_radiance.x,
                self.scene.light_radiance.y,
                self.scene.light_radiance.z,
                0.0,
            ],
        };
        self.queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&u));
    }

    fn encode_and_submit(&self) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("frame"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("path trace"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &self.compute_bind_group, &[]);
            cpass.dispatch_workgroups((self.width + 7) / 8, (self.height + 7) / 8, 1);
        }
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tonemap"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.display_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.set_pipeline(&self.display_pipeline);
            rpass.set_bind_group(0, &self.display_bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }
        self.queue.submit([encoder.finish()]);
    }

    /// Reads the display texture back and writes it as a PNG (used by the
    /// headless `--out` mode and by the UI's "save image" button).
    pub fn save_png(&self, path: &std::path::Path) -> anyhow_like::Result<()> {
        let bytes_per_row = self.width * 4;
        let padded = (bytes_per_row + 255) / 256 * 256;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (padded as u64) * (self.height as u64),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.display_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);

        let slice = buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        }).ok();

        let data = slice.get_mapped_range().expect("map readback");
        let mut pixels = Vec::with_capacity((self.width * self.height * 4) as usize);
        for row in 0..self.height {
            let start = (row * padded) as usize;
            let end = start + bytes_per_row as usize;
            pixels.extend_from_slice(&data[start..end]);
        }
        drop(data);
        buffer.unmap();

        let file = std::fs::File::create(path)?;
        let w = std::io::BufWriter::new(file);
        let mut encoder = png::Encoder::new(w, self.width, self.height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header()?;
        writer.write_image_data(&pixels)?;
        Ok(())
    }
}


impl PathTracer {
    /// Bind group used by the compute pass; rebuilt whenever `accum` is
    /// reallocated (i.e. on resize).
    fn make_compute_bind_group(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compute bg"),
            layout: &self.compute_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.uniforms.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.tlas.as_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.positions_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.normals_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.indices_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.tri_mat_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.materials_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: self.accum.as_entire_binding() },
            ],
        })
    }
}
fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn create_display_texture(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("display"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DISPLAY_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn make_display_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniforms: &wgpu::Buffer,
    accum: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("display bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: accum.as_entire_binding() },
        ],
    })
}

/// Tiny error alias so we do not need `anyhow` for the few fallible paths.
pub mod anyhow_like {
    pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
}






