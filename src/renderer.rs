use std::sync::Arc;

use image::EncodableLayout;
use rand::Rng;
use winit::window::Window;

use crate::Image;

pub struct Renderer<'a> {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'a>,
    image_texture: wgpu::Texture,
    k_means_state: KmeansState,
    composite_texture: wgpu::Texture,
    composite_bindgroup: wgpu::BindGroup,
    composite_pipeline: wgpu::ComputePipeline,
    k_means_done: bool,
}

impl<'a> Renderer<'a> {
    pub async fn new(window: Arc<Window>, image: Image) -> anyhow::Result<Self> {
        let dims = image.dimensions();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: wgpu::InstanceFlags::from_build_config(),
            ..Default::default()
        });
        let surface = instance.create_surface(window)?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::SHADER_INT64_ATOMIC_ALL_OPS
                        | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                        | wgpu::Features::PUSH_CONSTANTS
                        | wgpu::Features::SHADER_INT64
                        | wgpu::Features::BGRA8UNORM_STORAGE,
                    required_limits: wgpu::Limits {
                        max_push_constant_size: 4,
                        ..Default::default()
                    },
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await?;
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb() && *f == wgpu::TextureFormat::Bgra8Unorm)
            .unwrap_or(surface_caps.formats[0]);
        let surface_config = wgpu::SurfaceConfiguration {
            // I'm guessing we need TEXTURE_BINDING to use in a compute shader
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: dims.0,
            height: dims.1,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 1,
        };
        surface.configure(&device, &surface_config);

        let image_texture = Self::upload_image(&device, &queue, &image);
        let composite_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: image.width(),
                height: image.height(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8Unorm,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let k_means_state = KmeansState::new(&device, image_texture.clone(), 10)?;
        let composite_bindgroup_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::ReadOnly,
                            format: wgpu::TextureFormat::R32Uint,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Bgra8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });
        let composite_bindgroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &composite_bindgroup_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &k_means_state
                            .assignment_texture
                            .create_view(&Default::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(
                        k_means_state.centroids[1].as_entire_buffer_binding(), // TODO: need 2 bindgroups here
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        &composite_texture.create_view(&Default::default()),
                    ),
                },
            ],
        });
        let shader_module =
            device.create_shader_module(wgpu::include_wgsl!("shaders/composite.wgsl"));
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&composite_bindgroup_layout],
            push_constant_ranges: &[],
        });
        let composite_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Assignment pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Ok(Self {
            device,
            queue,
            surface,
            image_texture,
            k_means_state,
            composite_texture,
            composite_bindgroup,
            composite_pipeline,
            k_means_done: false,
        })
    }

    // Queue image upload
    fn upload_image(device: &wgpu::Device, queue: &wgpu::Queue, buf: &Image) -> wgpu::Texture {
        let size = wgpu::Extent3d {
            width: buf.width(),
            height: buf.height(),
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Image Buffer"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            buf.as_bytes(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * buf.width()),
                rows_per_image: Some(buf.height()),
            },
            size,
        );
        queue.submit([]);
        texture
    }

    pub fn render(&mut self) -> anyhow::Result<()> {
        let mut command_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("commands"),
                });
        if !self.k_means_done {
            println!("Running");
            // self.k_means_done = true;
            self.k_means_state.run(&self.device, &self.queue)
        } else {
        };
        let cur_texture = self.surface.get_current_texture()?;
        let _ = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &cur_texture
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor {
                        label: None,
                        format: Some(cur_texture.texture.format()),
                        usage: Some(wgpu::TextureUsages::RENDER_ATTACHMENT),
                        ..Default::default()
                    }),
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 1.0,
                        g: 1.0,
                        b: 1.0,
                        a: 0.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        {
            let mut composite_pass =
                command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
            composite_pass.set_pipeline(&self.composite_pipeline);
            composite_pass.set_bind_group(0, &self.composite_bindgroup, &[]);
            composite_pass.dispatch_workgroups(
                self.image_texture.width(),
                self.image_texture.height(),
                1,
            );
        }

        command_encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfoBase {
                texture: &self.composite_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfoBase {
                texture: &cur_texture.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.image_texture.width(),
                height: self.image_texture.height(),
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([command_encoder.finish()]);
        cur_texture.present();
        Ok(())
    }
}

struct KmeansState {
    // input image
    image_texture: wgpu::Texture,
    assignment_texture: wgpu::Texture,
    centroids: [wgpu::Buffer; 2],
    count_buf: wgpu::Buffer,
    assignment_pipeline: wgpu::ComputePipeline,
    assignment_bind_groups: [wgpu::BindGroup; 2],
    phase2_pipeline: wgpu::ComputePipeline,
    k: u32,
}

impl KmeansState {
    fn create_phase2_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::ComputePipeline {
        let shader = wgpu::include_wgsl!("shaders/assignment.wgsl");
        let shader_module = device.create_shader_module(shader);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..4,
            }],
        });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&layout),
            module: &shader_module,
            entry_point: Some("phase2"),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    fn new(device: &wgpu::Device, input_image: wgpu::Texture, k: u32) -> anyhow::Result<Self> {
        let centroid_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Centroids Buffer 1"),
            size: 8 * 4 * (k as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let centroid_buf2 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Centroids Buffer 2"),
            size: 8 * 4 * (k as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let count_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Counts Buffer"),
            size: 8 * k as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let assignment_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Assignment Buffer"),
            size: input_image.size(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let shader = wgpu::include_wgsl!("shaders/assignment.wgsl");
        let shader_module = device.create_shader_module(shader);
        let assign_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("assignment bind group"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::ReadOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::R32Uint,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&assign_bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..4,
            }],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Assignment pipeline"),
            layout: Some(&layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let create_bind_group = |centroid_buf1: &wgpu::Buffer, centroid_buf2: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &assign_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &input_image.create_view(&Default::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(
                            centroid_buf1.as_entire_buffer_binding(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(
                            centroid_buf2.as_entire_buffer_binding(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Buffer(
                            count_buf.as_entire_buffer_binding(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(
                            &assignment_texture.create_view(&Default::default()),
                        ),
                    },
                ],
            })
        };
        let assignment_bind_groups = [
            create_bind_group(&centroid_buf, &centroid_buf2),
            create_bind_group(&centroid_buf2, &centroid_buf),
        ];
        Ok(Self {
            image_texture: input_image,
            assignment_texture,
            centroids: [centroid_buf, centroid_buf2],
            count_buf,
            assignment_pipeline: pipeline,
            assignment_bind_groups,
            phase2_pipeline: Self::create_phase2_pipeline(device, &assign_bind_group_layout),
            k,
        })
    }

    fn run(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        assert!(self.k == 10);
        let mut rng = rand::rng();
        let zeros = vec![0; self.count_buf.size() as usize];
        let mut centroid_buf = vec![[255u64; 4]; self.k as usize];
        centroid_buf.iter_mut().for_each(|c| {
            c.iter_mut()
                .take(3)
                .for_each(|v| *v = rng.random_range(0..256))
        });
        queue.write_buffer(&self.centroids[0], 0, bytemuck::cast_slice(&centroid_buf));
        queue.write_buffer(&self.centroids[1], 0, bytemuck::cast_slice(&centroid_buf));
        queue.write_buffer(&self.count_buf, 0, &zeros);
        centroid_buf.iter_mut().for_each(|c| *c = [0; 4]);

        for i in 0..53 {
            {
                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut compute_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: None,
                            timestamp_writes: None,
                        });
                    compute_pass.set_pipeline(&self.assignment_pipeline);
                    compute_pass.set_push_constants(0, &self.k.to_le_bytes()[..]);
                    compute_pass.set_bind_group(0, &self.assignment_bind_groups[i % 2], &[]);
                    compute_pass.dispatch_workgroups(
                        self.image_texture.width().div_ceil(8),
                        self.image_texture.height().div_ceil(8),
                        1,
                    );
                    compute_pass.set_pipeline(&self.phase2_pipeline);
                    compute_pass.set_bind_group(0, &self.assignment_bind_groups[i % 2], &[]);
                    compute_pass.dispatch_workgroups(self.k, 1, 1);
                }

                queue.submit([encoder.finish()]);
            }
        }
        // encoder.finish()
    }
}
