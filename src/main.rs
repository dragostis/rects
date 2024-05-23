use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

mod query;

use query::Queries;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Config {
    width: u32,
    height: u32,
}

async fn run(event_loop: EventLoop<()>, window: Window) {
    let mut size = window.inner_size();
    size.width = size.width.max(1);
    size.height = size.height.max(1);

    let instance = wgpu::Instance::default();

    let surface = instance.create_surface(&window).unwrap();
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
                required_features: wgpu::Features::TIMESTAMP_QUERY
                    | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES
                    | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
            },
            None,
        )
        .await
        .unwrap();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("ssgi.wgsl"))),
    });

    let mut gbuffer = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("gbuffer"),
        size: wgpu::Extent3d {
            width: size.width,
            height: size.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    let mut image = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("image"),
        size: wgpu::Extent3d {
            width: size.width,
            height: size.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    let mut size_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("size_buffer"),
        contents: bytemuck::bytes_of(&[size.width, size.height]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let render_gbuffer_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("render_gbuffer_pipeline"),
            layout: None,
            module: &shader,
            entry_point: "renderGbuffer",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
    let render_gbuffer_group_layout = render_gbuffer_pipeline.get_bind_group_layout(0);

    let ssgi_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ssgi_pipeline"),
            layout: None,
            module: &shader,
            entry_point: "ssgi",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
    let ssgi_group_layout = ssgi_pipeline.get_bind_group_layout(0);

    let display_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("display_bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::ReadWrite,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            }],
        });

    let display_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("display_pipeline_layout"),
        bind_group_layouts: &[&display_bind_group_layout],
        push_constant_ranges: &[],
    });

    let swapchain_capabilities = surface.get_capabilities(&adapter);
    let swapchain_format = swapchain_capabilities.formats[0];

    let mut config = surface
        .get_default_config(&adapter, size.width, size.height)
        .unwrap();
    surface.configure(&device, &config);

    let display_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("triangles_pipeline"),
        layout: Some(&display_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "displayVertex",
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "displayFragment",
            compilation_options: Default::default(),
            targets: &[Some(swapchain_format.into())],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    let mut queries = Queries::new(&device, 6);

    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop
        .run(move |event, target| match event {
            Event::WindowEvent {
                window_id: _,
                event,
            } => {
                match event {
                    WindowEvent::Resized(new_size) => {
                        gbuffer.destroy();
                        gbuffer = device.create_texture(&wgpu::TextureDescriptor {
                            label: Some("gbuffer"),
                            size: wgpu::Extent3d {
                                width: new_size.width,
                                height: new_size.height,
                                depth_or_array_layers: 1,
                            },
                            mip_level_count: 1,
                            sample_count: 1,
                            dimension: wgpu::TextureDimension::D2,
                            format: wgpu::TextureFormat::Rgba32Float,
                            usage: wgpu::TextureUsages::STORAGE_BINDING,
                            view_formats: &[],
                        });

                        image.destroy();
                        image = device.create_texture(&wgpu::TextureDescriptor {
                            label: Some("image"),
                            size: wgpu::Extent3d {
                                width: new_size.width,
                                height: new_size.height,
                                depth_or_array_layers: 1,
                            },
                            mip_level_count: 1,
                            sample_count: 1,
                            dimension: wgpu::TextureDimension::D2,
                            format: wgpu::TextureFormat::Rgba16Float,
                            usage: wgpu::TextureUsages::STORAGE_BINDING,
                            view_formats: &[],
                        });

                        size_buffer.destroy();
                        size_buffer =
                            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("size_buffer"),
                                contents: bytemuck::bytes_of(&[new_size.width, new_size.height]),
                                usage: wgpu::BufferUsages::UNIFORM,
                            });

                        config.width = new_size.width.max(1);
                        config.height = new_size.height.max(1);
                        surface.configure(&device, &config);

                        size = new_size;
                    }
                    WindowEvent::CloseRequested => target.exit(),
                    _ => (),
                };
            }
            Event::AboutToWait => {
                let frame = surface.get_current_texture().unwrap();
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                let render_gbuffer_bind_group =
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &render_gbuffer_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(
                                    &gbuffer.create_view(&wgpu::TextureViewDescriptor::default()),
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: size_buffer.as_entire_binding(),
                            },
                        ],
                    });
                
                let ssgi_bind_group =
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &ssgi_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(
                                    &gbuffer.create_view(&wgpu::TextureViewDescriptor::default()),
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(
                                    &image.create_view(&wgpu::TextureViewDescriptor::default()),
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: size_buffer.as_entire_binding(),
                            },
                        ],
                    });

                let display_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &display_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &image.create_view(&wgpu::TextureViewDescriptor::default()),
                        ),
                    }],
                });

                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

                queries.reset();

                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });

                    queries.write_next_timestamp(&mut cpass);

                    cpass.set_pipeline(&render_gbuffer_pipeline);
                    cpass.set_bind_group(0, &render_gbuffer_bind_group, &[]);
                    cpass.dispatch_workgroups(size.width.div_ceil(16), size.height.div_ceil(8), 1);

                    queries.write_next_timestamp(&mut cpass);
                }

                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });

                    queries.write_next_timestamp(&mut cpass);

                    cpass.set_pipeline(&ssgi_pipeline);
                    cpass.set_bind_group(0, &ssgi_bind_group, &[]);
                    cpass.dispatch_workgroups(size.width.div_ceil(16), size.height.div_ceil(8), 1);

                    queries.write_next_timestamp(&mut cpass);
                }

                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });

                    queries.write_next_timestamp(&mut rpass);

                    rpass.set_bind_group(0, &display_bind_group, &[]);
                    rpass.set_pipeline(&display_pipeline);
                    rpass.draw(0..3, 0..1);

                    queries.write_next_timestamp(&mut rpass);
                }

                queries.resolve(&mut encoder);

                queue.submit(Some(encoder.finish()));
                frame.present();

                dbg!(queries.wait_for_results(&device, &queue));
            }
            _ => (),
        })
        .unwrap();
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    pollster::block_on(run(event_loop, window));
}
