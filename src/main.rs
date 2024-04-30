use std::{borrow::Cow, collections::BTreeMap, fs::File, io::Write, iter, mem};

use bytemuck::{Pod, Zeroable};
use rand::{rngs::StdRng, Rng, SeedableRng};
use wgpu::util::DeviceExt;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

mod query;

use query::Queries;

const QUADS_LEN: usize = 3_700_000;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Quad {
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    depth: u32,
}

async fn run_triangles(event_loop: EventLoop<()>, window: Window, quads: &[Quad]) {
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
                    | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
                required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
            },
            None,
        )
        .await
        .unwrap();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("triangles.wgsl"))),
    });

    let triangles_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("triangles_bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(2 * mem::size_of::<f32>() as u64),
                },
                count: None,
            }],
        });

    let triangles_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("triangles_pipeline_layout"),
            bind_group_layouts: &[&triangles_bind_group_layout],
            push_constant_ranges: &[],
        });

    let vertices: Vec<f32> = quads
        .iter()
        .map(|q| {
            let depth = q.depth as f32 / 1_024.0;

            let top_left = [q.x0 as f32, q.y0 as f32, depth];
            let top_right = [q.x1 as f32, q.y0 as f32, depth];
            let bottom_right = [q.x1 as f32, q.y1 as f32, depth];
            let bottom_left = [q.x0 as f32, q.y1 as f32, depth];

            [
                [top_left, top_right, bottom_left].concat(),
                [top_right, bottom_right, bottom_left].concat(),
            ]
            .concat()
        })
        .flatten()
        .collect();

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("vertex_buffer"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let mut size_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("size_buffer"),
        contents: bytemuck::cast_slice(&[size.width as f32, size.height as f32]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let swapchain_capabilities = surface.get_capabilities(&adapter);
    let swapchain_format = swapchain_capabilities.formats[0];

    let mut config = surface
        .get_default_config(&adapter, size.width, size.height)
        .unwrap();
    surface.configure(&device, &config);

    let triangles_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("triangles_pipeline"),
        layout: Some(&triangles_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            compilation_options: Default::default(),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                }],
            }],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            compilation_options: Default::default(),
            targets: &[Some(swapchain_format.into())],
        }),
        primitive: wgpu::PrimitiveState {
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    let mut queries = Queries::new(&device, 2);

    let window = &window;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop
        .run(move |event, target| match event {
            Event::WindowEvent {
                window_id: _,
                event,
            } => {
                match event {
                    WindowEvent::Resized(new_size) => {
                        size_buffer.destroy();
                        size_buffer =
                            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("size_buffer"),
                                contents: bytemuck::cast_slice(&[
                                    new_size.width as f32,
                                    new_size.height as f32,
                                ]),
                                usage: wgpu::BufferUsages::UNIFORM,
                            });

                        config.width = new_size.width.max(1);
                        config.height = new_size.height.max(1);
                        surface.configure(&device, &config);
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
                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &triangles_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: size_buffer.as_entire_binding(),
                    }],
                });

                queries.reset();

                queries.write_next_timestamp(&mut encoder);

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

                    rpass.set_bind_group(0, &bind_group, &[]);
                    rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
                    rpass.set_pipeline(&triangles_pipeline);
                    rpass.draw(0..vertices.len() as u32 / 3, 0..1);
                }

                queries.write_next_timestamp(&mut encoder);
                queries.resolve(&mut encoder);

                queue.submit(Some(encoder.finish()));
                frame.present();

                dbg!(queries.wait_for_results(&device, &queue));
            }
            _ => (),
        })
        .unwrap();
}

const BLOCK_SIZE: u32 = 16;
const WIDTH: u32 = 4_096;
const HEIGHT: u32 = 2_048;

fn split_into_blocks(quads: &[Quad]) -> (Vec<u32>, Vec<u32>) {
    let mut blocks = BTreeMap::new();

    for (i, q) in quads.iter().enumerate() {
        let i = i as u32;

        for x in q.x0 / BLOCK_SIZE..q.x1.div_ceil(BLOCK_SIZE) {
            for y in q.y0 / BLOCK_SIZE..q.y1.div_ceil(BLOCK_SIZE) {
                blocks
                    .entry((y, x))
                    .and_modify(|indices: &mut Vec<u32>| indices.push(i))
                    .or_insert_with(|| vec![i]);
            }
        }
    }

    let indices = blocks
        .values()
        .map(|b| b.len() as u32)
        .chain(iter::once(0))
        .scan(0, |state, len| {
            let next_index = *state + len;
            Some(mem::replace(state, next_index))
        })
        .collect();

    let blocks = blocks.values().flatten().copied().collect();

    (indices, blocks)
}

async fn run_compute(event_loop: EventLoop<()>, window: Window, quads: &[Quad]) {
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
                    | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
                    | wgpu::Features::SUBGROUP,
                required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
            },
            None,
        )
        .await
        .unwrap();

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("shader_module"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("compute.wgsl"))),
    });

    let count_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("count_pipeline"),
        layout: None,
        module: &shader_module,
        entry_point: "count",
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &[].into(),
            zero_initialize_workgroup_memory: false,
        },
    });
    let prefix_sum_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("prefix_sum_pipeline"),
        layout: None,
        module: &shader_module,
        entry_point: "prefixSum",
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &[].into(),
            zero_initialize_workgroup_memory: false,
        },
    });
    let scatter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("scatter_pipeline"),
        layout: None,
        module: &shader_module,
        entry_point: "scatter",
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &[].into(),
            zero_initialize_workgroup_memory: false,
        },
    });
    let rasterize_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("rasterize_pipeline"),
        layout: None,
        module: &shader_module,
        entry_point: "rasterize",
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &[].into(),
            zero_initialize_workgroup_memory: false,
        },
    });

    let count_bind_group_layout = count_pipeline.get_bind_group_layout(0);
    let prefix_sum_bind_group_layout = prefix_sum_pipeline.get_bind_group_layout(0);
    let scatter_bind_group_layout = scatter_pipeline.get_bind_group_layout(0);
    let rasterize_bind_group_layout = rasterize_pipeline.get_bind_group_layout(0);

    let (indices, blocks) = split_into_blocks(quads);

    let quad_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("quad_buffer"),
        contents: bytemuck::cast_slice(&quads),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("count_buffer"),
        size: WIDTH.div_ceil(BLOCK_SIZE) as u64
            * HEIGHT.div_ceil(BLOCK_SIZE) as u64
            * mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let norm_quad_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("norm_quad_buffer"),
        size: blocks.len() as u64 * mem::size_of::<[u32; 2]>() as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("index_buffer"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth_texture"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rg32Uint,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let depth_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("depth_buffer"),
        size: WIDTH as u64 * HEIGHT as u64 * mem::size_of::<[u32; 2]>() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    let count_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("count_bind_group"),
        layout: &count_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: quad_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: count_buffer.as_entire_binding(),
            },
        ],
    });
    let prefix_sum_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("prefix_sum_bind_group"),
        layout: &prefix_sum_bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 1,
            resource: count_buffer.as_entire_binding(),
        }],
    });
    let scatter_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scatter_bind_group"),
        layout: &scatter_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: quad_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: count_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: norm_quad_buffer.as_entire_binding(),
            },
        ],
    });
    let rasterize_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rasterize_bind_group"),
        layout: &rasterize_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 2,
                resource: norm_quad_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: index_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(
                    &depth_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                ),
            },
        ],
    });

    let mut queries = Queries::new(&device, 6);

    queries.write_next_timestamp(&mut encoder);

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        cpass.set_pipeline(&count_pipeline);
        cpass.set_bind_group(0, &count_bind_group, &[]);
        cpass.dispatch_workgroups(QUADS_LEN.div_ceil(256) as u32, 1, 1);
    }

    queries.write_next_timestamp(&mut encoder);

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        cpass.set_pipeline(&prefix_sum_pipeline);
        cpass.set_bind_group(0, &prefix_sum_bind_group, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    }

    queries.write_next_timestamp(&mut encoder);

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        cpass.set_pipeline(&scatter_pipeline);
        cpass.set_bind_group(0, &scatter_bind_group, &[]);
        cpass.dispatch_workgroups(QUADS_LEN.div_ceil(256) as u32, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&count_buffer, 0, &index_buffer, 0, count_buffer.size());

    queries.write_next_timestamp(&mut encoder);

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        cpass.set_pipeline(&rasterize_pipeline);
        cpass.set_bind_group(0, &rasterize_bind_group, &[]);
        cpass.dispatch_workgroups(WIDTH.div_ceil(BLOCK_SIZE), HEIGHT.div_ceil(BLOCK_SIZE), 1);
    }

    queries.write_next_timestamp(&mut encoder);
    queries.resolve(&mut encoder);

    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTextureBase {
            texture: &depth_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::default(),
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBufferBase {
            buffer: &depth_buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(WIDTH as u32 * mem::size_of::<[u32; 2]>() as u32),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
    );

    queue.submit(Some(encoder.finish()));

    let values = depth_buffer.slice(..);
    values.map_async(wgpu::MapMode::Read, |_| ());

    dbg!(queries.wait_for_results(&device, &queue));

    let slice = values.get_mapped_range();
    let values: &[[u32; 2]] = bytemuck::cast_slice(&slice);

    fn color_from_index(index: u32) -> [u8; 3] {
        if index == u32::MAX {
            return [0; 3];
        }

        let seed = index * 3;

        let r = (seed * 279470273) % 255;
        let g = ((seed + 1) * 279470273) % 255;
        let b = ((seed + 2) * 279470273) % 255;

        [r, g, b].map(|c| c as u8)
    }

    let bytes: Vec<_> = values
        .iter()
        .map(|[_, i]| color_from_index(*i))
        .flatten()
        .collect();

    let new_path = "capture.ppm";
    let mut output = File::options()
        .write(true)
        .create(true)
        .open(new_path)
        .unwrap();
    output
        .write_all(format!("P6\n{} {}\n255\n", WIDTH, HEIGHT,).as_bytes())
        .unwrap();
    output.write_all(&bytes).unwrap();
}

fn gen_uniformly_random_quads<R: Rng>(rng: &mut R) -> Vec<Quad> {
    (0..QUADS_LEN)
        .into_iter()
        .map(|_| {
            const MAX_SIZE: u32 = 4;

            let x0 = rng.gen_range(0..=WIDTH);
            let y0 = rng.gen_range(0..=HEIGHT);

            Quad {
                x0,
                y0,
                x1: (rng.gen_range(1..=MAX_SIZE) + x0).min(WIDTH),
                y1: (rng.gen_range(1..=MAX_SIZE) + y0).min(HEIGHT),
                depth: rng.gen_range(0..=1_024),
            }
        })
        .collect()
}

fn gen_clustered_random_quads<R: Rng>(rng: &mut R) -> Vec<Quad> {
    fn new_cluster<R: Rng>(rng: &mut R) -> [u32; 2] {
        [rng.gen_range(0..=WIDTH), rng.gen_range(0..=HEIGHT)]
    }

    (0..QUADS_LEN)
        .into_iter()
        .scan(new_cluster(rng), |state, _| {
            if rng.gen_bool(1.0 / 16.0) {
                *state = new_cluster(rng);
            }

            let cluster = *state;

            const MAX_SIZE: u32 = 4;

            let x0 = rng.gen_range(cluster[0]..=cluster[0] + BLOCK_SIZE);
            let y0 = rng.gen_range(cluster[1]..=cluster[1] + BLOCK_SIZE);

            Some(Quad {
                x0,
                y0,
                x1: (rng.gen_range(1..=MAX_SIZE) + x0).min(WIDTH),
                y1: (rng.gen_range(1..=MAX_SIZE) + y0).min(HEIGHT),
                depth: rng.gen_range(0..=1_024),
            })
        })
        .collect()
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let mut rng = StdRng::seed_from_u64(42);
    let quads = gen_uniformly_random_quads(&mut rng);
    // let quads = gen_clustered_random_quads(&mut rng);

    // let quads = [
    //     Quad {
    //         x0: 10,
    //         y0: 10,
    //         x1: 210,
    //         y1: 210,
    //         depth: 1,
    //     },
    //     Quad {
    //         x0: 50,
    //         y0: 50,
    //         x1: 250,
    //         y1: 250,
    //         depth: 2,
    //     },
    // ];

    // pollster::block_on(run_triangles(event_loop, window, &quads));
    pollster::block_on(run_compute(event_loop, window, &quads));
}
