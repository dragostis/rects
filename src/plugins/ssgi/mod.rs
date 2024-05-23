use std::{hash, mem};

use bevy::{
    app::{App, Plugin},
    asset::{load_internal_asset, Handle},
    core_pipeline::{
        core_3d::{
            graph::{Core3d, Node3d},
            Camera3d,
        },
        fullscreen_vertex_shader::fullscreen_shader_vertex_state,
        prepass::{DeferredPrepass, DepthPrepass, NormalPrepass, ViewPrepassTextures},
    },
    ecs::{
        bundle::Bundle,
        component::Component,
        entity::Entity,
        query::{Has, QueryItem, With},
        reflect::ReflectComponent,
        schedule::IntoSystemConfigs,
        system::{Commands, Query, Res, ResMut, Resource},
        world::{FromWorld, World},
    },
    log::{error, warn},
    reflect::Reflect,
    render::{
        camera::{Camera, ExtractedCamera, TemporalJitter},
        extract_component::ExtractComponent,
        globals::{GlobalsBuffer, GlobalsUniform},
        render_graph::{
            NodeRunError, RenderGraphApp, RenderGraphContext, RenderLabel, ViewNode, ViewNodeRunner,
        },
        render_resource::{
            binding_types::{
                sampler, texture_2d, texture_depth_2d, texture_storage_2d, uniform_buffer,
            },
            AddressMode, BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries,
            CachedComputePipelineId, CachedRenderPipelineId, ColorTargetState, ColorWrites,
            ComputePassDescriptor, ComputePipelineDescriptor, Extent3d, FilterMode, FragmentState,
            MultisampleState, Operations, PipelineCache, PrimitiveState, RenderPassColorAttachment,
            RenderPassDescriptor, RenderPipelineDescriptor, Sampler, SamplerBindingType,
            SamplerDescriptor, Shader, ShaderDefVal, ShaderStages, SpecializedComputePipeline,
            SpecializedComputePipelines, StorageTextureAccess, TextureDataOrder, TextureDescriptor,
            TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureView,
            TextureViewDescriptor, TextureViewDimension,
        },
        renderer::{RenderAdapter, RenderContext, RenderDevice, RenderQueue},
        texture::{CachedTexture, TextureCache},
        view::{Msaa, ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms},
        Extract, ExtractSchedule, Render, RenderApp, RenderSet,
    },
    utils::default,
};

const PREPROCESS_DEPTH_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(378907431010893);
const SSGI_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(406305990723329);
const SPATIAL_DENOISE_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(342193469482837);
const SSGI_UTILS_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(69792351744019);
const POST_PROCESS_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(515340081393064);

pub struct ScreenSpaceGlobalIlluminationPlugin;

impl Plugin for ScreenSpaceGlobalIlluminationPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            PREPROCESS_DEPTH_SHADER_HANDLE,
            "preprocess_depth.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(app, SSGI_SHADER_HANDLE, "ssgi.wgsl", Shader::from_wgsl);
        load_internal_asset!(
            app,
            SPATIAL_DENOISE_SHADER_HANDLE,
            "spatial_denoise.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            SSGI_UTILS_SHADER_HANDLE,
            "ssgi_utils.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            POST_PROCESS_SHADER_HANDLE,
            "post_process.wgsl",
            Shader::from_wgsl
        );
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        if !render_app
            .world()
            .resource::<RenderAdapter>()
            .get_texture_format_features(TextureFormat::Rgba16Float)
            .allowed_usages
            .contains(TextureUsages::STORAGE_BINDING)
        {
            warn!("ScreenSpaceGlobalIlluminationPlugin not loaded. GPU lacks support: TextureFormat::Rgba16Float does not support TextureUsages::STORAGE_BINDING.");
            return;
        }

        if render_app
            .world()
            .resource::<RenderDevice>()
            .limits()
            .max_storage_textures_per_shader_stage
            < 5
        {
            warn!("ScreenSpaceGlobalIlluminationPlugin not loaded. GPU lacks support: Limits::max_storage_textures_per_shader_stage is less than 5.");
            return;
        }

        render_app
            .init_resource::<SsgiPipelines>()
            .init_resource::<SpecializedComputePipelines<SsgiPipelines>>()
            .add_systems(ExtractSchedule, extract_ssgi_settings)
            .add_systems(
                Render,
                (
                    prepare_ssgi_pipelines.in_set(RenderSet::Prepare),
                    prepare_ssgi_textures.in_set(RenderSet::PrepareResources),
                    prepare_ssgi_bind_groups.in_set(RenderSet::PrepareBindGroups),
                ),
            )
            .add_render_graph_node::<ViewNodeRunner<SsgiNode>>(
                Core3d,
                ScreenSpaceGlobalIllumination,
            )
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::Tonemapping,
                    ScreenSpaceGlobalIllumination,
                    Node3d::EndMainPassPostProcessing,
                ),
            );
    }
}

#[derive(Bundle, Default)]
pub struct ScreenSpaceGlobalIlluminationBundle {
    pub depth_prepass: DepthPrepass,
    pub normal_prepass: NormalPrepass,
    pub deferred_prepass: DeferredPrepass,
    pub settings: ScreenSpaceGlobalIlluminationSettings,
}

#[derive(Component, ExtractComponent, Reflect, Clone, Debug)]
#[reflect(Component)]
pub struct ScreenSpaceGlobalIlluminationSettings {
    pub slice_count: u32,
    pub samples_per_slice_side: u32,
    pub thickness: f32,
}

impl Default for ScreenSpaceGlobalIlluminationSettings {
    fn default() -> Self {
        Self {
            slice_count: 4,
            samples_per_slice_side: 16,
            thickness: 0.25,
        }
    }
}

impl Eq for ScreenSpaceGlobalIlluminationSettings {}

impl PartialEq for ScreenSpaceGlobalIlluminationSettings {
    fn eq(&self, other: &Self) -> bool {
        self.slice_count == other.slice_count
            && self.samples_per_slice_side == other.samples_per_slice_side
            && self.thickness == other.thickness
    }
}

impl hash::Hash for ScreenSpaceGlobalIlluminationSettings {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.slice_count.hash(state);
        self.samples_per_slice_side.hash(state);
        self.thickness.to_bits().hash(state);
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, RenderLabel)]
struct ScreenSpaceGlobalIllumination;

#[derive(Default)]
struct SsgiNode {}

impl ViewNode for SsgiNode {
    type ViewQuery = (
        &'static ViewTarget,
        &'static ExtractedCamera,
        &'static SsgiPipelineId,
        &'static SsgiBindGroups,
        &'static ScreenSpaceGlobalIlluminationTextures,
        &'static ViewUniformOffset,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (view_target, camera, pipeline_id, bind_groups, textures, view_uniform_offset): QueryItem<
            Self::ViewQuery,
        >,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pipelines = world.resource::<SsgiPipelines>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let (
            Some(camera_size),
            Some(preprocess_depth_pipeline),
            Some(spatial_denoise_pipeline),
            Some(ssgi_pipeline),
            Some(post_process_pipeline),
        ) = (
            camera.physical_viewport_size,
            pipeline_cache.get_compute_pipeline(pipelines.preprocess_depth_pipeline),
            pipeline_cache.get_compute_pipeline(pipelines.spatial_denoise_pipeline),
            pipeline_cache.get_compute_pipeline(pipeline_id.0),
            pipeline_cache.get_render_pipeline(pipelines.post_process_pipeline),
        )
        else {
            return Ok(());
        };

        render_context.command_encoder().push_debug_group("ssgi");

        {
            let mut preprocess_depth_pass =
                render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("ssgi_preprocess_depth_pass"),
                        timestamp_writes: None,
                    });
            preprocess_depth_pass.set_pipeline(preprocess_depth_pipeline);
            preprocess_depth_pass.set_bind_group(0, &bind_groups.preprocess_depth_bind_group, &[]);
            preprocess_depth_pass.set_bind_group(
                1,
                &bind_groups.common_bind_group,
                &[view_uniform_offset.offset],
            );
            preprocess_depth_pass.dispatch_workgroups(
                div_ceil(camera_size.x, 16),
                div_ceil(camera_size.y, 16),
                1,
            );
        }

        let post_process = view_target.post_process_write();

        {
            let bind_group = render_context.render_device().create_bind_group(
                "ssgi_deferred_output_bind_group",
                &pipelines.deferred_output_bind_group_layout,
                &BindGroupEntries::sequential((post_process.source,)),
            );

            let mut ssgi_pass =
                render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("ssgi_ssgi_pass"),
                        timestamp_writes: None,
                    });
            ssgi_pass.set_pipeline(ssgi_pipeline);
            ssgi_pass.set_bind_group(0, &bind_groups.ssgi_bind_group, &[]);
            ssgi_pass.set_bind_group(
                1,
                &bind_groups.common_bind_group,
                &[view_uniform_offset.offset],
            );
            ssgi_pass.set_bind_group(2, &bind_group, &[]);
            ssgi_pass.dispatch_workgroups(
                div_ceil(camera_size.x, 16),
                div_ceil(camera_size.y, 8),
                1,
            );
        }

        {
            let mut spatial_denoise_pass =
                render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("ssgi_spatial_denoise_pass"),
                        timestamp_writes: None,
                    });
            spatial_denoise_pass.set_pipeline(spatial_denoise_pipeline);
            spatial_denoise_pass.set_bind_group(0, &bind_groups.spatial_denoise_bind_group, &[]);
            spatial_denoise_pass.set_bind_group(
                1,
                &bind_groups.common_bind_group,
                &[view_uniform_offset.offset],
            );
            spatial_denoise_pass.dispatch_workgroups(
                div_ceil(camera_size.x, 8),
                div_ceil(camera_size.y, 8),
                1,
            );
        }

        {
            let bind_group = render_context.render_device().create_bind_group(
                "post_process_bind_group",
                &pipelines.post_process_bind_group_layout,
                &BindGroupEntries::sequential((
                    &textures
                        .screen_space_global_illumination_texture
                        .default_view,
                    &pipelines.point_clamp_sampler,
                )),
            );

            let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                label: Some("post_process_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: post_process.destination,
                    resolve_target: None,
                    ops: Operations::default(),
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_render_pipeline(post_process_pipeline);
            render_pass.set_bind_group(0, &bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }

        render_context.command_encoder().pop_debug_group();
        Ok(())
    }
}

#[derive(Resource)]
struct SsgiPipelines {
    preprocess_depth_pipeline: CachedComputePipelineId,
    spatial_denoise_pipeline: CachedComputePipelineId,
    post_process_pipeline: CachedRenderPipelineId,

    common_bind_group_layout: BindGroupLayout,
    preprocess_depth_bind_group_layout: BindGroupLayout,
    ssgi_bind_group_layout: BindGroupLayout,
    deferred_output_bind_group_layout: BindGroupLayout,
    spatial_denoise_bind_group_layout: BindGroupLayout,
    post_process_bind_group_layout: BindGroupLayout,

    hilbert_index_lut: TextureView,
    point_clamp_sampler: Sampler,
    linear_point_clamp_sampler: Sampler,
}

impl FromWorld for SsgiPipelines {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let render_queue = world.resource::<RenderQueue>();
        let pipeline_cache = world.resource::<PipelineCache>();

        let hilbert_index_lut = render_device
            .create_texture_with_data(
                render_queue,
                &(TextureDescriptor {
                    label: Some("ssgi_hilbert_index_lut"),
                    size: Extent3d {
                        width: HILBERT_WIDTH as u32,
                        height: HILBERT_WIDTH as u32,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::R16Uint,
                    usage: TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                }),
                TextureDataOrder::default(),
                bytemuck::cast_slice(&generate_hilbert_index_lut()),
            )
            .create_view(&TextureViewDescriptor::default());

        let point_clamp_sampler = render_device.create_sampler(&SamplerDescriptor {
            min_filter: FilterMode::Nearest,
            mag_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            ..Default::default()
        });
        let linear_point_clamp_sampler = render_device.create_sampler(&SamplerDescriptor {
            min_filter: FilterMode::Linear,
            mag_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Nearest,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            ..Default::default()
        });

        let common_bind_group_layout = render_device.create_bind_group_layout(
            "ssgi_common_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    sampler(SamplerBindingType::NonFiltering),
                    sampler(SamplerBindingType::Filtering),
                    uniform_buffer::<ViewUniform>(true),
                ),
            ),
        );

        let preprocess_depth_bind_group_layout = render_device.create_bind_group_layout(
            "ssgi_preprocess_depth_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    texture_depth_2d(),
                    texture_storage_2d(TextureFormat::R16Float, StorageTextureAccess::WriteOnly),
                    texture_storage_2d(TextureFormat::R16Float, StorageTextureAccess::WriteOnly),
                    texture_storage_2d(TextureFormat::R16Float, StorageTextureAccess::WriteOnly),
                    texture_storage_2d(TextureFormat::R16Float, StorageTextureAccess::WriteOnly),
                    texture_storage_2d(TextureFormat::R16Float, StorageTextureAccess::WriteOnly),
                ),
            ),
        );

        let ssgi_bind_group_layout = render_device.create_bind_group_layout(
            "ssgi_ssgi_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    texture_2d(TextureSampleType::Float { filterable: true }),
                    texture_2d(TextureSampleType::Float { filterable: false }),
                    texture_2d(TextureSampleType::Uint),
                    texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                    texture_storage_2d(TextureFormat::R32Uint, StorageTextureAccess::WriteOnly),
                    uniform_buffer::<GlobalsUniform>(false),
                ),
            ),
        );

        let deferred_output_bind_group_layout = render_device.create_bind_group_layout(
            "ssgi_deferred_output_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (texture_2d(TextureSampleType::Float { filterable: false }),),
            ),
        );

        let spatial_denoise_bind_group_layout = render_device.create_bind_group_layout(
            "ssgi_spatial_denoise_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    texture_2d(TextureSampleType::Float { filterable: false }),
                    texture_2d(TextureSampleType::Uint),
                    texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                ),
            ),
        );

        let post_process_bind_group_layout = render_device.create_bind_group_layout(
            "post_process_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    texture_2d(TextureSampleType::Float { filterable: false }),
                    sampler(SamplerBindingType::NonFiltering),
                ),
            ),
        );

        let preprocess_depth_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("ssgi_preprocess_depth_pipeline".into()),
                layout: vec![
                    preprocess_depth_bind_group_layout.clone(),
                    common_bind_group_layout.clone(),
                ],
                push_constant_ranges: vec![],
                shader: PREPROCESS_DEPTH_SHADER_HANDLE,
                shader_defs: Vec::new(),
                entry_point: "preprocess_depth".into(),
            });

        let spatial_denoise_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("ssgi_spatial_denoise_pipeline".into()),
                layout: vec![
                    spatial_denoise_bind_group_layout.clone(),
                    common_bind_group_layout.clone(),
                ],
                push_constant_ranges: vec![],
                shader: SPATIAL_DENOISE_SHADER_HANDLE,
                shader_defs: Vec::new(),
                entry_point: "spatial_denoise".into(),
            });

        let post_process_pipeline =
            pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
                label: Some("post_process_pipeline".into()),
                layout: vec![post_process_bind_group_layout.clone()],
                vertex: fullscreen_shader_vertex_state(),
                fragment: Some(FragmentState {
                    shader: POST_PROCESS_SHADER_HANDLE,
                    shader_defs: vec![],
                    entry_point: "fragment".into(),
                    targets: vec![Some(ColorTargetState {
                        format: TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    })],
                }),
                primitive: PrimitiveState::default(),
                depth_stencil: None,
                multisample: MultisampleState::default(),
                push_constant_ranges: vec![],
            });

        Self {
            preprocess_depth_pipeline,
            spatial_denoise_pipeline,

            common_bind_group_layout,
            preprocess_depth_bind_group_layout,
            ssgi_bind_group_layout,
            deferred_output_bind_group_layout,
            spatial_denoise_bind_group_layout,
            post_process_bind_group_layout,

            hilbert_index_lut,
            point_clamp_sampler,
            linear_point_clamp_sampler,
            post_process_pipeline,
        }
    }
}

#[derive(PartialEq, Eq, Hash, Clone)]
struct SsgiPipelineKey {
    ssgi_settings: ScreenSpaceGlobalIlluminationSettings,
    temporal_jitter: bool,
}

impl SpecializedComputePipeline for SsgiPipelines {
    type Key = SsgiPipelineKey;

    fn specialize(&self, key: Self::Key) -> ComputePipelineDescriptor {
        let mut shader_defs = vec![
            ShaderDefVal::Int(
                "SLICE_COUNT".to_string(),
                key.ssgi_settings.slice_count as i32,
            ),
            ShaderDefVal::Int(
                "SAMPLES_PER_SLICE_SIDE".to_string(),
                key.ssgi_settings.samples_per_slice_side as i32,
            ),
            ShaderDefVal::UInt("METHOD".to_string(), 1),
            ShaderDefVal::UInt(
                "THICKNESS".to_string(),
                key.ssgi_settings.thickness.to_bits(),
            ),
        ];

        if key.temporal_jitter {
            shader_defs.push("TEMPORAL_JITTER".into());
        }

        ComputePipelineDescriptor {
            label: Some("ssgi_ssgi_pipeline".into()),
            layout: vec![
                self.ssgi_bind_group_layout.clone(),
                self.common_bind_group_layout.clone(),
                self.deferred_output_bind_group_layout.clone(),
            ],
            push_constant_ranges: vec![],
            shader: SSGI_SHADER_HANDLE,
            shader_defs,
            entry_point: "ssgi".into(),
        }
    }
}

fn extract_ssgi_settings(
    mut commands: Commands,
    cameras: Extract<
        Query<
            (Entity, &Camera, &ScreenSpaceGlobalIlluminationSettings),
            (With<Camera3d>, With<DepthPrepass>, With<NormalPrepass>),
        >,
    >,
    msaa: Extract<Res<Msaa>>,
) {
    for (entity, camera, ssgi_settings) in &cameras {
        if **msaa != Msaa::Off {
            error!(
                "SSGI is being used which requires Msaa::Off, but Msaa is currently set to Msaa::{:?}",
                **msaa
            );
            return;
        }

        if camera.is_active {
            commands.get_or_spawn(entity).insert(ssgi_settings.clone());
        }
    }
}

#[derive(Component)]
pub struct ScreenSpaceGlobalIlluminationTextures {
    preprocessed_depth_texture: CachedTexture,
    ssgi_noisy_texture: CachedTexture, // Pre-spatially denoised texture
    pub screen_space_global_illumination_texture: CachedTexture, // Spatially denoised texture
    depth_differences_texture: CachedTexture,
}

fn prepare_ssgi_textures(
    mut commands: Commands,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    views: Query<(Entity, &ExtractedCamera), With<ScreenSpaceGlobalIlluminationSettings>>,
) {
    for (entity, camera) in &views {
        let Some(physical_viewport_size) = camera.physical_viewport_size else {
            continue;
        };
        let size = Extent3d {
            width: physical_viewport_size.x,
            height: physical_viewport_size.y,
            depth_or_array_layers: 1,
        };

        let preprocessed_depth_texture = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("ssgi_preprocessed_depth_texture"),
                size,
                mip_level_count: 5,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::R16Float,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        let ssgi_noisy_texture = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("ssgi_noisy_texture"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        let ssgi_texture = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("ssgi_texture"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        let depth_differences_texture = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("ssgi_depth_differences_texture"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::R32Uint,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        commands
            .entity(entity)
            .insert(ScreenSpaceGlobalIlluminationTextures {
                preprocessed_depth_texture,
                ssgi_noisy_texture,
                screen_space_global_illumination_texture: ssgi_texture,
                depth_differences_texture,
            });
    }
}

#[derive(Component)]
struct SsgiPipelineId(CachedComputePipelineId);

fn prepare_ssgi_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedComputePipelines<SsgiPipelines>>,
    pipeline: Res<SsgiPipelines>,
    views: Query<(
        Entity,
        &ScreenSpaceGlobalIlluminationSettings,
        Has<TemporalJitter>,
    )>,
) {
    for (entity, ssgi_settings, temporal_jitter) in &views {
        let pipeline_id = pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            SsgiPipelineKey {
                ssgi_settings: ssgi_settings.clone(),
                temporal_jitter,
            },
        );

        commands.entity(entity).insert(SsgiPipelineId(pipeline_id));
    }
}

#[derive(Component)]
struct SsgiBindGroups {
    common_bind_group: BindGroup,
    preprocess_depth_bind_group: BindGroup,
    ssgi_bind_group: BindGroup,
    spatial_denoise_bind_group: BindGroup,
}

fn prepare_ssgi_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipelines: Res<SsgiPipelines>,
    view_uniforms: Res<ViewUniforms>,
    global_uniforms: Res<GlobalsBuffer>,
    views: Query<(
        Entity,
        &ScreenSpaceGlobalIlluminationTextures,
        &ViewPrepassTextures,
    )>,
) {
    let (Some(view_uniforms), Some(globals_uniforms)) = (
        view_uniforms.uniforms.binding(),
        global_uniforms.buffer.binding(),
    ) else {
        return;
    };

    for (entity, ssgi_textures, prepass_textures) in &views {
        let common_bind_group = render_device.create_bind_group(
            "ssgi_common_bind_group",
            &pipelines.common_bind_group_layout,
            &BindGroupEntries::sequential((
                &pipelines.point_clamp_sampler,
                &pipelines.linear_point_clamp_sampler,
                view_uniforms.clone(),
            )),
        );

        let create_depth_view = |mip_level| {
            ssgi_textures
                .preprocessed_depth_texture
                .texture
                .create_view(&TextureViewDescriptor {
                    label: Some("ssgi_preprocessed_depth_texture_mip_view"),
                    base_mip_level: mip_level,
                    format: Some(TextureFormat::R16Float),
                    dimension: Some(TextureViewDimension::D2),
                    mip_level_count: Some(1),
                    ..default()
                })
        };

        let preprocess_depth_bind_group = render_device.create_bind_group(
            "ssgi_preprocess_depth_bind_group",
            &pipelines.preprocess_depth_bind_group_layout,
            &BindGroupEntries::sequential((
                prepass_textures.depth_view().unwrap(),
                &create_depth_view(0),
                &create_depth_view(1),
                &create_depth_view(2),
                &create_depth_view(3),
                &create_depth_view(4),
            )),
        );

        let ssgi_bind_group = render_device.create_bind_group(
            "ssgi_ssgi_bind_group",
            &pipelines.ssgi_bind_group_layout,
            &BindGroupEntries::sequential((
                &ssgi_textures.preprocessed_depth_texture.default_view,
                prepass_textures.normal_view().unwrap(),
                &pipelines.hilbert_index_lut,
                &ssgi_textures.ssgi_noisy_texture.default_view,
                &ssgi_textures.depth_differences_texture.default_view,
                globals_uniforms.clone(),
            )),
        );

        let spatial_denoise_bind_group = render_device.create_bind_group(
            "ssgi_spatial_denoise_bind_group",
            &pipelines.spatial_denoise_bind_group_layout,
            &BindGroupEntries::sequential((
                &ssgi_textures.ssgi_noisy_texture.default_view,
                &ssgi_textures.depth_differences_texture.default_view,
                &ssgi_textures
                    .screen_space_global_illumination_texture
                    .default_view,
            )),
        );

        commands.entity(entity).insert(SsgiBindGroups {
            common_bind_group,
            preprocess_depth_bind_group,
            ssgi_bind_group: ssgi_bind_group,
            spatial_denoise_bind_group,
        });
    }
}

#[allow(clippy::needless_range_loop)]
fn generate_hilbert_index_lut() -> [[u16; 64]; 64] {
    let mut t = [[0; 64]; 64];

    for x in 0..64 {
        for y in 0..64 {
            t[x][y] = hilbert_index(x as u16, y as u16);
        }
    }

    t
}

// https://www.shadertoy.com/view/3tB3z3
const HILBERT_WIDTH: u16 = 64;
fn hilbert_index(mut x: u16, mut y: u16) -> u16 {
    let mut index = 0;

    let mut level: u16 = HILBERT_WIDTH / 2;
    while level > 0 {
        let region_x = (x & level > 0) as u16;
        let region_y = (y & level > 0) as u16;
        index += level * level * ((3 * region_x) ^ region_y);

        if region_y == 0 {
            if region_x == 1 {
                x = HILBERT_WIDTH - 1 - x;
                y = HILBERT_WIDTH - 1 - y;
            }

            mem::swap(&mut x, &mut y);
        }

        level /= 2;
    }

    index
}

/// Divide `numerator` by `denominator`, rounded up to the nearest multiple of `denominator`.
fn div_ceil(numerator: u32, denominator: u32) -> u32 {
    (numerator + denominator - 1) / denominator
}
