// Ground Truth-based Ambient Occlusion (GTAO)
// Paper: https://www.activision.com/cdn/research/Practical_Real_Time_Strategies_for_Accurate_Indirect_Occlusion_NEW%20VERSION_COLOR.pdf
// Presentation: https://blog.selfshadow.com/publications/s2016-shading-course/activision/s2016_pbs_activision_occlusion.pdf

// Source code heavily based on XeGTAO v1.30 from Intel
// https://github.com/GameTechDev/XeGTAO/blob/0d177ce06bfa642f64d8af4de1197ad1bcb862d4/Source/Rendering/Shaders/XeGTAO.hlsli

// Visibility Bitmask Ambient Occlusion (VBAO)
// Paper: https://ar5iv.labs.arxiv.org/html/2301.11376

// Source code based on the existing XeGTAO implementation and
// https://cdrinmatane.github.io/posts/ssaovb-code/

#import bevy_pbr::gtao_utils::fast_acos

#import bevy_render::{
    view::View,
    globals::Globals,
    maths::{PI, HALF_PI},
}

@group(0) @binding(0) var preprocessed_depth: texture_2d<f32>;
@group(0) @binding(1) var normals: texture_2d<f32>;
@group(0) @binding(2) var hilbert_index_lut: texture_2d<u32>;
@group(0) @binding(3) var ambient_occlusion: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var depth_differences: texture_storage_2d<r32uint, write>;
@group(0) @binding(5) var<uniform> globals: Globals;
@group(1) @binding(0) var point_clamp_sampler: sampler;
@group(1) @binding(1) var linear_point_clamp_sampler: sampler;
@group(1) @binding(2) var<uniform> view: View;

fn load_noise(pixel_coordinates: vec2<i32>) -> vec2<f32> {
    var index = textureLoad(hilbert_index_lut, pixel_coordinates % 64, 0).r;

#ifdef TEMPORAL_JITTER
    index += 288u * (globals.frame_count % 64u);
#endif

    // R2 sequence - http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences
    return fract(0.5 + f32(index) * vec2<f32>(0.75487766624669276005, 0.5698402909980532659114));
}

// Calculate differences in depth between neighbor pixels (later used by the spatial denoiser pass to preserve object edges)
fn calculate_neighboring_depth_differences(pixel_coordinates: vec2<i32>) -> f32 {
    // Sample the pixel's depth and 4 depths around it
    let uv = vec2<f32>(pixel_coordinates) / view.viewport.zw;
    let depths_upper_left = textureGather(0, preprocessed_depth, point_clamp_sampler, uv);
    let depths_bottom_right = textureGather(0, preprocessed_depth, point_clamp_sampler, uv, vec2<i32>(1i, 1i));
    let depth_center = depths_upper_left.y;
    let depth_left = depths_upper_left.x;
    let depth_top = depths_upper_left.z;
    let depth_bottom = depths_bottom_right.x;
    let depth_right = depths_bottom_right.z;

    // Calculate the depth differences (large differences represent object edges)
    var edge_info = vec4<f32>(depth_left, depth_right, depth_top, depth_bottom) - depth_center;
    let slope_left_right = (edge_info.y - edge_info.x) * 0.5;
    let slope_top_bottom = (edge_info.w - edge_info.z) * 0.5;
    let edge_info_slope_adjusted = edge_info + vec4<f32>(slope_left_right, -slope_left_right, slope_top_bottom, -slope_top_bottom);
    edge_info = min(abs(edge_info), abs(edge_info_slope_adjusted));
    let bias = 0.25; // Using the bias and then saturating nudges the values a bit
    let scale = depth_center * 0.011; // Weight the edges by their distance from the camera
    edge_info = saturate((1.0 + bias) - edge_info / scale); // Apply the bias and scale, and invert edge_info so that small values become large, and vice versa

    // Pack the edge info into the texture
    let edge_info_packed = vec4<u32>(pack4x8unorm(edge_info), 0u, 0u, 0u);
    textureStore(depth_differences, pixel_coordinates, edge_info_packed);

    return depth_center;
}

fn load_normal_view_space(uv: vec2<f32>) -> vec3<f32> {
    var world_normal = textureSampleLevel(normals, point_clamp_sampler, uv, 0.0).xyz;
    world_normal = (world_normal * 2.0) - 1.0;
    let inverse_view = mat3x3<f32>(
        view.inverse_view[0].xyz,
        view.inverse_view[1].xyz,
        view.inverse_view[2].xyz,
    );
    return inverse_view * world_normal;
}

fn reconstruct_view_space_position(depth: f32, uv: vec2<f32>) -> vec3<f32> {
    let clip_xy = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - 2.0 * uv.y);
    let t = view.inverse_projection * vec4<f32>(clip_xy, depth, 1.0);
    let view_xyz = t.xyz / t.w;
    return view_xyz;
}

fn load_and_reconstruct_view_space_position(uv: vec2<f32>, sample_mip_level: f32) -> vec3<f32> {
    let depth = textureSampleLevel(preprocessed_depth, linear_point_clamp_sampler, uv, sample_mip_level).r;
    return reconstruct_view_space_position(depth, uv);
}

fn updateSectors(
    min_horizon: f32,
    max_horizon: f32,
    samples_per_slice: f32,
    bitmask: u32,
) -> u32 {
    let start_horizon = u32(min_horizon * samples_per_slice);
    let angle_horizon = u32(ceil((max_horizon - min_horizon) * samples_per_slice));

    return insertBits(bitmask, 0xFFFFFFFFu, start_horizon, angle_horizon);
}

fn processSample(
    delta_position: vec3<f32>,
    view_vec: vec3<f32>,
    sampling_direction: f32,
    n: f32,
    samples_per_slice: f32,
    bitmask: ptr<function, u32>,
) {
    let delta_position_back_face = delta_position - view_vec * bitcast<f32>(#THICKNESS);

    var front_back_horizon = vec2(
        fast_acos(dot(normalize(delta_position), view_vec)),
        fast_acos(dot(normalize(delta_position_back_face), view_vec)),
    );

    front_back_horizon = saturate(((sampling_direction * -front_back_horizon) - n + HALF_PI) / PI);
    front_back_horizon = select(front_back_horizon.xy, front_back_horizon.yx, sampling_direction >= 0.0);

    *bitmask = updateSectors(front_back_horizon.x, front_back_horizon.y, samples_per_slice, *bitmask);
}

@compute
@workgroup_size(16, 8)
fn ssao(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let slice_count = f32(#SLICE_COUNT);
    let samples_per_slice_side = f32(#SAMPLES_PER_SLICE_SIDE);
    let effect_radius = 0.5 * 1.457;
    let falloff_range = 0.615 * effect_radius;
    let falloff_from = effect_radius * (1.0 - 0.615);
    let falloff_mul = -1.0 / falloff_range;
    let falloff_add = falloff_from / falloff_range + 1.0;

    let pixel_coordinates = vec2<i32>(global_id.xy);
    let uv = (vec2<f32>(pixel_coordinates) + 0.5) / view.viewport.zw;

    var pixel_depth = calculate_neighboring_depth_differences(pixel_coordinates);
    pixel_depth += 0.00001; // Avoid depth precision issues

    let pixel_position = reconstruct_view_space_position(pixel_depth, uv);
    let pixel_normal = load_normal_view_space(uv);
    let view_vec = normalize(-pixel_position);

    let noise = load_noise(pixel_coordinates);
    let sample_scale = (-0.5 * effect_radius * view.projection[0][0]) / pixel_position.z;

    var visibility = 0.0;
#if METHOD == 1
    var occluded_sample_count = 0u;
#endif
    for (var slice_t = 0.0; slice_t < slice_count; slice_t += 1.0) {
        let slice = slice_t + noise.x;
        let phi = (PI / slice_count) * slice;
        let omega = vec2<f32>(cos(phi), sin(phi));

        let direction = vec3<f32>(omega.xy, 0.0);
        let orthographic_direction = direction - (dot(direction, view_vec) * view_vec);
        let axis = cross(direction, view_vec);
        let projected_normal = pixel_normal - axis * dot(pixel_normal, axis);
        let projected_normal_length = length(projected_normal);

        let sign_norm = sign(dot(orthographic_direction, projected_normal));
        let cos_norm = saturate(dot(projected_normal, view_vec) / projected_normal_length);
        let n = sign_norm * fast_acos(cos_norm);

#if METHOD == 0
        let min_cos_horizon_1 = cos(n + HALF_PI);
        let min_cos_horizon_2 = cos(n - HALF_PI);
        var cos_horizon_1 = min_cos_horizon_1;
        var cos_horizon_2 = min_cos_horizon_2;
#else
        var bitmask = 0u;
#endif

        let sample_mul = vec2<f32>(omega.x, -omega.y) * sample_scale;
        for (var sample_t = 0.0; sample_t < samples_per_slice_side; sample_t += 1.0) {
            var sample_noise = (slice_t + sample_t * samples_per_slice_side) * 0.6180339887498948482;
            sample_noise = fract(noise.y + sample_noise);

            var s = (sample_t + sample_noise) / samples_per_slice_side;
            s *= s; // https://github.com/GameTechDev/XeGTAO#sample-distribution
            let sample = s * sample_mul;

            // * view.viewport.zw gets us from [0, 1] to [0, viewport_size], which is needed for this to get the correct mip levels
            let sample_mip_level = clamp(log2(length(sample * view.viewport.zw)) - 3.3, 0.0, 5.0); // https://github.com/GameTechDev/XeGTAO#memory-bandwidth-bottleneck
            let sample_position_1 = load_and_reconstruct_view_space_position(uv + sample, sample_mip_level);
            let sample_position_2 = load_and_reconstruct_view_space_position(uv - sample, sample_mip_level);

            let sample_difference_1 = sample_position_1 - pixel_position;
            let sample_difference_2 = sample_position_2 - pixel_position;

#if METHOD == 0
            let sample_distance_1 = length(sample_difference_1);
            let sample_distance_2 = length(sample_difference_2);
            var sample_cos_horizon_1 = dot(sample_difference_1 / sample_distance_1, view_vec);
            var sample_cos_horizon_2 = dot(sample_difference_2 / sample_distance_2, view_vec);

            let weight_1 = saturate(sample_distance_1 * falloff_mul + falloff_add);
            let weight_2 = saturate(sample_distance_2 * falloff_mul + falloff_add);
            sample_cos_horizon_1 = mix(min_cos_horizon_1, sample_cos_horizon_1, weight_1);
            sample_cos_horizon_2 = mix(min_cos_horizon_2, sample_cos_horizon_2, weight_2);

            cos_horizon_1 = max(cos_horizon_1, sample_cos_horizon_1);
            cos_horizon_2 = max(cos_horizon_2, sample_cos_horizon_2);
#else
            processSample(sample_difference_1, view_vec, -1.0, n, samples_per_slice_side * 2.0, &bitmask);
            processSample(sample_difference_2, view_vec, 1.0, n, samples_per_slice_side * 2.0, &bitmask);
#endif
        }

#if METHOD == 0
        let horizon_1 = fast_acos(cos_horizon_1);
        let horizon_2 = -fast_acos(cos_horizon_2);
        let v1 = (cos_norm + 2.0 * horizon_1 * sin(n) - cos(2.0 * horizon_1 - n)) / 4.0;
        let v2 = (cos_norm + 2.0 * horizon_2 * sin(n) - cos(2.0 * horizon_2 - n)) / 4.0;
        visibility += projected_normal_length * (v1 + v2);
#else
        occluded_sample_count += countOneBits(bitmask);
#endif
    }

#if METHOD == 0
    visibility /= slice_count;
#else
    visibility = 1.0 - f32(occluded_sample_count) / (slice_count * 2.0 * samples_per_slice_side);
#endif

    visibility = clamp(visibility, 0.03, 1.0);

    textureStore(ambient_occlusion, pixel_coordinates, vec4<f32>(vec3(visibility), 0.0));
}