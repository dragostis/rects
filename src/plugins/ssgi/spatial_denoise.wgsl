// 3x3 bilaterial filter (edge-preserving blur)
// https://people.csail.mit.edu/sparis/bf_course/course_notes.pdf

// Note: Does not use the Gaussian kernel part of a typical bilateral blur
// From the paper: "use the information gathered on a neighborhood of 4 × 4 using a bilateral filter for
// reconstruction, using _uniform_ convolution weights"

// Note: The paper does a 4x4 (not quite centered) filter, offset by +/- 1 pixel every other frame
// XeGTAO does a 3x3 filter, on two pixels at a time per compute thread, applied twice
// We do a 3x3 filter, on 1 pixel per compute thread, applied once

#import bevy_render::view::View

@group(0) @binding(0) var ambient_occlusion_noisy: texture_2d<f32>;
@group(0) @binding(1) var depth_differences: texture_2d<u32>;
@group(0) @binding(2) var ambient_occlusion: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var point_clamp_sampler: sampler;
@group(1) @binding(1) var linear_point_clamp_sampler: sampler;
@group(1) @binding(2) var<uniform> view: View;

@compute
@workgroup_size(8, 8, 1)
fn spatial_denoise(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel_coordinates = vec2<i32>(global_id.xy);
    let uv = vec2<f32>(pixel_coordinates) / view.viewport.zw;

    let edges0 = textureGather(0, depth_differences, point_clamp_sampler, uv);
    let edges1 = textureGather(0, depth_differences, point_clamp_sampler, uv, vec2<i32>(2i, 0i));
    let edges2 = textureGather(0, depth_differences, point_clamp_sampler, uv, vec2<i32>(1i, 2i));

    let left_edges = unpack4x8unorm(edges0.x);
    let right_edges = unpack4x8unorm(edges1.x);
    let top_edges = unpack4x8unorm(edges0.z);
    let bottom_edges = unpack4x8unorm(edges2.w);
    var center_edges = unpack4x8unorm(edges0.y);
    center_edges *= vec4<f32>(left_edges.y, right_edges.x, top_edges.w, bottom_edges.z);

    let center_weight = 1.2;
    let left_weight = center_edges.x;
    let right_weight = center_edges.y;
    let top_weight = center_edges.z;
    let bottom_weight = center_edges.w;
    let top_left_weight = 0.425 * (top_weight * top_edges.x + left_weight * left_edges.z);
    let top_right_weight = 0.425 * (top_weight * top_edges.y + right_weight * right_edges.z);
    let bottom_left_weight = 0.425 * (bottom_weight * bottom_edges.x + left_weight * left_edges.w);
    let bottom_right_weight = 0.425 * (bottom_weight * bottom_edges.y + right_weight * right_edges.w);

    let center_color = textureSampleLevel(ambient_occlusion_noisy, point_clamp_sampler, uv, 0.0);
    let left_color = textureSampleLevel(ambient_occlusion_noisy, point_clamp_sampler, uv, 0.0, vec2<i32>(-1i, 0i));
    let right_color = textureSampleLevel(ambient_occlusion_noisy, point_clamp_sampler, uv, 0.0, vec2<i32>(1i, 0i));
    let top_color = textureSampleLevel(ambient_occlusion_noisy, point_clamp_sampler, uv, 0.0, vec2<i32>(0i, -1i));
    let bottom_color = textureSampleLevel(ambient_occlusion_noisy, point_clamp_sampler, uv, 0.0, vec2<i32>(0i, 1i));
    let top_left_color = textureSampleLevel(ambient_occlusion_noisy, point_clamp_sampler, uv, 0.0, vec2<i32>(-1i, -1i));
    let top_right_color = textureSampleLevel(ambient_occlusion_noisy, point_clamp_sampler, uv, 0.0, vec2<i32>(1i, -1i));
    let bottom_left_color = textureSampleLevel(ambient_occlusion_noisy, point_clamp_sampler, uv, 0.0, vec2<i32>(-1i, 1i));
    let bottom_right_color = textureSampleLevel(ambient_occlusion_noisy, point_clamp_sampler, uv, 0.0, vec2<i32>(1i, 1i));

    var sum = center_color;
    sum += left_color * left_weight;
    sum += right_color * right_weight;
    sum += top_color * top_weight;
    sum += bottom_color * bottom_weight;
    sum += top_left_color * top_left_weight;
    sum += top_right_color * top_right_weight;
    sum += bottom_left_color * bottom_left_weight;
    sum += bottom_right_color * bottom_right_weight;

    var sum_weight = center_weight;
    sum_weight += left_weight;
    sum_weight += right_weight;
    sum_weight += top_weight;
    sum_weight += bottom_weight;
    sum_weight += top_left_weight;
    sum_weight += top_right_weight;
    sum_weight += bottom_left_weight;
    sum_weight += bottom_right_weight;

    let denoised_color = sum / sum_weight;

    textureStore(ambient_occlusion, pixel_coordinates, denoised_color);
}
