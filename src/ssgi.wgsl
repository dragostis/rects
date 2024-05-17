@vertex
fn displayVertex(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2(
        f32((i32(vertex_index) << 1u) & 2),
        f32(i32(vertex_index) & 2),
    );

    let pos = 2.0 * uv - vec2(1.0, 1.0);

    return vec4(pos.x, pos.y, 0.0, 1.0);
}

@group(0)
@binding(0)
var image: texture_storage_2d<rgba16float, read_write>;

@fragment
fn displayFragment(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(image, vec2<u32>(pos.xy));
}