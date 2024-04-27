@group(0)
@binding(0)
var<uniform> size: vec2<f32>;

fn colorFromDepth(depth: f32) -> vec4<f32> {
    let seed = bitcast<u32>(depth) * 3;

    let r = (seed * 279470273) % 255;
    let g = ((seed + 1) * 279470273) % 255;
    let b = ((seed + 2) * 279470273) % 255;

    return vec4(f32(r) / 255.0, f32(g) / 255.0, f32(b) / 255.0, 1.0);
}

@vertex
fn vs_main(
    @location(0) position: vec4<f32>,
) -> @builtin(position) vec4<f32> {
    var new_position = (position.xy / size) * 2.0 - vec2(1.0);
    new_position.y *= -1.0;

    return vec4(new_position, position.z, 1.0);
}

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return colorFromDepth(position.z);
}