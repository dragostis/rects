const WORKGOUP_SIZE = 256u;

struct Quad {
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    depth: u32,
}

@group(0)
@binding(0)
var<storage> quads: array<Quad>;
@group(0)
@binding(1)
var<storage, read_write> depth_buffer: array<atomic<u32>>;

@compute
@workgroup_size(WORKGOUP_SIZE)
fn rasterize(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let q = quads[global_id.x];

    for (var x = q.x0; x < q.x1; x++) {
        for (var y = q.y0; y < q.y1; y++) {
            atomicMax(&depth_buffer[y * 4096 + x], q.depth);
        }
    }
}