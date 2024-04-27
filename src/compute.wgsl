const BLOCK_SIZE = 16u;
const BLOCK_SIZE_SQUARE = BLOCK_SIZE * BLOCK_SIZE;
const WORKGOUP_SIZE = 256u;

struct Quad {
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    depth: u32,
}

const QUAD_ZERO = Quad(0, 0, 0, 0, 0);

@group(0)
@binding(0)
var<storage> indices: array<u32>;
@group(0)
@binding(1)
var<storage> blocks: array<u32>;
@group(0)
@binding(2)
var<storage> quads: array<Quad>;
@group(0)
@binding(3)
var depth_texture: texture_storage_2d<rg32uint, write>;

struct Cell {
    depth: atomic<u32>,
    index: u32,
}

var<workgroup> cells: array<Cell, BLOCK_SIZE_SQUARE>;

@compute
@workgroup_size(WORKGOUP_SIZE)
fn rasterize(
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let workgroup_index = workgroup_id.y * num_workgroups.x + workgroup_id.x;

    let block_x0 = workgroup_id.x * BLOCK_SIZE;
    let block_y0 = workgroup_id.y * BLOCK_SIZE;
    let block_x1 = (workgroup_id.x + 1) * BLOCK_SIZE;
    let block_y1 = (workgroup_id.y + 1) * BLOCK_SIZE;

    let start_index = indices[workgroup_index];
    let end_index = indices[workgroup_index + 1];
    let len = end_index - start_index;

    for (var j = 0u; j < (len + WORKGOUP_SIZE - 1) / WORKGOUP_SIZE; j++) {
        let i = start_index + j * WORKGOUP_SIZE + local_index;

        var q = QUAD_ZERO;
        if i < end_index {
            q = quads[blocks[i]];

            q.x0 = max(q.x0, block_x0);
            q.y0 = max(q.y0, block_y0);
            q.x1 = min(q.x1, block_x1);
            q.y1 = min(q.y1, block_y1);
        }

        for (var x = q.x0 - block_x0; x < q.x1 - block_x0; x++) {
            for (var y = q.y0 - block_y0; y < q.y1 - block_y0; y++) {
                atomicMax(&cells[y * BLOCK_SIZE + x].depth, q.depth);
            }
        }

        workgroupBarrier();

        for (var x = q.x0 - block_x0; x < q.x1 - block_x0; x++) {
            for (var y = q.y0 - block_y0; y < q.y1 - block_y0; y++) {
                if atomicLoad(&cells[y * BLOCK_SIZE + x].depth) == q.depth {
                    cells[y * BLOCK_SIZE + x].index = i;
                }
            }
        }
    }

    workgroupBarrier();

    for (var j = 0u; j < BLOCK_SIZE_SQUARE / WORKGOUP_SIZE; j++) {
        let i = j * WORKGOUP_SIZE + local_index;
        
        let x = block_x0 + i % BLOCK_SIZE;
        let y = block_y0 + i / BLOCK_SIZE;

        textureStore(
            depth_texture,
            vec2(x, y), 
            vec4(atomicLoad(&cells[i].depth), cells[i].index, 0u, 0u),
        );
    }
}