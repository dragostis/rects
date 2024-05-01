const BLOCK_SIZE = 32u;
const BLOCK_SIZE_SQUARE = BLOCK_SIZE * BLOCK_SIZE;
const WORKGOUP_SIZE = 256u;
const COORD_BITS = countTrailingZeros(BLOCK_SIZE) + 1;

struct Rect {
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    depth: u32,
}

const RECT_ZERO = Rect(0, 0, 0, 0, 0);

struct NormRect {
    coords: u32,
    depth: u32,
}

fn newNormRect(
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    depth: u32,
) -> NormRect {
    var coords = 0u;

    coords = insertBits(coords, x0, COORD_BITS * 0, COORD_BITS);
    coords = insertBits(coords, y0, COORD_BITS * 1, COORD_BITS);
    coords = insertBits(coords, x1, COORD_BITS * 2, COORD_BITS);
    coords = insertBits(coords, y1, COORD_BITS * 3, COORD_BITS);

    return NormRect(coords, depth);
}

fn toNormRect(r: NormRect) -> Rect {
    return Rect(
        extractBits(r.coords, COORD_BITS * 0, COORD_BITS),
        extractBits(r.coords, COORD_BITS * 1, COORD_BITS),
        extractBits(r.coords, COORD_BITS * 2, COORD_BITS),
        extractBits(r.coords, COORD_BITS * 3, COORD_BITS),
        r.depth,
    );
}

@group(0)
@binding(0)
var<storage> rects: array<Rect>;
@group(0)
@binding(1)
var<storage, read_write> counts: array<atomic<u32>>;
@group(0)
@binding(2)
var<storage, read_write> norm_rects: array<NormRect>;
@group(0)
@binding(3)
var depth_texture: texture_storage_2d<rg32uint, write>;

@compute
@workgroup_size(WORKGOUP_SIZE)
fn count(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let r = rects[global_id.x];

    for (var x = r.x0 / BLOCK_SIZE; x < (r.x1 + BLOCK_SIZE - 1) / BLOCK_SIZE; x++) {
        for (var y = r.y0 / BLOCK_SIZE; y < (r.y1 + BLOCK_SIZE - 1) / BLOCK_SIZE; y++) {
            atomicAdd(&counts[y * (4096 / BLOCK_SIZE) + x], 1u);
        }
    }
}

var<workgroup> prefix: array<u32, WORKGOUP_SIZE>;

fn workgroupPrefixSum(val: u32, local_index: u32) -> u32 {
    var sum = 0u;
    var shift = 1u;

    let shifted = (local_index + shift) & (WORKGOUP_SIZE - 1);
    prefix[shifted] = select(val, 0u, shifted < shift);

    loop {
        workgroupBarrier();

        sum += prefix[local_index];

        if shift == WORKGOUP_SIZE { break; }

        workgroupBarrier();

        let shifted = (local_index + shift) & (WORKGOUP_SIZE - 1);
        prefix[shifted] = select(sum, 0u, shifted < shift);

        shift <<= 1u;
    }

    return sum;
}

var<workgroup> carry: u32;

@compute
@workgroup_size(WORKGOUP_SIZE)
fn prefixSum(
    @builtin(local_invocation_index) local_index: u32,
) {
    if local_index == 0 {
        carry = 0u;
    }

    prefix[local_index] = 0u;

    workgroupBarrier();

    for (var i = 0u; i < (arrayLength(&counts) + WORKGOUP_SIZE - 1) / WORKGOUP_SIZE; i++) {
        let val = counts[i * WORKGOUP_SIZE + local_index];

        let sum = workgroupPrefixSum(val, local_index);

        counts[i * WORKGOUP_SIZE + local_index] = sum + carry;

        if local_index == WORKGOUP_SIZE - 1 {
            carry += val + sum;
        }

        workgroupBarrier();
    }
}

@compute
@workgroup_size(WORKGOUP_SIZE)
fn scatter(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let r = rects[global_id.x];

    for (var x = r.x0 / BLOCK_SIZE; x < (r.x1 + BLOCK_SIZE - 1) / BLOCK_SIZE; x++) {
        for (var y = r.y0 / BLOCK_SIZE; y < (r.y1 + BLOCK_SIZE - 1) / BLOCK_SIZE; y++) {
            let i = atomicAdd(&counts[y * (4096 / BLOCK_SIZE) + x], 1u);

            let x0 = u32(max(i32(r.x0) - i32(x * BLOCK_SIZE), 0));
            let y0 = u32(max(i32(r.y0) - i32(y * BLOCK_SIZE), 0));
            let x1 = min(r.x1 - x * BLOCK_SIZE, BLOCK_SIZE);
            let y1 = min(r.y1 - y * BLOCK_SIZE, BLOCK_SIZE);

            norm_rects[i] = newNormRect(x0, y0, x1, y1, r.depth);
        }
    }
}

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

    var start_index = 0u;
    if workgroup_index > 0 {
        start_index = counts[workgroup_index - 1];
    }
    let end_index = counts[workgroup_index];
    let len = end_index - start_index;

    for (var j = 0u; j < BLOCK_SIZE_SQUARE / WORKGOUP_SIZE; j++) {
        let i = j * WORKGOUP_SIZE + local_index;
        
        let x = block_x0 + i % BLOCK_SIZE;
        let y = block_y0 + i / BLOCK_SIZE;

        atomicStore(&cells[y * BLOCK_SIZE + x].depth, 0u);
        cells[y * BLOCK_SIZE + x].index = 0xFFFFFFFFu;
    }

    workgroupBarrier();

    for (var j = 0u; j < (len + WORKGOUP_SIZE - 1) / WORKGOUP_SIZE; j++) {
        let i = start_index + j * WORKGOUP_SIZE + local_index;

        var r = RECT_ZERO;
        if i < end_index {
            r = toNormRect(norm_rects[i]);
        }

        for (var x = r.x0; x < r.x1; x++) {
            for (var y = r.y0; y < r.y1; y++) {
                atomicMax(&cells[y * BLOCK_SIZE + x].depth, r.depth);
            }
        }

        workgroupBarrier();

        for (var x = r.x0; x < r.x1; x++) {
            for (var y = r.y0; y < r.y1; y++) {
                if atomicLoad(&cells[y * BLOCK_SIZE + x].depth) == r.depth {
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