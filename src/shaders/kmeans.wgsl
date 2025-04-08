@group(0) @binding(0)
var input_image: texture_storage_2d<rgba8unorm, read>;
@group(0) @binding(1)
var<storage, read_write> centroids: array<array<u64, 4>>;
@group(0) @binding(2)
var<storage, read_write> centroids_out: array<array<atomic<u64>, 4>>;
@group(0) @binding(3)
var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(4)
var assignment: texture_storage_2d<r32uint, write>;
@group(0) @binding(5)
var<storage, read_write> converged: array<atomic<u32>>;

override k: u32;

var<workgroup> sums_wg: array<array<atomic<u64>, 4>, k>;
var<workgroup> counts_wg: array<atomic<u32>, k>;

fn arr64_to_vecu32(arr: array<u64, 4>) -> vec4<u32> {
    return vec4(u32(arr[0]), u32(arr[1]), u32(arr[2]), u32(arr[3]));
}

fn abs_diff(a: u32, b: u32) -> u32 {
    return max(a, b) - min(a, b);
}
fn int_distance(v1: vec3<u32>, v2: vec3<u32>) -> u32 {
    return abs_diff(v1.x, v2.x) * abs_diff(v1.x, v2.x) + abs_diff(v1.y, v2.y) * abs_diff(v1.y, v2.y) + abs_diff(v1.z, v2.z) * abs_diff(v1.z, v2.z);
}

@compute
@workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_index) local_index: u32) {
    let dims = textureDimensions(input_image);
    if (global_id.x >= dims.x || global_id.y >= dims.y) {
        return;
    }
    let loadunorm: vec4<f32> = textureLoad(input_image, global_id.xy) * 255.0;
    let load: vec4<u32> = vec4<u32>(u32(loadunorm.r), u32(loadunorm.g), u32(loadunorm.b), u32(255));

    var min_idx = u32(0);
    var min_dist: u32 = 0xffffffff;
    for (var i = u32(0); i < k; i++) {
       let point_dist = int_distance(load.xyz, arr64_to_vecu32(centroids[i]).xyz);
       if (point_dist < min_dist) {
        min_idx = i;
        min_dist = point_dist;
       }
    }
    textureStore(assignment, global_id.xy, vec4<u32>(min_idx));

    atomicAdd(&sums_wg[min_idx][0], u64(load[0]));
    atomicAdd(&sums_wg[min_idx][1], u64(load[1]));
    atomicAdd(&sums_wg[min_idx][2], u64(load[2]));
    atomicAdd(&counts_wg[min_idx], u32(1));
    // subgroupBarrier();
    if (subgroupMin(local_index) == local_index) {
        for (var i = u32(0); i < k; i++) {
            atomicAdd(&centroids_out[i][0], sums_wg[i][0]);
            atomicAdd(&centroids_out[i][1], sums_wg[i][1]);
            atomicAdd(&centroids_out[i][2], sums_wg[i][2]);
            atomicAdd(&counts[i], counts_wg[i]);
        }
    }
}


// second phase of k-means where centroids are reassigned
// probably pretty low occupancy since only one thread is dispatched per k
@compute
@workgroup_size(1, 1, 1)
fn phase2(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let sum = array<u64, 4>(centroids_out[global_id.x][0], centroids_out[global_id.x][1], centroids_out[global_id.x][2], centroids_out[global_id.x][3]);
    let count = counts[global_id.x];
    centroids_out[global_id.x][0] = sum[0] / u64(max(count, u32(1)));
    centroids_out[global_id.x][1] = sum[1] / u64(max(count, u32(1)));
    centroids_out[global_id.x][2] = sum[2] / u64(max(count, u32(1)));

    let neq = (centroids[global_id.x][0] != centroids_out[global_id.x][0] || centroids[global_id.x][1] != centroids_out[global_id.x][1] || centroids[global_id.x][2] != centroids_out[global_id.x][2]);
    atomicOr(&converged[0], u32(neq));
    atomicAnd(&counts[global_id.x], u32(0));
    centroids[global_id.x] = array<u64, 4>(0, 0, 0, 0);
}
