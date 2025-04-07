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

override k: u32;

// todo: hard-coded k
var<workgroup> sums_wg: array<array<u64, 4>, k>;
var<workgroup> counts_wg: array<u32, k>;
@compute
@workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_index) local_index: u32) {
    let loadunorm: vec4<f32> = textureLoad(input_image, global_id.xy) * 255.0;
    let load: vec4<u32> = vec4<u32>(u32(loadunorm.r), u32(loadunorm.g), u32(loadunorm.b), u32(255));

    var min_idx = u32(0);
    var min_dist: u32 = 0xffffffff;
    for (var i = u32(0); i < k; i++) {
       let point_dist = (max(load[0], u32(centroids[i][0])) - min(load[0], u32(centroids[i][0])))
       + (max(load[1], u32(centroids[i][1])) - min(load[1], u32(centroids[i][1])))
       + (max(load[2], u32(centroids[i][2])) - min(load[2], u32(centroids[i][2])));
       if (point_dist < min_dist) {
        min_idx = i;
        min_dist = point_dist;
       }
    }
    textureStore(assignment, global_id.xy, vec4<u32>(min_idx));

    sums_wg[min_idx][0]+=u64(load[0]);
    sums_wg[min_idx][1]+=u64(load[1]);
    sums_wg[min_idx][2]+=u64(load[2]);
    counts_wg[min_idx]+=1;
    subgroupBarrier();
    if (subgroupMin(local_index) == local_index) {
        for (var i = u32(0); i < k; i++) {
            atomicAdd(&centroids_out[i][0], sums_wg[i][0]);
            atomicAdd(&centroids_out[i][1], sums_wg[i][1]);
            atomicAdd(&centroids_out[i][2], sums_wg[i][2]);
            atomicAdd(&counts[min_idx], counts_wg[i]);
        }
    }
}


// second phase of k-means where centroids are reassigned
// probably pretty low occupancy since only one thread is dispatched per k
@compute
@workgroup_size(1, 1, 1)
fn phase2(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let sum = array<u64, 4>(centroids_out[global_id.x][0], centroids_out[global_id.x][1], centroids_out[global_id.x][2], centroids_out[global_id.x][3]);
    let count = atomicLoad(&counts[global_id.x]);
    centroids_out[global_id.x][0] = sum[0] / u64(max(count, u32(1)));
    centroids_out[global_id.x][1] = sum[1] / u64(max(count, u32(1)));
    centroids_out[global_id.x][2] = sum[2] / u64(max(count, u32(1)));
    centroids[global_id.x] = array<u64, 4>(0, 0, 0, 0);
    atomicAnd(&counts[global_id.x], u32(0));
}
