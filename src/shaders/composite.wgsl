@group(0) @binding(0)
var assignment: texture_storage_2d<r32uint, read>;
@group(0) @binding(1)
var<storage, read> centroids: array<array<u64, 4>>;
@group(0) @binding(2)
var output_image: texture_storage_2d<bgra8unorm, write>;


@compute
@workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let load: u32 = textureLoad(assignment, global_id.xy).x;
    let centroid = centroids[load];
    let color = vec4<f32>(f32(centroid[0]) / 255.0,
                f32(centroid[1]) / 255.0,
                f32(centroid[2]) / 255.0,
                1.0);
    // let color = vec4<f32>(1.0, 1.0, 0.0, 1.0);
    textureStore(output_image, global_id.xy, color);
}
