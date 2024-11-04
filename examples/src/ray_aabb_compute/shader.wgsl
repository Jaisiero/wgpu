/*
let RAY_FLAG_NONE = 0x00u;
let RAY_FLAG_OPAQUE = 0x01u;
let RAY_FLAG_NO_OPAQUE = 0x02u;
let RAY_FLAG_TERMINATE_ON_FIRST_HIT = 0x04u;
let RAY_FLAG_SKIP_CLOSEST_HIT_SHADER = 0x08u;
let RAY_FLAG_CULL_BACK_FACING = 0x10u;
let RAY_FLAG_CULL_FRONT_FACING = 0x20u;
let RAY_FLAG_CULL_OPAQUE = 0x40u;
let RAY_FLAG_CULL_NO_OPAQUE = 0x80u;
let RAY_FLAG_SKIP_TRIANGLES = 0x100u;
let RAY_FLAG_SKIP_AABBS = 0x200u;

let RAY_QUERY_INTERSECTION_NONE = 0u;
let RAY_QUERY_INTERSECTION_TRIANGLE = 1u;
let RAY_QUERY_INTERSECTION_GENERATED = 2u;
let RAY_QUERY_INTERSECTION_AABB = 4u;

struct RayDesc {
    flags: u32,
    cull_mask: u32,
    t_min: f32,
    t_max: f32,
    origin: vec3<f32>,
    dir: vec3<f32>,
}

struct RayIntersection {
    kind: u32,
    t: f32,
    instance_custom_index: u32,
    instance_id: u32,
    sbt_record_offset: u32,
    geometry_index: u32,
    primitive_index: u32,
    barycentrics: vec2<f32>,
    front_face: bool,
    object_to_world: mat4x3<f32>,
    world_to_object: mat4x3<f32>,
}
*/

struct Uniforms {
    view_inv: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
};

@group(0) @binding(0)
var output: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(1)
var<uniform> uniforms: Uniforms;

@group(0) @binding(2)
var acc_struct: acceleration_structure;

fn query_loop(pos: vec3<f32>, dir: vec3<f32>, acs: acceleration_structure) -> RayIntersection {
    var rq: ray_query;
    rayQueryInitialize(&rq, acs, RayDesc(RAY_FLAG_TERMINATE_ON_FIRST_HIT, 0xFFu, 0.1, 100.0, pos, dir));

    // while(rayQueryProceed(&rq)) {
    //     let intersection = rayQueryGetIntersection(&rq);
    //     if (intersection.kind == RAY_QUERY_INTERSECTION_AABB) {
    //         rayQueryGenerateIntersection(&rq, 0.0);
    //     }
    // }

    // return rayQueryGetCommittedIntersection(&rq);

    while (rayQueryProceed(&rq)) {}

    return rayQueryGetCommittedIntersection(&rq);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let target_size = textureDimensions(output);
    var color =  vec4<f32>(vec2<f32>(global_id.xy) / vec2<f32>(target_size), 0.0, 1.0);


	let pixel_center = vec2<f32>(global_id.xy) + vec2<f32>(0.5);
	let in_uv = pixel_center/vec2<f32>(target_size.xy);
	let d = in_uv * 2.0 - 1.0;

	let origin = (uniforms.view_inv * vec4<f32>(0.0,0.0,0.0,1.0)).xyz;
	let temp = uniforms.proj_inv * vec4<f32>(d.x, d.y, 1.0, 1.0);
	let direction = (uniforms.view_inv * vec4<f32>(normalize(temp.xyz), 0.0)).xyz;

    let intersection = query_loop(origin, direction, acc_struct);

    if (intersection.kind != RAY_QUERY_INTERSECTION_NONE) {
        color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    }

    textureStore(output, global_id.xy, color);
}
