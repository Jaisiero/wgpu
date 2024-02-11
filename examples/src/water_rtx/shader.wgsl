struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

// meant to be called with 3 vertex indices: 0, 1, 2
// draws one large triangle over the clip space like this:
// (the asterisks represent the clip space bounds)
//-1,1           1,1
// ---------------------------------
// |              *              .
// |              *           .
// |              *        .
// |              *      .
// |              *    . 
// |              * .
// |***************
// |            . 1,-1 
// |          .
// |       .
// |     .
// |   .
// |.
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var result: VertexOutput;
    let x = i32(vertex_index) / 2;
    let y = i32(vertex_index) & 1;
    let tc = vec2<f32>(
        f32(x) * 2.0,
        f32(y) * 2.0
    );
    result.position = vec4<f32>(
        tc.x * 2.0 - 1.0,
        1.0 - tc.y * 2.0,
        0.0, 1.0
    );
    result.tex_coords = tc;
    return result;
}

struct Uniforms {
    view_inv: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
};

// Under water
const DARK_SAND: vec3f = vec3f(235.0, 175.0, 71.0) / 255.0f;
// Coast
const SAND: vec3f = vec3f(217.0, 191.0, 76.0) / 255.0f;
// Normal
const GRASS: vec3f = vec3f(122.0, 170.0, 19.0) / 255.0f;
// Mountain
const SNOW: vec3f = vec3f(175.0, 224.0, 237.0) / 255.0f;

const SUN_POS: vec3f = vec3f(1.0, 0.2, 0.0);
const SKY: vec3f = vec3f(0.3, 0.5, 1.0) / 1.5;
const SUN: vec3f = vec3f(1.0, 1.0, 1.0);

const MAX_BOUNCES: u32 = 5u;

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var tex: texture_2d<f32>;

@group(0) @binding(2)
var sam: sampler;

@group(1) @binding(0)
var acc_struct: acceleration_structure;

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {

    var color = vec4<f32>(1.0);

	let d = vertex.tex_coords * 2.0 - 1.0;

	var origin = (uniforms.view_inv * vec4<f32>(0.0,0.0,0.0,1.0)).xyz;
	let temp = uniforms.proj_inv * vec4<f32>(d.x, -d.y, 1.0, 1.0);
	var direction = (uniforms.view_inv * vec4<f32>(normalize(temp.xyz), 0.0)).xyz;
    //direction.y = -direction.y;
    var i = 0u;
    var color_brightness = 1.0;
    var rq: ray_query;
        rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, 0.1, 1000.0, origin, direction));
        rayQueryProceed(&rq);

        let intersection = rayQueryGetCommittedIntersection(&rq);
        if (intersection.kind != RAY_QUERY_INTERSECTION_NONE) {
            let hit_pos = at(intersection.t, origin, direction);
            color = color * vec4<f32>(get_color(hit_pos.y), 1.0) * color_brightness;
            origin = hit_pos;
            let new_dir = normalize(SUN_POS);
            let dotted = dot(new_dir, direction);
            direction = new_dir;

        } else {
            color = color * vec4f(sky(direction.xyz), 1.0);
        }
    var rq_shadow: ray_query;
    rayQueryInitialize(&rq_shadow, acc_struct, RayDesc(0u, 0xFFu, 0.1, 1000.0, origin, direction));
    rayQueryProceed(&rq_shadow);
    let intersection_shadow = rayQueryGetCommittedIntersection(&rq_shadow);
    if (intersection_shadow.kind != RAY_QUERY_INTERSECTION_NONE) {
        color = color * 0.75;
    } else {
        color = color * vec4f(SUN, 1.0);
    }

    return color; // vec4<f32>(vertex.tex_coords, 1.0, 1.0);
}

fn sky(dir: vec3f) -> vec3f {
    return mix(SKY, SUN, clamp(dot(normalize(SUN_POS), dir) - 0.15f, 0.0, 1.0));
}

fn at(t: f32, origin: vec3f, direction: vec3f) -> vec3f {
    return origin + (direction * t);
}

fn get_color(y:f32) -> vec3f {
    if (y <= 0.0) {
        return DARK_SAND;
    } else if (y <= 0.8) {
        return SAND;
    } else if (y <= 10.0) {
        return GRASS;
    } else {
        return SNOW;
    };
}