struct Uniforms {
    view_inv: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var output: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(2)
var acc_struct: acceleration_structure;

const R = 1.0;
const POS = vec3<f32>(0.0, 0.0, 0.0);

// see https://stackoverflow.com/questions/1986378/how-to-set-up-quadratic-equation-for-a-ray-sphere-intersection
@intersection
fn intersect_sphere(@builtin(object_ray_origin) origin: vec3<f32>, @builtin(object_ray_direction) dir: vec3<f32>) {
    let A = len_sqrd(dir);
    let half_B = dot(origin - POS, dir);
    let C = len_sqrd(origin - POS) - (R * R);
    let discriminant = (half_B*half_B) - (A * C);
    if (discriminant >= 0.0) {
        let sqrt_discriminant = sqrt(discriminant);
        let t1 = (-half_B - sqrt_discriminant) / A;
        let u = ReportIntersection(t1, 2u, 0u);
        let t2 = (-half_B + sqrt_discriminant) / A;
        if (sqrt_discriminant != 0.0) {
            let u2 = ReportIntersection(t2, 3u, 0u);
        }
    }
    ReportIntersection(1.0, 4u, 0u);
}

@ray_gen
fn ray_gen(@builtin(launch_id) global_id: vec3<u32>, @builtin(launch_size) target_size: vec3<u32>) {
    let pixel_center = vec2<f32>(global_id.xy) + vec2<f32>(0.5);
    let in_uv = pixel_center / vec2<f32>(target_size.xy);
    let d = in_uv * 2.0 - 1.0;

    let origin = (uniforms.view_inv * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
    let temp = uniforms.proj_inv * vec4<f32>(d.x, d.y, 1.0, 1.0);
    let direction = (uniforms.view_inv * vec4<f32>(normalize(temp.xyz), 0.0)).xyz;

    var colour = vec3<f32>(in_uv, 0.0);
    traceRay(acc_struct, RayDesc(0u, 0xFFu, 0.1, 200.0, origin, direction), &colour);

    textureStore(output, global_id.xy, vec4<f32>(colour, 1.0));
}

const HIT_COLOUR = vec3<f32>(0.5, 1.0, 0.5);

@ray_closest
fn closest_hit(@builtin(payload) colour: ptr<ray_tracing, vec3<f32>>, @builtin(intersection) intersection: u32, @builtin(ray_t) t: f32, @builtin(ray_origin) origin: vec3<f32>, @builtin(ray_direction) dir: vec3<f32>) {
    *colour = normalize(origin + (t * dir) - POS);
    //*colour = vec3<f32>(f32(intersection));
    //*colour = vec3<f32>(0.5);
}

@ray_any
fn any_hit(@builtin(payload) colour: ptr<ray_tracing, vec3<f32>>, @builtin(intersection) intersection: u32, @builtin(ray_t) t: f32,  @builtin(ray_origin) origin: vec3<f32>, @builtin(ray_direction) dir: vec3<f32>) {}

@ray_miss
fn miss(@builtin(payload) colour: ptr<ray_tracing, vec3<f32>>, @builtin(ray_origin) origin: vec3<f32>, @builtin(ray_direction) dir: vec3<f32>) {}

fn len_sqrd(vec:vec3<f32>) -> f32 {
    return (vec.x * vec.x) + (vec.y * vec.y) + (vec.z * vec.z);
}