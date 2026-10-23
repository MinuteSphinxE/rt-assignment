// WGSL path tracer using hardware ray queries (RT cores) instead of analytic
// intersections: the scene lives in a top level acceleration structure that the
// GPU traverses with `rayQueryInitialize` / `rayQueryProceed`.
enable wgpu_ray_query;

struct Uniforms {
    cam_pos: vec4<f32>,
    cam_fwd: vec4<f32>,      // w = tan(fov_y / 2)
    cam_right: vec4<f32>,
    cam_up: vec4<f32>,
    resolution: vec4<f32>,   // xy = size, z = sample index, w = max bounces
    params: vec4<f32>,       // x = roughness scale, y = exposure, z = seed, w = unused
    sky: vec4<f32>,          // rgb = sky colour
    light_origin: vec4<f32>,
    light_u: vec4<f32>,
    light_v: vec4<f32>,
    light_emission: vec4<f32>,
};

struct Material {
    albedo: vec4<f32>,
    emissive: vec4<f32>,  // rgb, w = kind (0 surface, 1 light, 2 checker floor)
    params: vec4<f32>,    // x = roughness
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var tlas: acceleration_structure;
@group(0) @binding(2) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> normals: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> indices: array<u32>;
@group(0) @binding(5) var<storage, read> tri_material: array<u32>;
@group(0) @binding(6) var<storage, read> materials: array<Material>;
@group(0) @binding(7) var<storage, read_write> accum: array<vec4<f32>>;

const PI: f32 = 3.14159265359;

// ---------------------------------------------------------------- random ----
// PCG hash: cheap, de-correlated between neighbouring pixels, which matters
// because every pixel accumulates its own independent samples.
fn pcg(state_in: u32) -> u32 {
    let state = state_in * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_f(seed: ptr<function, u32>) -> f32 {
    *seed = pcg(*seed);
    return f32(*seed) / 4294967296.0;
}

fn rand_vec(seed: ptr<function, u32>) -> vec3<f32> {
    return vec3<f32>(rand_f(seed), rand_f(seed), rand_f(seed));
}

fn cosine_hemisphere(n: vec3<f32>, seed: ptr<function, u32>) -> vec3<f32> {
    // Build a basis around n, then map a disk sample onto the hemisphere.
    var helper = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(n.y) > 0.99) {
        helper = vec3<f32>(1.0, 0.0, 0.0);
    }
    let t = normalize(cross(helper, n));
    let b = cross(n, t);
    let r1 = rand_f(seed);
    let r2 = rand_f(seed);
    let r = sqrt(r1);
    let phi = 6.28318530718 * r2;
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - r1)));
}

// ------------------------------------------------------------- ray query ----
struct Hit {
    hit: bool,
    t: f32,
    tri: u32,
    bary: vec2<f32>,
    front: bool,
    p: vec3<f32>,
};

fn trace(ro: vec3<f32>, rd: vec3<f32>, t_max: f32) -> Hit {
    var rq: ray_query;
    var desc: RayDesc;
    desc.origin = ro;
    desc.dir = rd;
    desc.tmin = 0.0005;
    desc.tmax = t_max;
    desc.flags = 0u;
    desc.cull_mask = 0xFFu;

    // The BLASes are marked opaque, so a single Proceed() commits the closest
    // hit and returns false; there is no need to loop over candidates.
    rayQueryInitialize(&rq, tlas, desc);
    rayQueryProceed(&rq);
    let isect = rayQueryGetCommittedIntersection(&rq);

    var h: Hit;
    h.hit = isect.kind != 0u;
    h.t = isect.t;
    h.tri = isect.primitive_index;
    h.bary = isect.barycentrics;
    h.front = isect.front_face;
    h.p = ro + rd * isect.t;
    return h;
}

fn tri_normal(h: Hit) -> vec3<f32> {
    let i0 = indices[h.tri * 3u];
    let i1 = indices[h.tri * 3u + 1u];
    let i2 = indices[h.tri * 3u + 2u];
    let b = h.bary;
    let n0 = normals[i0].xyz;
    let n1 = normals[i1].xyz;
    let n2 = normals[i2].xyz;
    let n = n0 * (1.0 - b.x - b.y) + n1 * b.x + n2 * b.y;
    return normalize(n);
}

fn tri_vertex_normal(i: u32) -> vec3<f32> {
    return normals[i].xyz;
}

fn tri_vertex(i: u32) -> vec3<f32> {
    return positions[i].xyz;
}

fn sky_radiance(dir: vec3<f32>) -> vec3<f32> {
    let up = clamp(dir.y, -1.0, 1.0);
    let horizon = u.sky.rgb * 0.55;
    let zenith = u.sky.rgb * 1.05;
    let color = mix(horizon, zenith, pow(max(up, 0.0), 0.65));
    return color;
}

// One shadow ray towards a uniformly sampled point on the emissive quad.
// Without it a diffuse bounce would have to hit the panel by chance, which is
// the dominant source of noise in a naive path tracer.
fn direct_light(p: vec3<f32>, n: vec3<f32>, seed: ptr<function, u32>) -> vec3<f32> {
    let a = rand_f(seed);
    let b = rand_f(seed);
    let lp = u.light_origin.xyz + u.light_u.xyz * a + u.light_v.xyz * b;
    let to_light = lp - p;
    let dist2 = dot(to_light, to_light);
    let dist = sqrt(dist2);
    let wi = to_light / dist;

    let cos_surface = dot(n, wi);
    if (cos_surface <= 0.0) {
        return vec3<f32>(0.0);
    }
    let light_n = normalize(cross(u.light_u.xyz, u.light_v.xyz));
    let cos_light = abs(dot(light_n, wi));
    let area = length(u.light_u.xyz) * length(u.light_v.xyz);

    let shadow = trace(p + n * 0.002, wi, dist - 0.004);
    if (shadow.hit) {
        return vec3<f32>(0.0);
    }
    return u.light_emission.rgb * (cos_surface * cos_light * area / dist2) / PI;
}

// --------------------------------------------------------------- tracing ----
@compute @workgroup_size(8, 8)
fn path_trace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(u.resolution.xy);
    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }
    let idx = gid.y * size.x + gid.x;
    let sample_index = u.resolution.z;
    var seed = pcg(idx * 1973u ^ pcg(u32(u.params.z) + u32(sample_index) * 9781u));

    // Sub-pixel jitter (anti-aliasing) and a primary ray through the pixel.
    let jitter = vec2<f32>(rand_f(&seed), rand_f(&seed));
    let ndc = vec2<f32>(
        (f32(gid.x) + jitter.x) / u.resolution.x * 2.0 - 1.0,
        1.0 - (f32(gid.y) + jitter.y) / u.resolution.y * 2.0,
    );
    let aspect = u.resolution.x / u.resolution.y;
    let tan_half = u.cam_fwd.w;
    var rd = normalize(
        u.cam_fwd.xyz + u.cam_right.xyz * (ndc.x * aspect * tan_half) + u.cam_up.xyz * (ndc.y * tan_half),
    );
    var ro = u.cam_pos.xyz;

    var radiance = vec3<f32>(0.0);
    var throughput = vec3<f32>(1.0);
    // Emission is counted for camera rays and after specular bounces; after a
    // diffuse bounce the light was already sampled by direct_light().
    var count_emission = true;

    let max_bounces = u32(u.resolution.w);
    for (var bounce = 0u; bounce < max_bounces; bounce = bounce + 1u) {
        let h = trace(ro, rd, 1.0e5);
        if (!h.hit) {
            radiance += throughput * sky_radiance(rd);
            break;
        }

        let mat_index = tri_material[h.tri];
        let mat = materials[mat_index];
        let kind = u32(mat.emissive.w);
        if (kind == 1u) {
            if (count_emission) {
                radiance += throughput * mat.emissive.rgb;
            }
            break;
        }

        var n = tri_normal(h);
        if (dot(n, rd) > 0.0) {
            n = -n;
        }

        var albedo = mat.albedo.rgb;
        if (kind == 2u) {
            // Procedural checkerboard on the floor.
            let c = floor(h.p.x) + floor(h.p.z);
            let odd = fract(c * 0.5) > 0.25;
            albedo = albedo * select(0.06, 1.0, odd);
        }
        throughput *= albedo;

        let rough = clamp(mat.params.x, 0.0, 1.0);
        let is_diffuse = rand_f(&seed) < rough;

        if (is_diffuse) {
            // Direct light, then a cosine-weighted indirect bounce.
            radiance += throughput * direct_light(h.p, n, &seed);
            rd = cosine_hemisphere(n, &seed);
            ro = h.p + n * 0.002;
            count_emission = false;
        } else {
            // (Glossy) mirror: perturb the reflection a little by roughness.
            var r = reflect(rd, n);
            if (rough > 0.001) {
                r = normalize(r + rand_vec(&seed) * rough * 0.35);
                if (dot(r, n) <= 0.0) {
                    r = reflect(rd, n);
                }
            }
            rd = r;
            ro = h.p + n * 0.002;
            count_emission = true;
        }

        // Cheap Russian roulette: kill paths that can no longer contribute.
        if (bounce > 2u) {
            let m = max(throughput.r, max(throughput.g, throughput.b));
            if (m < 0.01) {
                break;
            }
        }
    }

    // Running average: mix(previous average, new sample, 1/(n+1)).
    let prev = accum[idx].rgb;
    let n_samples = sample_index + 1.0;
    accum[idx] = vec4<f32>(mix(prev, radiance, 1.0 / n_samples), 1.0);
}

// ---------------------------------------------------------------- display ----
@group(0) @binding(0) var<uniform> du: Uniforms;
@group(0) @binding(1) var<storage, read> daccum: array<vec4<f32>>;

// ACES filmic approximation (Narkowicz), applied in linear space.
fn aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

struct VsOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VsOut {
    // One oversized triangle covering the viewport; no vertex buffer needed.
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(3.0, 1.0),
    );
    var out: VsOut;
    out.pos = vec4<f32>(corners[vi], 0.0, 1.0);
    return out;
}

@fragment
fn fs_tonemap(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let size = vec2<u32>(du.resolution.xy);
    let x = min(u32(frag.x), size.x - 1u);
    let y = min(u32(frag.y), size.y - 1u);
    let idx = y * size.x + x;

    var color = daccum[idx].rgb * du.params.y;
    color = aces(color);
    // The render target is an sRGB texture, so no manual gamma is needed.
    return vec4<f32>(color, 1.0);
}
