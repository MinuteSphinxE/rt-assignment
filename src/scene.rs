//! Scene description: everything is baked into one triangle mesh (a single
//! bottom-level acceleration structure) with a per-triangle material index.
//!
//! The layout is the same "corridor" as the GLSL version, but the geometry is
//! real triangles now, and the light is an emissive quad so the path tracer can
//! sample it directly (next-event estimation).

use glam::Vec3;

pub const MAT_DIFFUSE: u32 = 0;
pub const MAT_LIGHT: u32 = 1;
pub const MAT_CHECKER: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Material {
    pub albedo: [f32; 4],
    /// rgb = emitted radiance, w = material kind
    pub emissive: [f32; 4],
    /// x = roughness (0 = mirror, 1 = fully diffuse)
    pub params: [f32; 4],
}

impl Material {
    pub fn new(albedo: Vec3, roughness: f32, kind: u32) -> Self {
        Self {
            albedo: [albedo.x, albedo.y, albedo.z, 0.0],
            emissive: [0.0, 0.0, 0.0, kind as f32],
            params: [roughness, 0.0, 0.0, 0.0],
        }
    }
    pub fn light(radiance: Vec3) -> Self {
        Self {
            albedo: [0.0, 0.0, 0.0, 0.0],
            emissive: [radiance.x, radiance.y, radiance.z, MAT_LIGHT as f32],
            params: [1.0, 0.0, 0.0, 0.0],
        }
    }
}

pub struct Scene {
    /// xyz + padding, 16 byte stride: usable both by the BLAS (Float32x3 with
    /// a 16 byte stride) and by the shader as `array<vec4<f32>>`.
    pub positions: Vec<[f32; 4]>,
    pub normals: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
    pub tri_material: Vec<u32>,
    pub materials: Vec<Material>,
    /// Area light, as an origin plus two edge vectors (a parallelogram).
    pub light_origin: Vec3,
    pub light_u: Vec3,
    pub light_v: Vec3,
    pub light_radiance: Vec3,
}

impl Scene {
    pub fn build() -> Self {
        let mut s = Scene {
            positions: Vec::new(),
            normals: Vec::new(),
            indices: Vec::new(),
            tri_material: Vec::new(),
            materials: Vec::new(),
            light_origin: Vec3::ZERO,
            light_u: Vec3::ZERO,
            light_v: Vec3::ZERO,
            light_radiance: Vec3::new(1.0, 0.93, 0.82) * 1.9,
        };

        let floor = s.add_material(Material::new(Vec3::new(1.0, 0.88, 0.55), 0.5, MAT_CHECKER));
        let wall_grey = s.add_material(Material::new(Vec3::new(0.55, 0.55, 0.60), 0.45, MAT_DIFFUSE));
        let wall_green = s.add_material(Material::new(Vec3::new(0.10, 0.42, 0.20), 0.45, MAT_DIFFUSE));
        let wall_red = s.add_material(Material::new(Vec3::new(0.58, 0.12, 0.14), 0.45, MAT_DIFFUSE));
        let mirror = s.add_material(Material::new(Vec3::new(0.92, 0.93, 0.96), 0.02, MAT_DIFFUSE));
        let cube_mat = s.add_material(Material::new(Vec3::new(0.90, 0.55, 0.18), 0.25, MAT_DIFFUSE));
        let satin = s.add_material(Material::new(Vec3::new(0.90, 0.75, 0.95), 0.6, MAT_DIFFUSE));
        let cyan = s.add_material(Material::new(Vec3::new(0.35, 0.80, 0.95), 0.3, MAT_DIFFUSE));
        let light_mat = s.add_material(Material::light(s.light_radiance));

        // ---- room ----------------------------------------------------------
        // floor (y = -2), checker pattern comes from the shader
        s.quad(
            Vec3::new(-8.0, -2.0, -9.8),
            Vec3::new(8.0, -2.0, -9.8),
            Vec3::new(8.0, -2.0, 9.8),
            Vec3::new(-8.0, -2.0, 9.8),
            Vec3::Y,
            floor,
        );

        // back wall, split around a doorway (x = +-2.2, y from -2 to 5.4)
        s.quad(
            Vec3::new(-8.0, -2.0, -9.8),
            Vec3::new(-2.2, -2.0, -9.8),
            Vec3::new(-2.2, 8.6, -9.8),
            Vec3::new(-8.0, 8.6, -9.8),
            Vec3::Z,
            wall_grey,
        );
        s.quad(
            Vec3::new(2.2, -2.0, -9.8),
            Vec3::new(8.0, -2.0, -9.8),
            Vec3::new(8.0, 8.6, -9.8),
            Vec3::new(2.2, 8.6, -9.8),
            Vec3::Z,
            wall_grey,
        );
        s.quad(
            Vec3::new(-2.2, 5.4, -9.8),
            Vec3::new(2.2, 5.4, -9.8),
            Vec3::new(2.2, 8.6, -9.8),
            Vec3::new(-2.2, 8.6, -9.8),
            Vec3::Z,
            wall_grey,
        );

        // side walls
        s.quad(
            Vec3::new(-8.0, -2.0, -9.8),
            Vec3::new(-8.0, -2.0, 9.8),
            Vec3::new(-8.0, 8.6, 9.8),
            Vec3::new(-8.0, 8.6, -9.8),
            Vec3::X,
            wall_green,
        );
        s.quad(
            Vec3::new(8.0, -2.0, -9.8),
            Vec3::new(8.0, -2.0, 9.8),
            Vec3::new(8.0, 8.6, 9.8),
            Vec3::new(8.0, 8.6, -9.8),
            -Vec3::X,
            wall_red,
        );

        // ---- objects -------------------------------------------------------
        // mirror ball resting on the floor
        s.sphere(Vec3::new(-3.2, -0.3, 3.0), 1.7, mirror, 40, 24);
        // glossy cube
        s.cube(Vec3::new(3.0, -1.0, 2.0), 1.0, cube_mat);
        // small floating sphere
        s.sphere(Vec3::new(0.9, 1.2, 4.2), 0.45, cyan, 28, 16);
        // satin sphere
        s.sphere(Vec3::new(4.6, -1.6, 6.0), 0.4, satin, 28, 16);

        // ---- area light ----------------------------------------------------
        // Big emissive panel just above the room opening.
        let light_center = Vec3::new(0.0, 8.2, 0.0);
        let hu = Vec3::new(3.2, 0.0, 0.0);
        let hv = Vec3::new(0.0, 0.0, 2.2);
        s.light_origin = light_center - hu - hv;
        s.light_u = hu * 2.0;
        s.light_v = hv * 2.0;
        s.quad(
            s.light_origin,
            s.light_origin + s.light_u,
            s.light_origin + s.light_u + s.light_v,
            s.light_origin + s.light_v,
            -Vec3::Y,
            light_mat,
        );

        s
    }

    fn add_material(&mut self, m: Material) -> u32 {
        self.materials.push(m);
        (self.materials.len() - 1) as u32
    }

    fn push_vertex(&mut self, p: Vec3, n: Vec3) -> u32 {
        self.positions.push([p.x, p.y, p.z, 0.0]);
        self.normals.push([n.x, n.y, n.z, 0.0]);
        (self.positions.len() - 1) as u32
    }

    /// Adds a planar quad (two triangles) with a constant normal.
    pub fn quad(&mut self, a: Vec3, b: Vec3, c: Vec3, d: Vec3, normal: Vec3, material: u32) {
        let n = normal.normalize();
        let ia = self.push_vertex(a, n);
        let ib = self.push_vertex(b, n);
        let ic = self.push_vertex(c, n);
        let id = self.push_vertex(d, n);
        // Winding follows a,b,c / a,c,d; the shader flips the normal for
        // back-facing hits, so the orientation only matters for culling (off).
        self.indices.extend_from_slice(&[ia, ib, ic, ia, ic, id]);
        self.tri_material.push(material);
        self.tri_material.push(material);
    }

    pub fn cube(&mut self, center: Vec3, half: f32, material: u32) {
        let h = Vec3::splat(half);
        let c = center;
        let faces = [
            (Vec3::X, [c + Vec3::new(h.x, -h.y, -h.z), c + Vec3::new(h.x, h.y, -h.z), c + Vec3::new(h.x, h.y, h.z), c + Vec3::new(h.x, -h.y, h.z)]),
            (-Vec3::X, [c + Vec3::new(-h.x, -h.y, h.z), c + Vec3::new(-h.x, h.y, h.z), c + Vec3::new(-h.x, h.y, -h.z), c + Vec3::new(-h.x, -h.y, -h.z)]),
            (Vec3::Y, [c + Vec3::new(-h.x, h.y, -h.z), c + Vec3::new(h.x, h.y, -h.z), c + Vec3::new(h.x, h.y, h.z), c + Vec3::new(-h.x, h.y, h.z)]),
            (-Vec3::Y, [c + Vec3::new(-h.x, -h.y, h.z), c + Vec3::new(h.x, -h.y, h.z), c + Vec3::new(h.x, -h.y, -h.z), c + Vec3::new(-h.x, -h.y, -h.z)]),
            (Vec3::Z, [c + Vec3::new(h.x, -h.y, h.z), c + Vec3::new(h.x, h.y, h.z), c + Vec3::new(-h.x, h.y, h.z), c + Vec3::new(-h.x, -h.y, h.z)]),
            (-Vec3::Z, [c + Vec3::new(-h.x, -h.y, -h.z), c + Vec3::new(-h.x, h.y, -h.z), c + Vec3::new(h.x, h.y, -h.z), c + Vec3::new(h.x, -h.y, -h.z)]),
        ];
        for (n, v) in faces {
            self.quad(v[0], v[1], v[2], v[3], n, material);
        }
    }

    /// UV sphere tessellated into triangles (smooth per-vertex normals).
    pub fn sphere(&mut self, center: Vec3, radius: f32, material: u32, slices: u32, stacks: u32) {
        let mut grid: Vec<Vec<u32>> = Vec::with_capacity((stacks + 1) as usize);
        for i in 0..=stacks {
            let theta = std::f32::consts::PI * i as f32 / stacks as f32;
            let mut row = Vec::with_capacity((slices + 1) as usize);
            for j in 0..=slices {
                let phi = std::f32::consts::TAU * j as f32 / slices as f32;
                let n = Vec3::new(theta.sin() * phi.cos(), theta.cos(), theta.sin() * phi.sin());
                row.push(self.push_vertex(center + n * radius, n));
            }
            grid.push(row);
        }
        for i in 0..stacks {
            for j in 0..slices {
                let a = grid[i as usize][j as usize];
                let b = grid[(i + 1) as usize][j as usize];
                let c = grid[(i + 1) as usize][(j + 1) as usize];
                let d = grid[i as usize][(j + 1) as usize];
                self.indices.extend_from_slice(&[a, b, c, a, c, d]);
                self.tri_material.push(material);
                self.tri_material.push(material);
            }
        }
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
}

#[derive(Clone, Debug)]
pub struct Camera {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub target: Vec3,
    pub fov_y: f32,
}

impl Default for Camera {
    fn default() -> Self {
        // The GLSL version looked down the corridor from z ~ +14.
        Self {
            yaw: 0.0,
            pitch: -0.06,
            distance: 14.0,
            target: Vec3::new(0.0, 1.6, 0.0),
            fov_y: 52.0_f32.to_radians(),
        }
    }
}

impl Camera {
    pub fn eye(&self) -> Vec3 {
        let cp = self.pitch.cos();
        self.target
            + Vec3::new(
                self.distance * cp * self.yaw.sin(),
                self.distance * self.pitch.sin(),
                self.distance * cp * self.yaw.cos(),
            )
    }

    /// Returns (forward, right, up).
    pub fn basis(&self) -> (Vec3, Vec3, Vec3) {
        let fwd = (self.target - self.eye()).normalize();
        let right = fwd.cross(Vec3::Y).normalize();
        let up = right.cross(fwd);
        (fwd, right, up)
    }

    /// A simple orbit camera, used by the UI.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.005;
        self.pitch = (self.pitch + dy * 0.005).clamp(-1.2, 1.2);
    }


}




