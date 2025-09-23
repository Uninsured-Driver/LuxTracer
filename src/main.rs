#![allow(dead_code)]

use std::{
    f32::{self, consts::PI},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
    time::Duration,
};

use clap::{ArgAction, Parser};
use image::RgbImage;
use indicatif::{ProgressBar, ProgressStyle};
use rand::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3 {
    // ---------- Constructors ----------
    fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
    fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
    fn one() -> Self {
        Self::new(1.0, 1.0, 1.0)
    }

    // ---------- Arithmetic ----------
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
    fn smul(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }
    fn mul(self, rhs: Self) -> Self {
        Self::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }

    // ---------- Vector operations ----------
    fn dot(&self, rhs: &Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }
    fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }
    fn length(&self) -> f32 {
        self.dot(self).sqrt()
    }
    fn norm(self) -> Self {
        let len = self.length();
        if len == 0.0 {
            self
        } else {
            self.smul(1.0 / len)
        }
    }

    // ---------- Utilities ----------
    fn max_comp(self) -> f32 {
        self.x.max(self.y).max(self.z)
    }
}

#[derive(Debug, Clone, Copy)]
struct Ray {
    origin: Vec3,
    dir: Vec3,
}

impl Ray {
    fn new(origin: Vec3, dir: Vec3) -> Self {
        Self {
            origin,
            dir: dir.norm(),
        }
    }

    fn at(&self, t: f32) -> Vec3 {
        self.origin.add(self.dir.smul(t))
    }
}

#[derive(Debug, Clone, Copy)]
struct Camera {
    origin: Vec3,
    lower_left: Vec3,
    horiz: Vec3,
    vert: Vec3,
}

impl Camera {
    /// Builds a new perspective pinhole camera.
    fn new(aspect: f32, vfov: f32, look_from: Vec3, look_at: Vec3, vup: Vec3) -> Camera {
        let theta = vfov.to_radians();
        let h = (theta * 0.5).tan();
        let viewport_h = 2.0 * h;
        let viewport_w = aspect * viewport_h;

        let w = (look_from.sub(look_at)).norm();
        let u = vup.cross(w).norm();
        let v = w.cross(u);

        let origin = look_from;
        let horiz = u.smul(viewport_w);
        let vert = v.smul(viewport_h);

        let lower_left = origin.sub(horiz.smul(0.5)).sub(vert.smul(0.5)).sub(w);
        Camera {
            origin,
            lower_left,
            horiz,
            vert,
        }
    }

    fn get_ray(&self, u: f32, v: f32) -> Ray {
        let p = self
            .lower_left
            .add(self.horiz.smul(u))
            .add(self.vert.smul(v));
        Ray::new(self.origin, p.sub(self.origin))
    }

    /// Builds a perspective pinhole camera based on the physical properties of cameras, like **focal length**, **sensor width** and **sensor height** (both in millimeters).
    /// # Arguments
    /// * `focal_length` - Lens focal length (in millimeters).
    /// * `sensor_width` - Sensor width (in millimeters)
    /// * `sensor_height` - Sensor height (in millimeters).
    /// * `look_from` - Camera position from world origin.
    /// * `look_at` - Target that the camera is aimed at.
    /// * `vup` - Up direction ((0, 0, 1) would be Z-up)
    /// # Returns
    /// A new `Camera` positioned at `look_from` and aimed at `look_at`.
    fn from_physical_camera(
        focal_length: f32,
        sensor_width: f32,
        sensor_height: f32,
        look_from: Vec3,
        look_at: Vec3,
        vup: Vec3,
    ) -> Self {
        let vfov_rad = 2.0 * ((sensor_height * 0.5) / focal_length).atan();
        let aspect = sensor_width / sensor_height;
        Self::new(aspect, vfov_rad.to_degrees(), look_from, look_at, vup)
    }

    fn from_resolution(
        focal_length: f32,
        sensor_height: f32,
        width: u32,
        height: u32,
        look_from: Vec3,
        look_at: Vec3,
        vup: Vec3,
    ) -> Self {
        let focal_length_px = focal_length * (height as f32 / sensor_height);

        let vfov_rad = 2.0 * ((height as f32 * 0.5) / focal_length_px).atan();
        let aspect = width as f32 / height as f32;
        Self::new(aspect, vfov_rad.to_degrees(), look_from, look_at, vup)
    }
}

trait Hittable: Send + Sync {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitInfo>;
}

#[derive(Clone)]
struct HitInfo {
    p: Vec3,
    normal: Vec3,
    t: f32,
    front_face: bool,
    material: Materials,
}

impl HitInfo {
    fn set_face_normal(ray: &Ray, outward_normal: Vec3) -> (Vec3, bool) {
        let front_face = ray.dir.dot(&outward_normal) < 0.0;
        let normal = if front_face {
            outward_normal
        } else {
            outward_normal.smul(-1.0)
        };
        (normal, front_face)
    }

    fn new(ray: &Ray, p: Vec3, outward_normal: Vec3, t: f32, material: Materials) -> Self {
        let (normal, front_face) = Self::set_face_normal(ray, outward_normal);
        Self {
            p,
            normal,
            t,
            front_face,
            material,
        }
    }
}

#[derive(Clone)]
enum Materials {
    Reflective(Arc<dyn Reflective>),
    Emissive(Arc<dyn Emissive>),
}

trait Reflective: Send + Sync {
    fn eval_brdf(&self, wi: Vec3, wo: Vec3, n: Vec3) -> Vec3;
    fn sample_brdf(&self, n: Vec3, rng: &mut ThreadRng) -> (Vec3, f32, Vec3);
}

trait Emissive: Send + Sync {
    fn emission(&self, wo: Vec3, n: Vec3) -> Vec3;
}

struct Lambert {
    albedo: Vec3,
}

impl Lambert {
    fn new(albedo: Vec3) -> Self {
        Self { albedo }
    }
}

impl Reflective for Lambert {
    fn eval_brdf(&self, _wi: Vec3, _wo: Vec3, _n: Vec3) -> Vec3 {
        self.albedo.smul(1.0 / PI)
    }

    fn sample_brdf(&self, n: Vec3, rng: &mut ThreadRng) -> (Vec3, f32, Vec3) {
        let (sample, pdf) = cosine_weighted_sampling(rng);
        let (t, b) = build_onb(n);
        let wi = local_to_world(sample, t, b, n);

        let brdf = self.albedo.smul(1.0 / PI);

        (wi, pdf, brdf)
    }
}

struct DiffuseEmitter {
    radiance: Vec3,
}

impl DiffuseEmitter {
    fn new(radiance: Vec3) -> Self {
        Self { radiance }
    }

    fn from_power(color: Vec3, power: f32, area: f32) -> Self {
        let scale = power / (area.max(1e-8) * PI);
        Self {
            radiance: color.smul(scale),
        }
    }
}

impl Emissive for DiffuseEmitter {
    fn emission(&self, wo: Vec3, n: Vec3) -> Vec3 {
        if n.dot(&wo) > 0.0 {
            self.radiance
        } else {
            Vec3::zero()
        }
    }
}

#[derive(Clone)]
struct Sphere {
    center: Vec3,
    radius: f32,
    material: Materials,
}

impl Sphere {
    fn new(center: Vec3, radius: f32, material: Materials) -> Self {
        Self {
            center,
            radius,
            material,
        }
    }
}

impl Hittable for Sphere {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitInfo> {
        let oc = ray.origin.sub(self.center);

        let a = ray.dir.dot(&ray.dir);
        let h = ray.dir.dot(&oc);
        let c_term = oc.dot(&oc) - self.radius.powi(2);

        let discriminant = h * h - a * c_term;
        if discriminant < 0.0 {
            return None;
        }
        let sqrt_d = discriminant.sqrt();

        let mut t = (-h - sqrt_d) / a;
        if t < t_min || t > t_max {
            t = (-h + sqrt_d) / a;
            if t < t_min || t > t_max {
                return None;
            }
        }

        let p = ray.at(t);
        let outward_normal = p.sub(self.center).smul(1.0 / self.radius);
        let (normal, front_face) = HitInfo::set_face_normal(ray, outward_normal);

        Some(HitInfo {
            p,
            normal,
            t,
            front_face,
            material: self.material.clone(),
        })
    }
}

struct Quad {
    center: Vec3,
    ux: Vec3,
    vy: Vec3,
    material: Materials,
}

impl Quad {
    fn new(center: Vec3, ux: Vec3, vy: Vec3, material: Materials) -> Self {
        Self {
            center,
            ux,
            vy,
            material,
        }
    }

    fn area(&self) -> f32 {
        self.ux.cross(self.vy).length() * 4.0
    }
}

impl Hittable for Quad {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitInfo> {
        let n = self.ux.cross(self.vy);
        if n.length() < 1e-3 {
            return None;
        }
        let n_norm = n.norm();

        let denom = n_norm.dot(&ray.dir);
        if denom.abs() < 1e-3 {
            return None;
        }

        let oc = self.center.sub(ray.origin);
        let t = n_norm.dot(&oc) / denom;
        if t < t_min || t > t_max {
            return None;
        }

        let p = ray.at(t);
        let r = p.sub(self.center);

        let m00 = self.ux.dot(&self.ux);
        let m01 = self.ux.dot(&self.vy);
        let m11 = self.vy.dot(&self.vy);
        let rhs0 = self.ux.dot(&r);
        let rhs1 = self.vy.dot(&r);

        let det = m00 * m11 - m01 * m01;
        if det.abs() < 1e-3 {
            return None;
        }

        let inv_det = 1.0 / det;
        let a = (rhs0 * m11 - rhs1 * m01) * inv_det;
        let b = (rhs1 * m00 - rhs0 * m01) * inv_det;

        let padding = 1.0 + 1e-3;
        if a < -padding || a > padding || b < -padding || b > padding {
            return None;
        }

        let (normal, front_face) = HitInfo::set_face_normal(ray, n_norm);

        Some(HitInfo {
            p,
            normal,
            t,
            front_face,
            material: self.material.clone(),
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct AreaLight {
    center: Vec3,
    ux: Vec3,
    vy: Vec3,
    color: Vec3,
    power: f32,
}

impl AreaLight {
    fn new(center: Vec3, ux: Vec3, vy: Vec3, color: Vec3, power: f32) -> Self {
        Self {
            center,
            ux,
            vy,
            color,
            power,
        }
    }

    fn normal(&self) -> Vec3 {
        self.ux.cross(self.vy).norm()
    }

    fn area(&self) -> f32 {
        self.ux.cross(self.vy).length() * 4.0
    }

    fn sample_point_uniform<R: Rng + ?Sized>(&self, rng: &mut R) -> (Vec3, f32) {
        let sx = rng.random::<f32>() * 2.0 - 1.0;
        let sy = rng.random::<f32>() * 2.0 - 1.0;

        let y = self.center.add(self.ux.smul(sx)).add(self.vy.smul(sy));

        let p_a = 1.0 / self.area().max(1e-8);

        (y, p_a)
    }

    fn sample_point_stratified<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        ix: u32,
        iy: u32,
        m: u32,
        n: u32,
    ) -> (Vec3, f32) {
        let jx = rng.random::<f32>();
        let jy = rng.random::<f32>();

        let u = ((ix as f32 + jx) / m as f32) * 2.0 - 1.0;
        let v = ((iy as f32 + jy) / n as f32) * 2.0 - 1.0;

        let y = self.center.add(self.ux.smul(u)).add(self.vy.smul(v));

        let p_a = 1.0 / self.area().max(1e-8);
        (y, p_a)
    }

    fn emission(&self) -> Vec3 {
        let scale = self.power / (self.area().max(1e-8) * PI);
        self.color.smul(scale)
    }
}

struct HittableList {
    objects: Vec<Box<dyn Hittable + Send + Sync>>,
}

impl HittableList {
    fn new() -> Self {
        Self { objects: vec![] }
    }

    fn add<T: Hittable + Send + Sync + 'static>(&mut self, object: T) {
        self.objects.push(Box::new(object));
    }
}

impl Hittable for HittableList {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitInfo> {
        let mut closest = t_max;
        let mut hit = None;

        for o in &self.objects {
            if let Some(h) = o.hit(ray, t_min, closest) {
                closest = h.t;
                hit = Some(h);
            }
        }
        hit
    }
}

fn visible_to_point(p: Vec3, y: Vec3, world: &dyn Hittable) -> bool {
    let eps = 1e-3;
    let v = y.sub(p);
    let dist = v.length();
    if dist <= eps {
        return true;
    }
    let dir = v.norm();
    let shadow_ray = Ray::new(p.add(dir.smul(eps)), dir);
    world.hit(&shadow_ray, eps, (dist - eps).max(eps)).is_none()
}

fn lambert_direct_mc<R: Rng + ?Sized>(
    point: Vec3,
    normal: Vec3,
    albedo: Vec3,
    light: &AreaLight,
    world: &dyn Hittable,
    m: u32,
    n: u32,
    rng: &mut R,
) -> Vec3 {
    let mut sum = Vec3::new(0.0, 0.0, 0.0);
    let ln = light.normal();
    let emission = light.emission();

    for ix in 0..m {
        for iy in 0..n {
            let (y, p_a) = light.sample_point_stratified(rng, ix, iy, m, n);

            let v = y.sub(point);
            let r2 = v.dot(&v).max(1e-9);
            let wi = v.smul(1.0 / r2.sqrt());

            let cos_i = normal.dot(&wi).max(0.0);
            if cos_i == 0.0 {
                continue;
            }
            let cos_l = ln.dot(&wi.smul(-1.0)).max(0.0);
            if cos_l == 0.0 {
                continue;
            }

            if !visible_to_point(point, y, world) {
                continue;
            }

            let brdf = albedo.smul(1.0 / PI);
            let geom = (cos_i * cos_l) / r2;
            let contribution = emission.mul(brdf).smul(geom / p_a);

            sum = sum.add(contribution);
        }
    }

    let total = (m * n) as f32;
    sum.smul(1.0 / total)
}

fn local_to_world(local: Vec3, t: Vec3, b: Vec3, n: Vec3) -> Vec3 {
    t.smul(local.x).add(b.smul(local.y)).add(n.smul(local.z))
}

fn cosine_weighted_sampling<R: Rng + ?Sized>(rng: &mut R) -> (Vec3, f32) {
    let u1 = rng.random::<f32>();
    let u2 = rng.random::<f32>();

    let r = u1.sqrt();
    let phi = 2.0 * PI * u2;

    let x = r * phi.cos();
    let y = r * phi.sin();
    let z = (1.0 - u1).max(0.0).sqrt();

    let pdf = (z / PI).max(1e-8);
    (Vec3::new(x, y, z), pdf)
}

// From https://graphics.pixar.com/library/OrthonormalB/paper.pdf
fn build_onb(n: Vec3) -> (Vec3, Vec3) {
    let n = n.norm();

    let sign = 1.0_f32.copysign(n.z);
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;

    let t = Vec3::new(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    let b = Vec3::new(b, sign + n.y * n.y * a, -n.y);

    (t, b)
}

fn trace(
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    camera: &Camera,
    world: &dyn Hittable,
    max_depth: u32,
    rr_start: u32,
    spp: u32,
    rng: &mut ThreadRng,
) -> Vec3 {
    let mut sum = Vec3::zero();

    for _ in 0..spp {
        let mut radiance = Vec3::zero();
        let mut throughput = Vec3::one();

        let jx = rng.random::<f32>();
        let jy = rng.random::<f32>();

        let u = (x as f32 + jx) / width as f32;
        let v = (y as f32 + jy) / height as f32;

        let mut ray = camera.get_ray(u, v);

        for i in 0..max_depth {
            let Some(hit) = world.hit(&ray, 1e-3, f32::INFINITY) else {
                break;
            };

            let n = hit.normal;
            let wo = ray.dir.smul(-1.0);

            match &hit.material {
                Materials::Emissive(emissive) => {
                    let emission = emissive.emission(wo, n);
                    radiance = radiance.add(throughput.mul(emission));
                    break;
                }

                Materials::Reflective(reflective) => {
                    let (wi, pdf, brdf) = reflective.sample_brdf(n, rng);
                    if pdf <= 1e-8 {
                        break;
                    }

                    let cos_i = n.dot(&wi).max(0.0);
                    if cos_i == 0.0 {
                        break;
                    }

                    throughput = throughput.mul(brdf).smul(cos_i / pdf);

                    if i >= rr_start {
                        let mut q = throughput.max_comp();
                        q = q.clamp(0.05, 0.95);
                        if rng.random::<f32>() > q {
                            break;
                        }
                        throughput = throughput.smul(1.0 / q);
                    }

                    ray = Ray::new(hit.p.add(wi.smul(1e-3)), wi);
                }
            }
        }
        sum = sum.add(radiance);
    }
    sum.smul(1.0 / spp as f32)
}

fn linear_to_srgb(rgb: (f32, f32, f32)) -> [u8; 3] {
    fn to8(x: f32) -> u8 {
        let x = x.clamp(0.0, 1.0);

        let s = if x <= 0.0031308 {
            12.92 * x
        } else {
            1.055 * x.powf(1.0 / 2.4) - 0.055
        };
        (s * 255.0 + 0.5) as u8
    }
    [to8(rgb.0), to8(rgb.1), to8(rgb.2)]
}

#[derive(Parser)]
#[command(version, disable_help_flag = true)]
struct Args {
    /// Width of the rendered image
    #[arg(long = "width", default_value_t = 2560)]
    width: u32,

    /// Height of the rendered image
    #[arg(long = "height", default_value_t = 1440)]
    height: u32,

    /// Number of samples used for the renderer
    #[arg(short = 's', long = "samples", default_value_t = 128)]
    samples: u32,

    /// Exposure setting
    #[arg(short = 'e', long = "exposure", default_value_t = 0.0)]
    exposure: f32,

    /// Print help
    #[arg(long = "help", action = ArgAction::Help)]
    help: Option<bool>,
}

fn aces_tonemapping(rgb: Vec3) -> (f32, f32, f32) {
    fn map(x: f32) -> f32 {
        let a = 2.51_f32;
        let b = 0.03_f32;
        let c = 2.43_f32;
        let d = 0.59_f32;
        let e = 0.14_f32;
        ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0.0, 1.0)
    }
    (map(rgb.x), map(rgb.y), map(rgb.z))
}

fn main() {
    let args = Args::parse();

    let width = args.width;
    let height = args.height;

    let exposure = args.exposure;

    let camera = Camera::from_resolution(
        50.0,                       // Focal length
        24.0,                       // Sensor size
        width,                      // Image width
        height,                     // Image height
        Vec3::new(0.0, -10.0, 0.0), // Look from
        Vec3::new(0.0, 0.0, 0.0),   // Look at
        Vec3::new(0.0, 0.0, 1.0),   // Up (0.0, 0.0, 1.0) for Z-up
    );

    let material1 = Arc::new(Lambert::new(Vec3::new(0.74, 0.1, 0.1))); // Red
    let material2 = Arc::new(Lambert::new(Vec3::new(0.16, 0.4, 0.9))); // Blue
    let material3 = Arc::new(Lambert::new(Vec3::new(0.07, 0.82, 0.29))); // Green

    let sphere1 = Sphere::new(
        Vec3::new(0.0, 0.0, 0.0),
        1.0,
        Materials::Reflective(material1),
    );
    let sphere2 = Sphere::new(
        Vec3::new(5.0, 0.0, 1.5),
        3.0,
        Materials::Reflective(material2),
    );
    let sphere3 = Sphere::new(
        Vec3::new(-2.5, 0.0, 0.0),
        0.5,
        Materials::Reflective(material3),
    );

    let light_u = Vec3::new(1.5, 0.0, 0.0);
    let light_v = Vec3::new(0.0, 1.5, 0.0);

    let emissive_material = Arc::new(DiffuseEmitter::from_power(
        Vec3::new(1.0, 0.85, 0.73), // ~4500k
        70.0,
        light_u.cross(light_v).length() * 4.0,
    ));
    let light = Quad::new(
        Vec3::new(0.0, 0.0, 3.0),
        light_u,
        light_v,
        Materials::Emissive(emissive_material),
    );

    let quad = Quad::new(
        Vec3::new(0.0, 0.0, -1.75),
        Vec3::new(10.0, 0.0, 0.0),
        Vec3::new(0.0, 10.0, 0.0),
        Materials::Reflective(Arc::new(Lambert::new(Vec3::new(1.0, 0.957, 0.898)))),
    );

    let mut hl = HittableList::new();
    hl.add(sphere1);
    hl.add(sphere2);
    hl.add(sphere3);
    hl.add(light);
    hl.add(quad);

    let counter = Arc::new(AtomicU64::new(0));
    let is_done = Arc::new(AtomicBool::new(false));

    let pb_counter = Arc::clone(&counter);
    let rayon_counter = Arc::clone(&counter);
    let pb_is_done = Arc::clone(&is_done);

    let pb = ProgressBar::new(width as u64 * height as u64);
    pb.set_style(ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} {percent_precise}% {eta_precise}",
    )
    .unwrap()
    .progress_chars("#>-"));
    pb.enable_steady_tick(Duration::from_millis(33));

    let pb_handle = thread::spawn(move || {
        while !pb_is_done.load(Ordering::Relaxed) {
            let pos = pb_counter.load(Ordering::Relaxed);
            pb.set_position(pos);
            thread::sleep(Duration::from_millis(33));
        }
        pb.finish();
        println!();
    });

    let row_stride = width as usize * 3;
    let mut pixels = vec![0u8; width as usize * height as usize * 3];

    pixels
        .par_chunks_mut(row_stride)
        .enumerate()
        .map_init(
            || (rand::rng(), Arc::clone(&rayon_counter)),
            |(rng, counter), (y_idx, row)| {
                let y = height - 1 - y_idx as u32;

                for x in 0..width {
                    let mut color =
                        trace(x, y, width, height, &camera, &hl, 10, 5, args.samples, rng);
                    color = color.smul(2.0f32.powf(exposure));
                    let [r, g, b] = linear_to_srgb(aces_tonemapping(color));

                    let i = (x as usize) * 3;
                    row[i] = r;
                    row[i + 1] = g;
                    row[i + 2] = b;

                    counter.fetch_add(1, Ordering::Relaxed);
                }
            },
        )
        .for_each(drop);
    is_done.store(true, Ordering::Relaxed);
    pb_handle.join().unwrap();

    RgbImage::from_raw(width, height, pixels)
        .unwrap()
        .save("render.png")
        .unwrap();
}
