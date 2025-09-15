#![allow(dead_code)]

use std::f32;

use image::{Rgb, RgbImage};
use indicatif::{ProgressBar, ProgressStyle};
use rand::prelude::*;

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
    fn clamp01(self) -> Self {
        Self::new(
            self.x.clamp(0.0, 1.0),
            self.y.clamp(0.0, 1.0),
            self.z.clamp(0.0, 1.0),
        )
    }
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
    /// Builds a new perspective pinhole camera
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

    /// Builds a perspective pinhole camera based on **focal length** and **sensor height** (both in millimeters).
    /// # Arguments
    /// * `aspect` - Image aspect ratio (width / height).
    /// * `focal_length` - Lens focal length (in millimeters).
    /// * `sensor_height` - Sensor height (in millimeters).
    /// * `look_from` - Camera position from world origin.
    /// * `look_at` - Target that the camera is aimed at.
    /// * `vup` - Up direction ((0, 0, 1) would be Z-up)
    /// # Returns
    /// A new `Camera` positioned at `look_from` and aimed at `look_at`.
    fn from_focal_sensor(
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
}

#[derive(Debug, Clone, Copy)]
struct HitInfo {
    p: Vec3,
    normal: Vec3,
    t: f32,
    front_face: bool,
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

    fn new(ray: &Ray, p: Vec3, outward_normal: Vec3, t: f32) -> Self {
        let (normal, front_face) = Self::set_face_normal(ray, outward_normal);
        Self {
            p,
            normal,
            t,
            front_face,
        }
    }
}

trait Hittable {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitInfo>;
}

struct Sphere {
    center: Vec3,
    radius: f32,
}

impl Sphere {
    fn new(center: Vec3, radius: f32) -> Self {
        Self { center, radius }
    }
}

impl Hittable for Sphere {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitInfo> {
        // This is pure magic copied from ChatGPT
        // I have no idea how this works or why
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
        let scale = self.power / (self.area().max(1e-8) * f32::consts::PI);
        self.color.smul(scale)
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
    world.hit(&shadow_ray, 0.0, dist - eps).is_none()
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

            let brdf = albedo.smul(1.0 / f32::consts::PI);
            let geom = (cos_i * cos_l) / r2;
            let contribution = emission.mul(brdf).smul(geom / p_a);

            sum = sum.add(contribution);
        }
    }

    let total = (m * n) as f32;
    sum.smul(1.0 / total)
}

fn linear_to_srgb(c: Vec3) -> [u8; 3] {
    let clamp = |x: f32| x.clamp(0.0, 1.0);
    let convert = |x: f32| (clamp(x).powf(1.0 / 2.2) * 255.0 + 0.5) as u8;
    [convert(c.x), convert(c.y), convert(c.z)]
}

fn main() {
    let width = 2160;
    let height = 1440;

    let spp = 128;

    let camera = Camera::from_focal_sensor(
        50.0,
        36.0,
        24.0,
        Vec3::new(0.0, -5.0, 0.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    );
    let sphere = Sphere::new(Vec3::new(0.0, 0.0, 0.0), 1.0);
    let area_light = AreaLight::new(
        Vec3::new(0.0, 0.0, 2.75),  // Position
        Vec3::new(0.0, 1.5, 0.0),   // Width
        Vec3::new(1.5, 0.0, 0.0),   // Height
        Vec3::new(1.0, 0.85, 0.73), // Color
        70.0,
    );

    let pb = ProgressBar::new(width as u64 * height as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} {eta}",
        )
        .unwrap()
        .progress_chars("#>-"),
    );

    let mut img = RgbImage::new(width, height);

    let mut rng = rand::rng();

    for y in 0..height {
        for x in 0..width {
            let mut color = Vec3::new(0.0, 0.0, 0.0);
            for _ in 0..spp {
                let jx = rng.random::<f32>();
                let jy = rng.random::<f32>();

                let u = (x as f32 + jx) / width as f32;
                let v = (y as f32 + jy) / height as f32;

                let ray = camera.get_ray(u, v);

                let sample_color = if let Some(hit) = sphere.hit(&ray, 1e-3, f32::INFINITY) {
                    lambert_direct_mc(
                        hit.p,
                        hit.normal,
                        Vec3::new(0.74, 0.1, 0.1),
                        &area_light,
                        &sphere,
                        3,
                        3,
                        &mut rng,
                    )
                } else {
                    Vec3::new(0.0, 0.0, 0.0)
                };

                color = color.add(sample_color);
            }
            pb.inc(1);
            color = color.smul(1.0 / spp as f32);
            img.put_pixel(x, height - 1 - y, Rgb(linear_to_srgb(color)));
        }
    }
    pb.finish();

    img.save("render.png").unwrap();
}
