#![allow(dead_code)]

use std::{
    f32::{self, consts::PI},
    ops::Neg,
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
    fn splat(value: f32) -> Self {
        Self::new(value, value, value)
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
    fn max_comp(&self) -> f32 {
        self.x.max(self.y).max(self.z)
    }
    fn mean(&self) -> f32 {
        (self.x + self.y + self.z) / 3.0
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

impl Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y, -self.z)
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
    Emissive(Arc<dyn Emissive>),
    Reflective(Arc<dyn Reflective>),
    Refractive(Arc<dyn Refractive>),
}

trait Emissive: Send + Sync {
    fn emission(&self, wo: Vec3, n: Vec3) -> Vec3;
}

trait Reflective: Send + Sync {
    fn eval_brdf(&self, wi: Vec3, wo: Vec3, n: Vec3) -> Vec3;
    fn sample_brdf(&self, wo: Vec3, n: Vec3, rng: &mut ThreadRng) -> (Vec3, f32, Vec3);
    fn sample_pdf(&self, wi: Vec3, wo: Vec3, n: Vec3) -> f32;
}

trait Refractive: Send + Sync {
    fn sample(
        &self,
        wo: Vec3,
        n: Vec3,
        front_face: bool,
        ior_i: f32,
        ior_t: f32,
        rng: &mut ThreadRng,
    ) -> (Vec3, f32, Vec3, bool);
    fn ior(&self) -> f32;
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

    fn sample_brdf(&self, _wo: Vec3, n: Vec3, rng: &mut ThreadRng) -> (Vec3, f32, Vec3) {
        let (sample, pdf) = cosine_weighted_sampling(rng);
        let (t, b) = build_onb(n);
        let wi = local_to_world(sample, t, b, n);

        let brdf = self.albedo.smul(1.0 / PI);

        (wi, pdf, brdf)
    }

    fn sample_pdf(&self, wi: Vec3, _wo: Vec3, n: Vec3) -> f32 {
        let c = n.dot(&wi).max(0.0);
        (c / PI).max(1e-8)
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

struct LightSample {
    wi: Vec3,
    dist: f32,
    y: Vec3,
    light_normal: Vec3,
    radiance: Vec3,
    pdf: f32,
}

trait Light: Send + Sync {
    fn sample(&self, x: Vec3, n: Vec3, rng: &mut ThreadRng) -> LightSample;
    fn pdf(&self, x: Vec3, wi: Vec3) -> f32;
}

struct Medium {
    ior: f32,
}

impl Medium {
    fn new(ior: f32) -> Self {
        Self { ior }
    }
}

struct Glass {
    ior: f32,
}

impl Glass {
    fn new(ior: f32) -> Self {
        Self { ior }
    }
}

impl Refractive for Glass {
    fn sample(
        &self,
        wo: Vec3,
        n: Vec3,
        _front_face: bool,
        ior_i: f32,
        ior_t: f32,
        rng: &mut ThreadRng,
    ) -> (Vec3, f32, Vec3, bool) {
        let eta = ior_i / ior_t;

        let cos_i = n.dot(&wo).abs().max(1e-8);
        let f0 = f0_from_ior(ior_i, ior_t);
        let fresnel = f0 + (1.0 - f0) * (1.0 - cos_i).powi(5);

        if let Some(refracted) = refract(-wo, n, eta) {
            if rng.random::<f32>() < fresnel {
                let reflected = reflect(-wo, n).norm();
                return (reflected, fresnel, Vec3::splat(fresnel), false);
            } else {
                let cos_t = n.dot(&refracted).abs().max(1e-8);
                let eta_scale = 1.0 / (eta * eta) * (cos_t / cos_i);
                return (
                    refracted,
                    1.0 - fresnel,
                    Vec3::splat((1.0 - fresnel) * eta_scale),
                    true,
                );
            }
        } else {
            let reflected = reflect(-wo, n).norm();
            return (reflected, 1.0, Vec3::one(), false);
        }
    }

    fn ior(&self) -> f32 {
        self.ior
    }
}

struct Pbr {
    base_color: Vec3,
    metallic: f32,
    alpha: f32,
    f0: Option<Vec3>,
}

impl Pbr {
    fn new(base_color: Vec3, metallic: f32, roughness: f32) -> Self {
        Self {
            base_color,
            metallic: metallic.clamp(0.0, 1.0),
            alpha: roughness * roughness,
            f0: None,
        }
    }
    fn from_f0(mut self, f0: Vec3) -> Self {
        self.f0 = Some(f0);
        self
    }

    #[inline]
    fn safe_rcp(x: f32) -> f32 {
        1.0 / x.max(1e-8)
    }
    #[inline]
    fn reflect(v: Vec3, n: Vec3) -> Vec3 {
        v.sub(n.smul(2.0 * v.dot(&n)))
    }

    fn f0(&self) -> Vec3 {
        if let Some(f0) = self.f0 {
            return f0;
        }

        let f0_dielectric = Vec3::new(0.04, 0.04, 0.04);
        f0_dielectric
            .smul(1.0 - self.metallic)
            .add(self.base_color.smul(self.metallic))
    }

    fn kd(&self, wi: Vec3, wo: Vec3) -> Vec3 {
        if self.metallic > 0.9999 {
            return Vec3::zero();
        }
        let h = wi.add(wo).norm();
        let v_dot_h = wo.dot(&h).max(0.0);
        let f = schlick_fresnel(v_dot_h, self.f0());
        Vec3::one().sub(f).smul(1.0 - self.metallic)
    }

    fn ggx_ndf(n_dot_h: f32, alpha: f32) -> f32 {
        let a2 = alpha * alpha;
        let cos2 = (n_dot_h * n_dot_h).max(0.0);
        let denom = cos2 * (a2 - 1.0) + 1.0;
        a2 / (PI * denom * denom).max(1e-8)
    }

    fn smith_g1_exact(n_dot_v: f32, alpha: f32) -> f32 {
        if n_dot_v <= 0.0 {
            return 0.0;
        }
        let sin2 = (1.0 - n_dot_v * n_dot_v).max(0.0);
        let tan2 = sin2 * Self::safe_rcp(n_dot_v * n_dot_v);
        let root = (1.0 + alpha * alpha * tan2).sqrt();
        2.0 / (1.0 + root)
    }

    fn smith_g1_schlick(n_dot_v: f32, alpha: f32) -> f32 {
        if n_dot_v <= 0.0 {
            return 0.0;
        }
        let k = {
            let t = alpha + 1.0;
            (t * t) * 0.125
        };
        n_dot_v / (n_dot_v * (1.0 - k) + k)
    }

    fn smith_g(n_dot_l: f32, n_dot_v: f32, alpha: f32) -> f32 {
        Self::smith_g1_exact(n_dot_l, alpha) * Self::smith_g1_exact(n_dot_v, alpha)
    }

    fn sample_ggx_vndf_local(wo_local: Vec3, alpha: f32, rng: &mut ThreadRng) -> Vec3 {
        let vh = Vec3::new(alpha * wo_local.x, alpha * wo_local.y, wo_local.z).norm();

        let lensq = vh.x * vh.x + vh.y * vh.y;
        let (t1, t2) = if lensq > 0.0 {
            let inv = 1.0 / lensq.sqrt();
            let t1 = Vec3::new(-vh.y * inv, vh.x * inv, 0.0);
            (t1, vh.cross(t1))
        } else {
            (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0))
        };

        let u1 = rng.random::<f32>();
        let u2 = rng.random::<f32>();
        let r = u1.sqrt();
        let phi = 2.0 * PI * u2;
        let t1c = r * phi.cos();
        let mut t2c = r * phi.sin();

        let s = 0.5 * (1.0 + vh.z);
        let tmp = (1.0 - t1c * t1c).max(0.0).sqrt();
        t2c = (1.0 - s) * tmp + s * t2c;

        let nh = t1
            .smul(t1c)
            .add(t2.smul(t2c))
            .add(vh.smul((1.0 - t1c * t1c - t2c * t2c).max(0.0).sqrt()));

        Vec3::new(alpha * nh.x, alpha * nh.y, nh.z).norm()
    }

    fn pdf_spec_vndf(wi: Vec3, wo: Vec3, n: Vec3, alpha: f32) -> f32 {
        let h = wi.add(wo).norm();
        let n_dot_h = n.dot(&h).max(0.0);
        let n_dot_v = n.dot(&wo).max(0.0);
        if n_dot_v <= 0.0 {
            return 0.0;
        }

        let v_dot_h = wo.dot(&h).max(1e-8);
        let d = Self::ggx_ndf(n_dot_h, alpha);
        let g1 = Self::smith_g1_exact(n_dot_v, alpha);

        (d * g1 * n_dot_h / (4.0 * v_dot_h)).max(1e-8)
    }

    fn eval_spec(&self, wi: Vec3, wo: Vec3, n: Vec3) -> Vec3 {
        let n_dot_l = n.dot(&wi).max(0.0);
        let n_dot_v = n.dot(&wo).max(0.0);
        if n_dot_l <= 0.0 || n_dot_v <= 0.0 {
            return Vec3::zero();
        }

        let h = (wi.add(wo)).norm();
        let n_dot_h = n.dot(&h).max(0.0);
        let v_dot_h = wo.dot(&h).max(0.0);

        let a = self.alpha;

        let d = Self::ggx_ndf(n_dot_h, a);
        let g = Self::smith_g(n_dot_l, n_dot_v, a);
        let f = schlick_fresnel(v_dot_h, self.f0());

        f.smul((d * g) * Self::safe_rcp(4.0 * n_dot_l * n_dot_v))
    }

    fn pdf_spec(&self, wi: Vec3, wo: Vec3, n: Vec3) -> f32 {
        Self::pdf_spec_vndf(wi, wo, n, self.alpha)
    }
}

impl Reflective for Pbr {
    fn eval_brdf(&self, wi: Vec3, wo: Vec3, n: Vec3) -> Vec3 {
        let cos_l = n.dot(&wi).max(0.0);
        let cos_v = n.dot(&wo).max(0.0);
        if cos_l <= 0.0 || cos_v <= 0.0 {
            return Vec3::zero();
        }

        let spec = self.eval_spec(wi, wo, n);

        let kd = self.kd(wi, wo);
        let diff = self.base_color.mul(kd).smul(1.0 / PI);

        diff.add(spec)
    }

    fn sample_brdf(&self, wo: Vec3, n: Vec3, rng: &mut ThreadRng) -> (Vec3, f32, Vec3) {
        let n_dot_v = n.dot(&wo).max(0.0);
        if n_dot_v <= 0.0 {
            return (Vec3::zero(), 1.0, Vec3::zero());
        }

        if self.alpha < 1e-5 {
            let wi = Self::reflect(wo.smul(-1.0), n);
            if n.dot(&wi) <= 0.0 {
                return (Vec3::zero(), 1.0, Vec3::zero());
            }

            let n_dot_v = n.dot(&wo).max(0.0);
            let n_dot_i = n.dot(&wi).max(1e-8);
            let f = schlick_fresnel(n_dot_v, self.f0());
            let brdf = f.smul(1.0 / n_dot_i);
            return (wi, 1.0, brdf);
        }

        let (t, b) = build_onb(n);
        let wo_local = world_to_local(wo, t, b, n);

        let f0 = self.f0();
        let f_avg = f0.add(Vec3::one().sub(f0).smul(1.0 / 21.0));
        let ks = f_avg.mean();

        let pdf_diff = |wi: Vec3| -> f32 {
            let c = n.dot(&wi).max(0.0);
            (c / PI).max(1e-8)
        };

        if rng.random::<f32>() < ks {
            let h_local = Self::sample_ggx_vndf_local(wo_local, self.alpha, rng);
            let h = local_to_world(h_local, t, b, n);
            let wi = Self::reflect(wo.smul(-1.0), h);

            if n.dot(&wi) <= 0.0 {
                return (Vec3::zero(), 1.0, Vec3::zero());
            }

            let pdf_s = self.pdf_spec(wi, wo, n);
            let pdf_d = pdf_diff(wi);
            let pdf = (ks * pdf_s + (1.0 - ks) * pdf_d).max(1e-8);

            let brdf = self.eval_brdf(wi, wo, n);
            (wi, pdf, brdf)
        } else {
            let (samp, _pdf_local) = cosine_weighted_sampling(rng);
            let wi = local_to_world(samp, t, b, n);
            if n.dot(&wi) <= 0.0 {
                return (Vec3::zero(), 1.0, Vec3::zero());
            }

            let pdf_s = self.pdf_spec(wi, wo, n);
            let pdf_d = pdf_diff(wi);
            let pdf = (ks * pdf_s + (1.0 - ks) * pdf_d).max(1e-8);

            let brdf = self.eval_brdf(wi, wo, n);
            (wi, pdf, brdf)
        }
    }

    fn sample_pdf(&self, wi: Vec3, wo: Vec3, n: Vec3) -> f32 {
        let n_dot_v = n.dot(&wo).max(0.0);
        let n_dot_i = n.dot(&wi).max(0.0);
        if n_dot_v <= 0.0 || n_dot_i <= 0.0 {
            return 0.0;
        }

        if self.alpha < 1e-5 {
            return 0.0;
        }

        let p_spec = self.pdf_spec(wi, wo, n);
        let p_dff = (n_dot_i / PI).max(1e-8);
        let f0 = self.f0();
        let f_avg = f0.add(Vec3::one().sub(f0).smul(1.0 / 21.0));
        let ks = f_avg.mean();
        ((ks * p_spec) + (1.0 - ks) * p_dff).max(1e-8)
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

#[derive(Clone)]
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

    fn emissive_from_power(center: Vec3, ux: Vec3, vy: Vec3, color: Vec3, power: f32) -> Self {
        let area = ux.cross(vy).length() * 4.0;
        let emitter = Arc::new(DiffuseEmitter::from_power(color, power, area));
        Quad::new(center, ux, vy, Materials::Emissive(emitter))
    }

    #[inline]
    fn normal(&self) -> Vec3 {
        self.ux.cross(self.vy).norm()
    }

    #[inline]
    fn sample_point(&self, rng: &mut ThreadRng) -> (Vec3, f32) {
        let u = rng.random::<f32>() * 2.0 - 1.0;
        let v = rng.random::<f32>() * 2.0 - 1.0;
        let y = self.center.add(self.ux.smul(u)).add(self.vy.smul(v));
        let pdf_a = 1.0 / self.area().max(1e-8);

        (y, pdf_a)
    }

    fn intersect_from(&self, x: Vec3, wi: Vec3) -> Option<(f32, Vec3)> {
        let n = self.normal();
        let denom = n.dot(&wi);
        if denom.abs() < 1e-6 {
            return None;
        }
        let t = n.dot(&self.center.sub(x)) / denom;
        if t <= 0.0 {
            return None;
        }

        let p = x.add(wi.smul(t));
        let r = p.sub(self.center);

        let m00 = self.ux.dot(&self.ux);
        let m01 = self.ux.dot(&self.vy);
        let m11 = self.vy.dot(&self.vy);
        let rhs0 = self.ux.dot(&r);
        let rhs1 = self.vy.dot(&r);

        let det = m00 * m11 - m01 * m01;
        if det.abs() < 1e-8 {
            return None;
        }

        let inv = 1.0 / det;
        let a = (rhs0 * m11 - rhs1 * m01) * inv;
        let b = (rhs1 * m00 - rhs0 * m01) * inv;

        if a < -1.0 || a > 1.0 || b < -1.0 || b > 1.0 {
            return None;
        }
        Some((t, n))
    }

    fn is_emissive(&self) -> Option<&dyn Emissive> {
        match &self.material {
            Materials::Emissive(e) => Some(e.as_ref()),
            _ => None,
        }
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
        if det.abs() < 1e-8 {
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

impl Light for Quad {
    fn sample(&self, x: Vec3, _n: Vec3, rng: &mut ThreadRng) -> LightSample {
        let light_normal = self.normal();

        let (y, pdf_a) = self.sample_point(rng);

        let to_y = y.sub(x);
        let dist2 = to_y.dot(&to_y).max(1e-12);
        let dist = dist2.sqrt();
        let wi = if dist > 0.0 {
            to_y.norm()
        } else {
            Vec3::new(0.0, 0.0, 1.0)
        };

        let cos_l = light_normal.dot(&wi.smul(-1.0)).max(0.0);

        if cos_l <= 0.0 {
            return LightSample {
                wi,
                dist,
                y,
                light_normal,
                radiance: Vec3::zero(),
                pdf: 0.0,
            };
        }

        let pdf = ((dist2 / cos_l) * pdf_a).max(1e-8);
        let radiance = if let Some(emissive) = self.is_emissive() {
            emissive.emission(wi.smul(-1.0), light_normal)
        } else {
            Vec3::zero()
        };

        LightSample {
            wi,
            dist,
            y,
            light_normal,
            radiance,
            pdf,
        }
    }
    fn pdf(&self, x: Vec3, wi: Vec3) -> f32 {
        if let Some((t, light_normal)) = self.intersect_from(x, wi) {
            let cos_l = light_normal.dot(&wi.smul(-1.0)).max(0.0);
            if cos_l <= 0.0 {
                return 0.0;
            }

            let dist2 = (t * t).max(1e-12);
            let pdf_a = (1.0 / self.area()).max(1e-8);
            return (dist2 / cos_l) * pdf_a;
        }
        0.0
    }
}

#[derive(Clone)]
struct Triangle {
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
    material: Materials,
}

impl Triangle {
    fn new(p0: Vec3, p1: Vec3, p2: Vec3, material: Materials) -> Self {
        Self {
            p0,
            p1,
            p2,
            material,
        }
    }
}

impl Hittable for Triangle {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitInfo> {
        let e1 = self.p1.sub(self.p0);
        let e2 = self.p2.sub(self.p0);

        let p = ray.dir.cross(e1);
        let det = e2.dot(&p);
        if det.abs() < 1e-8 {
            return None;
        }
        let inv_det = 1.0 / det;

        let tvec = ray.origin.sub(self.p0);
        let u = tvec.dot(&p) * inv_det;
        if u < 0.0 || u > 1.0 {
            return None;
        }

        let q = tvec.cross(e2);
        let v = ray.dir.dot(&q) * inv_det;
        if v < 0.0 || u + v > 1.0 {
            return None;
        }

        let t = e1.dot(&q) * inv_det;
        if t < t_min || t > t_max {
            return None;
        }

        let hit_p = ray.at(t);

        let n = e1.cross(e2).norm();

        Some(HitInfo::new(ray, hit_p, n, t, self.material.clone()))
    }
}

struct BvhNode {
    bbox: Aabb,
    start: u32,
    count: u32,
    left: i32,
    right: i32,
}

impl BvhNode {
    fn new_leaf(bbox: Aabb, start: u32, count: u32) -> Self {
        Self {
            bbox,
            start,
            count,
            left: -1,
            right: -1,
        }
    }

    fn new_internal(bbox: Aabb, left: i32, right: i32) -> Self {
        Self {
            bbox,
            start: 0,
            count: 0,
            left,
            right,
        }
    }
}

struct Mesh {
    tris: Vec<Triangle>,
    indices: Vec<usize>,
    nodes: Vec<BvhNode>,
}

impl Mesh {
    fn from_obj(path: &str, material: Materials, offset: Vec3, scale: Vec3) -> Self {
        let (models, _materials) = tobj::load_obj(
            path,
            &tobj::LoadOptions {
                triangulate: true,
                single_index: true,
                ..Default::default()
            },
        )
        .unwrap();

        let mut tris = Vec::new();

        for m in models {
            let mesh = m.mesh;
            let pos = &mesh.positions;
            let idx = &mesh.indices;

            for i in (0..idx.len()).step_by(3) {
                let i0 = idx[i] as usize;
                let i1 = idx[i + 1] as usize;
                let i2 = idx[i + 2] as usize;

                let p0 = Vec3::new(pos[3 * i0], pos[3 * i0 + 1], pos[3 * i0 + 2])
                    .mul(scale)
                    .add(offset);
                let p1 = Vec3::new(pos[3 * i1], pos[3 * i1 + 1], pos[3 * i1 + 2])
                    .mul(scale)
                    .add(offset);
                let p2 = Vec3::new(pos[3 * i2], pos[3 * i2 + 1], pos[3 * i2 + 2])
                    .mul(scale)
                    .add(offset);

                tris.push(Triangle::new(p0, p1, p2, material.clone()));
            }
        }

        Self {
            tris,
            indices: Vec::new(),
            nodes: Vec::new(),
        }
    }

    fn bvh_depth(&self) -> (usize, usize) {
        if self.nodes.is_empty() {
            return (0, 0);
        }

        fn traverse(nodes: &Vec<BvhNode>, id: i32, depth: usize) -> (usize, usize) {
            let node = &nodes[id as usize];
            if node.left < 0 && node.right < 0 {
                return (depth, depth);
            }

            let mut min_d = usize::MAX;
            let mut max_d = 0;

            if node.left >= 0 {
                let (min_l, max_l) = traverse(nodes, node.left, depth + 1);
                min_d = min_d.min(min_l);
                max_d = max_d.max(max_l);
            }
            if node.right >= 0 {
                let (min_r, max_r) = traverse(nodes, node.right, depth + 1);
                min_d = min_d.min(min_r);
                max_d = max_d.max(max_r);
            }

            (min_d, max_d)
        }

        traverse(&self.nodes, (self.nodes.len() - 1) as i32, 0)
    }

    fn build_bvh(&mut self) {
        if self.tris.is_empty() {
            self.indices.clear();
            self.nodes.clear();
            return;
        }

        self.nodes.clear();
        self.indices = (0..self.tris.len()).collect();

        let tri_bb: Vec<Aabb> = self.tris.iter().map(Aabb::from_triangle).collect();
        let cents: Vec<Vec3> = tri_bb.iter().map(Aabb::centroid).collect();

        fn range_bbox(ids: &[usize], bb: &[Aabb]) -> Aabb {
            let mut b = bb[ids[0]];
            for &i in &ids[1..] {
                b = b.union(bb[i]);
            }
            b
        }

        fn centroid_bbox(ids: &[usize], c: &[Vec3]) -> (Vec3, Vec3) {
            let mut mn = c[ids[0]];
            let mut mx = c[ids[0]];
            for &i in &ids[1..] {
                let v = c[i];
                mn = Vec3::new(mn.x.min(v.x), mn.y.min(v.y), mn.z.min(v.z));
                mx = Vec3::new(mx.x.max(v.x), mx.y.max(v.y), mx.z.max(v.z));
            }
            (mn, mx)
        }

        fn build(
            nodes: &mut Vec<BvhNode>,
            ids: &mut [usize],
            bb: &[Aabb],
            c: &[Vec3],
            leaf_max: usize,
            depth: usize,
            max_depth: usize,
        ) -> i32 {
            let bbox = range_bbox(ids, bb);

            if ids.len() <= leaf_max || depth >= max_depth {
                let id = nodes.len() as i32;
                nodes.push(BvhNode::new_leaf(bbox, 0, ids.len() as u32));
                return id;
            }

            let (mn, mx) = centroid_bbox(ids, c);
            let ext = mx.sub(mn);
            let (axis, extent) = if ext.x >= ext.y && ext.x >= ext.z {
                (0, ext.x)
            } else if ext.y >= ext.z {
                (1, ext.y)
            } else {
                (2, ext.z)
            };
            if extent < 1e-6 {
                let id = nodes.len() as i32;
                nodes.push(BvhNode::new_leaf(bbox, 0, ids.len() as u32));
                return id;
            }

            let mid = ids.len() / 2;
            ids.select_nth_unstable_by(mid, |&a, &b| {
                let ca = c[a];
                let cb = c[b];
                match axis {
                    0 => ca.x.partial_cmp(&cb.x).unwrap_or(std::cmp::Ordering::Equal),
                    1 => ca.y.partial_cmp(&cb.y).unwrap_or(std::cmp::Ordering::Equal),
                    _ => ca.z.partial_cmp(&cb.z).unwrap_or(std::cmp::Ordering::Equal),
                }
            });

            let (l_ids, r_ids) = ids.split_at_mut(mid);
            let l = build(nodes, l_ids, bb, c, leaf_max, depth + 1, max_depth);
            let r = build(nodes, r_ids, bb, c, leaf_max, depth + 1, max_depth);

            let id = nodes.len() as i32;
            nodes.push(BvhNode::new_internal(bbox, l, r));
            id
        }

        let leaf_max = 4;
        let max_depth = (2.0 * (self.tris.len().max(1) as f32).log2()).ceil() as usize + 8;

        let root = build(
            &mut self.nodes,
            &mut self.indices[..],
            &tri_bb,
            &cents,
            leaf_max,
            0,
            max_depth,
        );

        let mut packed: Vec<usize> = Vec::with_capacity(self.indices.len());
        fn assign(nodes: &mut [BvhNode], node_id: i32, src: &mut [usize], out: &mut Vec<usize>) {
            let n = &nodes[node_id as usize];
            if n.left < 0 {
                let start = out.len() as u32;
                out.extend_from_slice(src);
                let count = src.len() as u32;
                let nn = &mut nodes[node_id as usize];
                nn.start = start;
                nn.count = count;
                return;
            }

            let mid = src.len() / 2;
            let (l, r) = src.split_at_mut(mid);
            let left = n.left;
            let right = n.right;
            assign(nodes, left, l, out);
            assign(nodes, right, r, out);
        }
        assign(&mut self.nodes, root, &mut self.indices[..], &mut packed);
        self.indices = packed;
    }
}

impl Hittable for Mesh {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitInfo> {
        if self.nodes.is_empty() {
            let mut closest = t_max;
            let mut info = None;
            for tri in &self.tris {
                if let Some(h) = tri.hit(ray, t_min, closest) {
                    closest = h.t;
                    info = Some(h);
                }
            }
            return info;
        }

        let mut stack: [i32; 128] = [0; 128];
        let mut sp;

        stack[0] = (self.nodes.len() as i32) - 1;
        sp = 1;

        let mut closest = t_max;
        let mut info = None;

        while sp > 0 {
            sp -= 1;
            let id = stack[sp];
            let node = &self.nodes[id as usize];

            if !node.bbox.hit(ray, t_min, closest) {
                continue;
            }

            if node.left < 0 {
                let start = node.start as usize;
                let end = start + node.count as usize;
                for &tri_id in &self.indices[start..end] {
                    if let Some(h) = self.tris[tri_id].hit(ray, t_min, closest) {
                        closest = h.t;
                        info = Some(h);
                    }
                }
                continue;
            }

            let l = node.left;
            let r = node.right;

            let cl = self.nodes[l as usize]
                .bbox
                .centroid()
                .sub(ray.origin)
                .dot(&ray.dir);
            let cr = self.nodes[r as usize]
                .bbox
                .centroid()
                .sub(ray.origin)
                .dot(&ray.dir);

            if cl <= cr {
                stack[sp] = r;
                sp += 1;
                stack[sp] = l;
                sp += 1;
            } else {
                stack[sp] = l;
                sp += 1;
                stack[sp] = r;
                sp += 1;
            }
        }

        info
    }
}

struct HittableList {
    objects: Vec<Box<dyn Hittable>>,
}

impl HittableList {
    fn new() -> Self {
        Self { objects: vec![] }
    }

    fn add<T: Hittable + 'static>(&mut self, object: T) {
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

fn local_to_world(local: Vec3, t: Vec3, b: Vec3, n: Vec3) -> Vec3 {
    t.smul(local.x).add(b.smul(local.y)).add(n.smul(local.z))
}

fn world_to_local(v: Vec3, t: Vec3, b: Vec3, n: Vec3) -> Vec3 {
    Vec3::new(v.dot(&t), v.dot(&b), v.dot(&n))
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

#[inline]
fn offset_ray_origin(p: Vec3, n: Vec3, wi: Vec3) -> Vec3 {
    let threshold: f32 = 1.0 / 32.0;
    let float_scale = 1.0 / 65536.0;
    let int_scale = 256;

    let n = if n.dot(&wi) > 0.0 { n } else { n.smul(-1.0) };

    let bump = |p: f32, n: f32| -> f32 {
        if p.abs() < threshold {
            p + n * float_scale
        } else {
            let step = if n >= 0.0 { int_scale } else { -int_scale };
            let bits = p.to_bits() as i32;
            let shifted = if p < 0.0 { bits - step } else { bits + step };
            f32::from_bits(shifted as u32)
        }
    };

    Vec3::new(bump(p.x, n.x), bump(p.y, n.y), bump(p.z, n.z))
}

#[inline]
fn mis_power(pdf_a: f32, pdf_b: f32) -> f32 {
    let a2 = pdf_a * pdf_a;
    let b2 = pdf_b * pdf_b;
    if a2 + b2 <= 1e-16 {
        0.0
    } else {
        a2 / (a2 + b2)
    }
}

fn trace(
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    camera: &Camera,
    world: &dyn Hittable,
    light: &dyn Light,
    max_depth: u32,
    rr_start: u32,
    spp: u32,
    rng: &mut ThreadRng,
) -> Vec3 {
    let mut pixel = Vec3::zero();

    for sample in 0..spp {
        let mut radiance = Vec3::zero();
        let mut throughput = Vec3::one();

        let jx = rng.random::<f32>();
        let jy = rng.random::<f32>();

        let u = (x as f32 + jx) / width as f32;
        let v = (y as f32 + jy) / height as f32;

        let mut ray = camera.get_ray(u, v);

        let mut medium_stack = vec![Medium::new(1.0)];

        for i in 0..max_depth {
            let Some(hit) = world.hit(&ray, 0.0, f32::INFINITY) else {
                break;
            };

            let p = hit.p;
            let n = hit.normal;
            let wo = ray.dir.smul(-1.0);

            match &hit.material {
                Materials::Emissive(emissive) => {
                    let emission = emissive.emission(wo, n);
                    radiance = radiance.add(throughput.mul(emission));
                    break;
                }

                Materials::Reflective(reflective) => {
                    // ---------- Next-Event Estimation ----------
                    let ls = light.sample(p, n, rng);
                    if ls.pdf > 1e-8 {
                        let cos_i = n.dot(&ls.wi).max(0.0);
                        if cos_i > 0.0 && ls.radiance.max_comp() > 0.0 {
                            let origin = offset_ray_origin(p, n, ls.wi);
                            let shadow_ray = Ray::new(origin, ls.wi);

                            let int_scale = 256;
                            let bits = ls.dist.to_bits();
                            let shifted = bits - int_scale;
                            let t_max = f32::from_bits(shifted as u32);

                            let visible = world.hit(&shadow_ray, 0.0, t_max).is_none();

                            if visible {
                                let f = reflective.eval_brdf(ls.wi, wo, n);
                                let p_brdf = reflective.sample_pdf(ls.wi, wo, n);

                                let w = mis_power(ls.pdf, p_brdf);
                                let contrib = f.mul(ls.radiance).smul(cos_i * w / ls.pdf);
                                radiance = radiance.add(throughput.mul(contrib));
                            }
                        }
                    }

                    // ---------- BRDF Sampling ----------
                    let (wi, p_brdf, brdf) = reflective.sample_brdf(wo, n, rng);
                    if p_brdf <= 1e-8 {
                        break;
                    }

                    let cos_i = n.dot(&wi).max(0.0);
                    if cos_i == 0.0 {
                        break;
                    }

                    let offset_origin = offset_ray_origin(p, n, wi);
                    let test_ray = Ray::new(offset_origin, wi);
                    if let Some(hit) = world.hit(&test_ray, 0.0, f32::INFINITY) {
                        if let Materials::Emissive(emissive) = &hit.material {
                            let emission = emissive.emission(wi.smul(-1.0), hit.normal);
                            if emission.max_comp() > 0.0 {
                                let p_light = light.pdf(p, wi);

                                let w = mis_power(p_brdf, p_light);
                                let contrib = brdf.mul(emission).smul(cos_i * w / p_brdf);
                                radiance = radiance.add(throughput.mul(contrib));
                                break;
                            }
                        }
                    }

                    if i >= rr_start {
                        let mut q = throughput.max_comp();
                        q = q.clamp(0.05, 0.95);
                        if rng.random::<f32>() > q {
                            break;
                        }
                        throughput = throughput.smul(1.0 / q);
                    }

                    throughput = throughput.mul(brdf).smul(cos_i / p_brdf);

                    ray = Ray::new(offset_origin, wi);
                }

                Materials::Refractive(refractive) => {
                    let ior_i = medium_stack.last().unwrap().ior;
                    let ior_t = if hit.front_face {
                        refractive.ior()
                    } else if medium_stack.len() > 1 {
                        medium_stack[medium_stack.len() - 2].ior
                    } else {
                        1.0
                    };

                    let (wi, pdf, weight, transmitted) =
                        refractive.sample(wo, n, hit.front_face, ior_i, ior_t, rng);

                    if transmitted {
                        if hit.front_face {
                            medium_stack.push(Medium::new(refractive.ior()));
                        } else if medium_stack.len() > 1 {
                            medium_stack.pop();
                        }
                    }

                    throughput = throughput.mul(weight).smul(1.0 / pdf);

                    if i >= rr_start {
                        let mut q = throughput.max_comp();
                        q = q.clamp(0.05, 0.95);
                        if rng.random::<f32>() > q {
                            break;
                        }
                        throughput = throughput.smul(1.0 / q);
                    }

                    let offset_origin = offset_ray_origin(p, n, wi);
                    ray = Ray::new(offset_origin, wi);
                }
            }
        }
        pixel = pixel.add(radiance.sub(pixel).smul(1.0 / (sample as f32 + 1.0)))
    }
    pixel
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
    #[arg(short = 'w', long = "width", default_value_t = 2560)]
    width: u32,

    /// Height of the rendered image
    #[arg(short = 'h', long = "height", default_value_t = 1440)]
    height: u32,

    /// Number of samples used for the renderer
    #[arg(short = 's', long = "samples", default_value_t = 256)]
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

#[derive(Clone, Copy)]
struct Aabb {
    min: Vec3,
    max: Vec3,
}

impl Aabb {
    fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    fn union(self, o: Aabb) -> Self {
        Self {
            min: Vec3::new(
                self.min.x.min(o.min.x),
                self.min.y.min(o.min.y),
                self.min.z.min(o.min.z),
            ),
            max: Vec3::new(
                self.max.x.max(o.max.x),
                self.max.y.max(o.max.y),
                self.max.z.max(o.max.z),
            ),
        }
    }

    fn from_triangle(t: &Triangle) -> Self {
        let eps = 1e-5;
        let min = Vec3::new(
            t.p0.x.min(t.p1.x).min(t.p2.x) - eps,
            t.p0.y.min(t.p1.y).min(t.p2.y) - eps,
            t.p0.z.min(t.p1.z).min(t.p2.z) - eps,
        );
        let max = Vec3::new(
            t.p0.x.max(t.p1.x).max(t.p2.x) + eps,
            t.p0.y.max(t.p1.y).max(t.p2.y) + eps,
            t.p0.z.max(t.p1.z).max(t.p2.z) + eps,
        );
        Aabb { min, max }
    }

    fn centroid(&self) -> Vec3 {
        self.min.add(self.max).smul(0.5)
    }

    fn hit(&self, ray: &Ray, mut t_min: f32, mut t_max: f32) -> bool {
        for axis in 0..3 {
            let (o, d, mn, mx) = match axis {
                0 => (ray.origin.x, ray.dir.x, self.min.x, self.max.x),
                1 => (ray.origin.y, ray.dir.y, self.min.y, self.max.y),
                _ => (ray.origin.z, ray.dir.z, self.min.z, self.max.z),
            };
            let inv_d = 1.0 / d;
            let mut t0 = (mn - o) * inv_d;
            let mut t1 = (mx - o) * inv_d;
            if inv_d < 0.0 {
                std::mem::swap(&mut t0, &mut t1);
            }

            t_min = t_min.max(t0);
            t_max = t_max.min(t1);
            if t_max <= t_min {
                return false;
            }
        }
        true
    }
}

fn clamp01(x: f32) -> f32 {
    x.max(0.0).min(1.0)
}

fn schlick_fresnel(cos_theta: f32, f0: Vec3) -> Vec3 {
    f0.add(Vec3::one().sub(f0).smul((1.0 - clamp01(cos_theta)).powi(5)))
}

fn refract(v: Vec3, n: Vec3, eta: f32) -> Option<Vec3> {
    let n = n.norm();
    let cos_i = (v.smul(-1.0)).dot(&n).clamp(-1.0, 1.0);

    let sin2_t = eta * eta * (1.0 - cos_i * cos_i);
    if sin2_t > 1.0 {
        return None;
    }

    let cos_t = (1.0 - sin2_t).sqrt();

    let refracted = v.smul(eta).add(n.smul(eta * cos_i - cos_t));
    Some(refracted.norm())
}

fn f0_from_ior(ior_i: f32, ior_t: f32) -> f32 {
    let r0 = (ior_i - ior_t) / (ior_i + ior_t);
    r0 * r0
}

fn reflect(v: Vec3, n: Vec3) -> Vec3 {
    v.sub(n.smul(2.0 * v.dot(&n)))
}

fn srgb_to_linear_u8(r: u32, g: u32, b: u32) -> Vec3 {
    fn convert(c: f32) -> f32 {
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    }
    Vec3::new(
        convert(r as f32 / 255.0),
        convert(g as f32 / 255.0),
        convert(b as f32 / 255.0),
    )
}

fn blackbody(kelvin: f64) -> Vec3 {
    let t = kelvin.clamp(1667.0, 25000.0);

    let x = if t <= 4000.0 {
        (-0.2661239e9 / (t * t * t)) - (0.2343580e6 / (t * t)) + (0.8776956e3 / t) + 0.179910
    } else {
        (-3.0258469e9 / (t * t * t)) + (2.1070379e6 / (t * t)) + (0.2226347e3 / t) + 0.240390
    };

    let y = if t <= 2222.0 {
        -1.1063814 * x * x * x - 1.3481102 * x * x + 2.18555832 * x - 0.20219683
    } else if t <= 4000.0 {
        -0.9549476 * x * x * x - 1.37418593 * x * x + 2.09137015 * x - 0.16748867
    } else {
        3.0817580 * x * x * x - 5.87338670 * x * x + 3.75112997 * x - 0.37001483
    };

    let y_lum = 1.0;
    let x_tristim = (x / y) * y_lum;
    let z_tristim = ((1.0 - x - y) / y) * y_lum;

    let r = 3.2406 * x_tristim - 1.5372 * y_lum - 0.4986 * z_tristim;
    let g = -0.9689 * x_tristim + 1.8758 * y_lum + 0.0415 * z_tristim;
    let b = 0.0557 * x_tristim - 0.2040 * y_lum + 1.0570 * z_tristim;

    let max = r.max(g).max(b);
    Vec3::new((r / max) as f32, (g / max) as f32, (b / max) as f32)
}

fn main() {
    let args = Args::parse();

    let width = args.width;
    let height = args.height;

    let exposure = args.exposure;
    let exposure_scale = 2.0f32.powf(exposure);

    let camera = Camera::from_resolution(
        24.0,                         // Focal length
        24.0,                         // Sensor size
        width,                        // Image width
        height,                       // Image height
        Vec3::new(0.0, -2.197, 0.76), // Look from
        Vec3::new(0.0, 0.0, 0.76),    // Look at
        Vec3::new(0.0, 0.0, 1.0),     // Up (0.0, 0.0, 1.0) for Z-up
    );

    let material1 = Arc::new(Lambert::new(srgb_to_linear_u8(222, 50, 50))); // Red
    let material2 = Arc::new(Lambert::new(srgb_to_linear_u8(94, 199, 85))); // Green
    let material3 = Arc::new(Lambert::new(Vec3::new(0.725, 0.71, 0.68))); // White
    let material4 = Arc::new(Lambert::new(srgb_to_linear_u8(101, 146, 237))); // Blue
    let material5 = Arc::new(Glass::new(1.6));

    let mut mesh = Mesh::from_obj(
        "stanford_dragon_remeshed.obj",
        Materials::Refractive(material5.clone()),
        Vec3::new(-0.05, 0.0, 0.395),
        Vec3::new(0.75, 0.75, 0.75),
    );
    mesh.build_bvh();

    let (min_depth, max_depth) = mesh.bvh_depth();
    println!("BVH node count: {}", mesh.nodes.len());
    println!("BVH min depth: {}", min_depth);
    println!("BVH max depth: {}", max_depth);
    println!("Mesh triangle count: {}", mesh.tris.len());

    let light = Quad::emissive_from_power(
        Vec3::new(0.000, 0.000, 1.523),
        Vec3::new(0.2231081 / 1.5, 0.000, 0.000),
        Vec3::new(0.000, 0.1802027 / 1.5, 0.000),
        blackbody(4500.0),
        7.0,
    );

    // let sphere1 = Sphere::new(
    //     Vec3::new(0.0, 0.0, 0.6),
    //     0.3,
    //     Materials::Refractive(material5.clone()),
    // );

    let quad1 = Quad::new(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.762, 0.0, 0.0),
        Vec3::new(0.0, 0.762, 0.0),
        Materials::Reflective(material3.clone()),
    );
    let quad2 = Quad::new(
        Vec3::new(-0.762, 0.0, 0.762),
        Vec3::new(0.000, 0.762, 0.000),
        Vec3::new(0.000, 0.000, 0.762),
        Materials::Reflective(material1.clone()),
    );
    let quad3 = Quad::new(
        Vec3::new(0.762, 0.000, 0.762),
        Vec3::new(0.000, 0.762, 0.000),
        Vec3::new(0.000, 0.000, 0.762),
        Materials::Reflective(material2.clone()),
    );
    let quad4 = Quad::new(
        Vec3::new(0.000, 0.762, 0.762),
        Vec3::new(0.762, 0.000, 0.000),
        Vec3::new(0.000, 0.000, 0.762),
        Materials::Reflective(material4.clone()),
    );
    let quad5 = Quad::new(
        Vec3::new(0.000, 0.000, 1.524),
        Vec3::new(0.762, 0.000, 0.000),
        Vec3::new(0.000, 0.762, 0.000),
        Materials::Reflective(material3.clone()),
    );

    let mut hl = HittableList::new();
    hl.add(mesh);
    // hl.add(sphere1);
    hl.add(light.clone());
    hl.add(quad1);
    hl.add(quad2);
    hl.add(quad3);
    hl.add(quad4);
    hl.add(quad5);

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
                    let mut color = trace(
                        x,
                        y,
                        width,
                        height,
                        &camera,
                        &hl,
                        &light,
                        16,
                        6,
                        args.samples,
                        rng,
                    );
                    color = color.smul(exposure_scale);
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
