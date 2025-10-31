#![allow(dead_code)]

use sobol::{Sobol, params::JoeKuoD6};

#[inline]
fn mix32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

#[inline]
fn reverse_bits32(x: u32) -> u32 {
    x.reverse_bits()
}

// #[inline]
// fn f32_to_u32(u: f32) -> u32 {
//     // (u.clamp(0.0, 0.99999994) * 4294967296.0) as u32
//     let mut x = (u * 4294967296.0) as u64;
//     x = x.min(0xFFFF_FFFF);
//     x as u32
//     // let x = (u * 4294967295.0).to_bits();
//     // x & 0x7FFF_FFFF
// }
#[inline]
fn u32_to_f32(x: u32) -> f32 {
    ((x as f32) + 0.5) * (1.0 / 4294967296.0)
}

#[inline]
fn true_owen(mut u: u32, key: u32, dim: u32) -> f32 {
    let mut prefix: u32 = 0;

    for bit in (0..32).rev() {
        let current_bit = (u >> bit) & 1;
        let h = mix32(key ^ dim.wrapping_mul(0x9e37_79b9) ^ prefix ^ (bit as u32).rotate_left(13));
        let flip = h & 1;
        let new_bit = current_bit ^ flip;
        u = (u & !(1 << bit)) | (new_bit << bit);
        prefix = (prefix << 1) | new_bit;
    }
    u32_to_f32(u)
}

// From https://psychopath.io/post/2021_01_30_building_a_better_lk_hash
#[inline]
fn lk_mix(mut n: u32, seed: u32) -> u32 {
    n ^= n.wrapping_mul(0x3d20_adea);
    n = n.wrapping_add(seed);
    n = n.wrapping_mul((seed >> 16) | 1);
    n ^= n.wrapping_mul(0x0552_6c56);
    n ^= n.wrapping_mul(0x53a2_2864);
    n
}

#[inline]
fn fast_owen_lk(u: u32, key: u32, dim: u32) -> f32 {
    let seed = key ^ dim.wrapping_mul(0x9e37_79b9);
    let y = reverse_bits32(lk_mix(reverse_bits32(u), seed));
    u32_to_f32(y)
}

#[derive(Clone, Copy)]
pub struct SobolLayout {
    cam_jx: usize,
    cam_jy: usize,
    cam_count: usize,

    bounce_stride: usize,
    off_dir_u: usize,
    off_dir_v: usize,
    off_lobe: usize,
    off_rr: usize,
    off_nee_u: usize,
    off_nee_v: usize,

    max_depth: usize,
}

impl SobolLayout {
    pub const fn new(max_depth: usize) -> Self {
        let cam_jx = 0;
        let cam_jy = 1;
        let cam_count = 2;

        let bounce_stride = 6;
        let off_dir_u = 0;
        let off_dir_v = 1;
        let off_lobe = 2;
        let off_rr = 3;
        let off_nee_u = 4;
        let off_nee_v = 5;

        Self {
            cam_jx,
            cam_jy,
            cam_count,
            bounce_stride,
            off_dir_u,
            off_dir_v,
            off_lobe,
            off_nee_u,
            off_nee_v,
            off_rr,
            max_depth,
        }
    }

    #[inline]
    pub const fn total_dims(&self) -> usize {
        self.cam_count + self.bounce_stride * self.max_depth
    }
    #[inline]
    pub const fn base(&self, b: usize) -> usize {
        self.cam_count + b * self.bounce_stride
    }

    #[inline]
    pub const fn cam_jx(&self) -> usize {
        self.cam_jx
    }
    #[inline]
    pub const fn cam_jy(&self) -> usize {
        self.cam_jy
    }

    #[inline]
    pub const fn dir_u(&self, b: usize) -> usize {
        self.base(b) + self.off_dir_u
    }
    #[inline]
    pub const fn dir_v(&self, b: usize) -> usize {
        self.base(b) + self.off_dir_v
    }
    #[inline]
    pub const fn lobe(&self, b: usize) -> usize {
        self.base(b) + self.off_lobe
    }
    #[inline]
    pub const fn rr(&self, b: usize) -> usize {
        self.base(b) + self.off_rr
    }
    #[inline]
    pub const fn nee_u(&self, b: usize) -> usize {
        self.base(b) + self.off_nee_u
    }
    #[inline]
    pub const fn nee_v(&self, b: usize) -> usize {
        self.base(b) + self.off_nee_v
    }
}

pub struct SobolTable {
    dims: usize,
    spp: usize,
    data: Vec<u32>,
}

impl SobolTable {
    pub fn new(dims: usize, spp: usize) -> Self {
        let mut sobol = Sobol::<u32>::new(dims, &JoeKuoD6::extended());
        let mut buf = vec![0u32; dims * spp];
        for s in 0..spp {
            let row = sobol.next().unwrap();
            buf[s * dims..(s + 1) * dims].copy_from_slice(&row);
        }
        buf.shrink_to_fit();
        SobolTable {
            dims,
            spp,
            data: buf,
        }
    }
}

pub struct SobolSampler<'a> {
    table: &'a SobolTable,
    key: u32,
    layout: SobolLayout,
    sample_index: usize,
}

impl<'a> SobolSampler<'a> {
    pub const fn new(key: u32, layout: SobolLayout, table: &'a SobolTable) -> Self {
        Self {
            table,
            key,
            layout,
            sample_index: 0,
        }
    }

    #[inline]
    pub fn next_sample(&mut self) {
        self.sample_index += 1;
    }
    #[inline]
    pub fn fetch(&self, dim: usize) -> u32 {
        self.table.data[self.sample_index * self.table.dims + dim]
    }
    #[inline]
    pub fn at(&self, dim: usize) -> f32 {
        fast_owen_lk(self.fetch(dim), self.key, dim as u32)
    }

    #[inline]
    pub fn cam_jitter(&self) -> (f32, f32) {
        (self.at(self.layout.cam_jx()), self.at(self.layout.cam_jy()))
    }
    #[inline]
    pub fn bsdf(&self, b: usize) -> (f32, f32) {
        (self.at(self.layout.dir_u(b)), self.at(self.layout.dir_v(b)))
    }
    #[inline]
    pub fn lobe(&self, b: usize) -> f32 {
        self.at(self.layout.lobe(b))
    }
    #[inline]
    pub fn rr(&self, b: usize) -> f32 {
        self.at(self.layout.rr(b))
    }
    #[inline]
    pub fn nee(&self, b: usize) -> (f32, f32) {
        (self.at(self.layout.nee_u(b)), self.at(self.layout.nee_v(b)))
    }
}
