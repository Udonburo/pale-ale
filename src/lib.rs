use once_cell::sync::Lazy;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::Mutex;

const PROJ_SEED: u64 = 0x1337_C0DE_CAFE_BABE;
const K_VIEWS: usize = 4;
const AXIS_EPS: f64 = 1e-12;
const AXIS_RETRIES: usize = 8;
const SEMANTIC_EPS: f64 = 1e-9;
const VIEW_EPS: f64 = 1e-12;

#[derive(Clone)]
struct ViewProjection {
    r0: Vec<f64>,
    r1: Vec<f64>,
    r2: Vec<f64>,
    valid: bool,
}

#[derive(Clone)]
struct MultiViewProjection {
    views: Vec<ViewProjection>,
}

static PROJ_CACHE: Lazy<Mutex<HashMap<usize, MultiViewProjection>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        let v = self.next_u64();
        let u = (v >> 11) as f64;
        u / ((1u64 << 53) as f64)
    }

    fn next_f64_signed(&mut self) -> f64 {
        (self.next_f64() * 2.0) - 1.0
    }
}

#[inline]
fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline]
fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

#[inline]
fn norm3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn axpy(v: &mut [f64], a: f64, x: &[f64]) {
    for (vi, xi) in v.iter_mut().zip(x.iter()) {
        *vi += a * xi;
    }
}

fn fill_random(v: &mut [f64], rng: &mut SplitMix64) {
    for x in v.iter_mut() {
        *x = rng.next_f64_signed();
    }
}

fn orthonormalize_axis(
    n: usize,
    rng: &mut SplitMix64,
    out: &mut Vec<f64>,
    basis: &[&[f64]],
) -> bool {
    if out.len() != n {
        out.resize(n, 0.0);
    }
    for _ in 0..AXIS_RETRIES {
        fill_random(out, rng);
        for b in basis {
            let proj = dot(out, b);
            axpy(out, -proj, b);
        }
        let nrm = norm(out);
        if nrm >= AXIS_EPS {
            for x in out.iter_mut() {
                *x /= nrm;
            }
            return true;
        }
    }
    false
}

fn build_view(n: usize, rng: &mut SplitMix64) -> ViewProjection {
    let mut r0 = Vec::new();
    let mut r1 = Vec::new();
    let mut r2 = Vec::new();
    let mut valid = true;

    if !orthonormalize_axis(n, rng, &mut r0, &[]) {
        valid = false;
    }
    if valid && !orthonormalize_axis(n, rng, &mut r1, &[&r0]) {
        valid = false;
    }
    if valid && !orthonormalize_axis(n, rng, &mut r2, &[&r0, &r1]) {
        valid = false;
    }

    if !valid {
        if r0.len() != n {
            r0.resize(n, 0.0);
        }
        if r1.len() != n {
            r1.resize(n, 0.0);
        }
        if r2.len() != n {
            r2.resize(n, 0.0);
        }
    }

    ViewProjection { r0, r1, r2, valid }
}

fn build_projection(n: usize) -> MultiViewProjection {
    let seed = PROJ_SEED ^ (n as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut rng = SplitMix64::new(seed);
    let mut views = Vec::with_capacity(K_VIEWS);
    for _ in 0..K_VIEWS {
        views.push(build_view(n, &mut rng));
    }
    MultiViewProjection { views }
}

fn get_projection(n: usize) -> MultiViewProjection {
    if let Ok(cache) = PROJ_CACHE.lock() {
        if let Some(p) = cache.get(&n) {
            return p.clone();
        }
    }

    let proj = build_projection(n);

    if let Ok(mut cache) = PROJ_CACHE.lock() {
        cache.insert(n, proj.clone());
    }
    proj
}

fn project3(view: &ViewProjection, x: &[f64]) -> [f64; 3] {
    [dot(&view.r0, x), dot(&view.r1, x), dot(&view.r2, x)]
}

#[pyfunction]
fn spin3_distance(u: Vec<f64>, v: Vec<f64>, alpha: Option<f64>) -> PyResult<f64> {
    if u.len() != v.len() {
        return Err(PyValueError::new_err("u and v must have the same length"));
    }
    let dim = u.len();
    if dim == 0 {
        return Ok(0.0);
    }

    let a = clamp(alpha.unwrap_or(0.15), 0.0, 1.0);

    let nu = norm(&u);
    let nv = norm(&v);
    let semantic_dist = if nu < SEMANTIC_EPS || nv < SEMANTIC_EPS {
        1.0
    } else {
        let sim = clamp(dot(&u, &v) / (nu * nv), -1.0, 1.0);
        0.5 * (1.0 - sim)
    };

    let proj = get_projection(dim);
    let mut sum = 0.0;
    let mut count = 0usize;

    for view in proj.views.iter() {
        if !view.valid {
            continue;
        }
        let u3 = project3(view, &u);
        let v3 = project3(view, &v);
        let nu3 = norm3(u3);
        let nv3 = norm3(v3);
        let denom = nu3 * nv3;
        if denom < VIEW_EPS {
            continue;
        }

        let dot3 = u3[0] * v3[0] + u3[1] * v3[1] + u3[2] * v3[2];
        let cross0 = u3[1] * v3[2] - u3[2] * v3[1];
        let cross1 = u3[2] * v3[0] - u3[0] * v3[2];
        let cross2 = u3[0] * v3[1] - u3[1] * v3[0];
        let cross_norm = (cross0 * cross0 + cross1 * cross1 + cross2 * cross2).sqrt();

        let dot_hat = dot3 / denom;
        let cross_hat = cross_norm / denom;
        let theta = cross_hat.atan2(dot_hat);
        let rotor_k = theta / PI;

        sum += rotor_k;
        count += 1;
    }

    let structural_dist = if count > 0 {
        sum / (count as f64)
    } else {
        0.0
    };

    let d = (1.0 - a) * semantic_dist + a * structural_dist;
    Ok(clamp(d, 0.0, 1.0))
}

#[pymodule]
fn pale_ale_core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(spin3_distance, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_is_zeroish() {
        let u = vec![1.0, 2.0, 3.0, 4.0];
        let d = spin3_distance(u.clone(), u, Some(0.1)).unwrap();
        assert!(d >= 0.0 && d <= 1.0);
        assert!(d.abs() < 1e-12);
    }

    #[test]
    fn symmetry() {
        let u = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let v = vec![0.5, 0.4, 0.3, 0.2, 0.1];
        let d1 = spin3_distance(u.clone(), v.clone(), Some(0.25)).unwrap();
        let d2 = spin3_distance(v, u, Some(0.25)).unwrap();
        assert!((d1 - d2).abs() < 1e-12);
    }

    #[test]
    fn bounds_for_384() {
        let mut u = vec![0.0; 384];
        let mut v = vec![0.0; 384];
        for i in 0..384 {
            u[i] = (i as f64).sin();
            v[i] = (i as f64).cos();
        }
        let d = spin3_distance(u, v, Some(0.10)).unwrap();
        assert!(d >= 0.0 && d <= 1.0);
    }
}
