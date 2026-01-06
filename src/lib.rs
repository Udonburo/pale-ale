use once_cell::sync::Lazy;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::f64::consts::PI;

const ROOT_DIM: usize = 8;
const SEMANTIC_EPS: f64 = 1e-9;
const BLOCK_EPS: f64 = 1e-12;
const BIVECTOR_DIM: usize = ROOT_DIM * (ROOT_DIM - 1) / 2;
const BIVECTOR_EPS: f64 = 1e-12;

static E8_ROOTS: Lazy<Vec<[f64; ROOT_DIM]>> = Lazy::new(|| {
    let mut roots = Vec::with_capacity(240);
    let inv_sqrt2 = 1.0 / (2.0_f64).sqrt();
    let signs = [-1.0, 1.0];

    for i in 0..ROOT_DIM {
        for j in (i + 1)..ROOT_DIM {
            for &si in signs.iter() {
                for &sj in signs.iter() {
                    let mut v = [0.0; ROOT_DIM];
                    v[i] = si * inv_sqrt2;
                    v[j] = sj * inv_sqrt2;
                    roots.push(v);
                }
            }
        }
    }

    for mask in 0..(1 << ROOT_DIM) {
        let mut neg_count = 0;
        for bit in 0..ROOT_DIM {
            if (mask >> bit) & 1 == 1 {
                neg_count += 1;
            }
        }
        if neg_count % 2 != 0 {
            continue;
        }

        let mut v = [0.0; ROOT_DIM];
        for bit in 0..ROOT_DIM {
            let sign = if (mask >> bit) & 1 == 1 { -0.5 } else { 0.5 };
            v[bit] = sign * inv_sqrt2;
        }
        roots.push(v);
    }

    debug_assert_eq!(roots.len(), 240);
    roots
});

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
fn dot8(a: &[f64; ROOT_DIM], b: &[f64; ROOT_DIM]) -> f64 {
    let mut sum = 0.0;
    for i in 0..ROOT_DIM {
        sum += a[i] * b[i];
    }
    sum
}

#[inline]
fn norm8(a: &[f64; ROOT_DIM]) -> f64 {
    dot8(a, a).sqrt()
}

fn wedge8(u: &[f64; ROOT_DIM], v: &[f64; ROOT_DIM]) -> [f64; BIVECTOR_DIM] {
    let mut res = [0.0; BIVECTOR_DIM];
    let mut k = 0;
    for i in 0..ROOT_DIM {
        for j in (i + 1)..ROOT_DIM {
            res[k] = u[i] * v[j] - u[j] * v[i];
            k += 1;
        }
    }
    res
}

#[inline]
fn bivector_dot(a: &[f64; BIVECTOR_DIM], b: &[f64; BIVECTOR_DIM]) -> f64 {
    let mut sum = 0.0;
    for i in 0..BIVECTOR_DIM {
        sum += a[i] * b[i];
    }
    sum
}

#[inline]
fn bivector_norm(a: &[f64; BIVECTOR_DIM]) -> f64 {
    bivector_dot(a, a).sqrt()
}

fn bivector_normalize(
    a: &[f64; BIVECTOR_DIM],
    eps: f64,
) -> Option<[f64; BIVECTOR_DIM]> {
    let n = bivector_norm(a);
    if n < eps {
        return None;
    }
    let inv = 1.0 / n;
    let mut out = [0.0; BIVECTOR_DIM];
    for i in 0..BIVECTOR_DIM {
        out[i] = a[i] * inv;
    }
    Some(out)
}

fn bivector_angle_dist(a: &[f64; BIVECTOR_DIM], b: &[f64; BIVECTOR_DIM]) -> f64 {
    let dot = clamp(bivector_dot(a, b), -1.0, 1.0);
    let wedge_sq = 1.0 - dot * dot;
    let wedge_norm = if wedge_sq <= 0.0 { 0.0 } else { wedge_sq.sqrt() };
    let theta = wedge_norm.atan2(dot);
    theta / PI
}

fn normalize_block(slice: &[f64]) -> Option<[f64; ROOT_DIM]> {
    if slice.len() != ROOT_DIM {
        return None;
    }
    let mut v = [0.0; ROOT_DIM];
    v.copy_from_slice(slice);
    let n = norm8(&v);
    if n < BLOCK_EPS {
        return None;
    }
    let inv = 1.0 / n;
    for x in v.iter_mut() {
        *x *= inv;
    }
    Some(v)
}

fn snap_e8(a: &[f64; ROOT_DIM]) -> [f64; ROOT_DIM] {
    let mut best = E8_ROOTS[0];
    let mut best_dot = dot8(a, &best);
    for root in E8_ROOTS.iter().skip(1) {
        let d = dot8(a, root);
        if d > best_dot {
            best_dot = d;
            best = *root;
        }
    }
    best
}

fn rotor_distance(a: &[f64; ROOT_DIM], b: &[f64; ROOT_DIM]) -> f64 {
    let dot = clamp(dot8(a, b), -1.0, 1.0);
    let wedge_sq = 1.0 - dot * dot;
    let wedge_norm = if wedge_sq <= 0.0 { 0.0 } else { wedge_sq.sqrt() };
    let theta = wedge_norm.atan2(dot);
    theta / PI
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
    if dim % ROOT_DIM != 0 {
        return Err(PyValueError::new_err(
            "vector length must be a multiple of 8",
        ));
    }

    let alpha_weight = clamp(alpha.unwrap_or(0.15), 0.0, 1.0);

    let nu = norm(&u);
    let nv = norm(&v);
    let semantic_dist = if nu < SEMANTIC_EPS || nv < SEMANTIC_EPS {
        1.0
    } else {
        let sim = clamp(dot(&u, &v) / (nu * nv), -1.0, 1.0);
        0.5 * (1.0 - sim)
    };

    let blocks = dim / ROOT_DIM;
    let mut u_blocks = Vec::with_capacity(blocks);
    let mut v_blocks = Vec::with_capacity(blocks);
    for block in 0..blocks {
        let start = block * ROOT_DIM;
        u_blocks.push(normalize_block(&u[start..start + ROOT_DIM]));
        v_blocks.push(normalize_block(&v[start..start + ROOT_DIM]));
    }

    let mut intra_sum = 0.0;
    let mut intra_count = 0usize;
    for block in 0..blocks {
        let (Some(u_block), Some(v_block)) = (u_blocks[block], v_blocks[block]) else {
            continue;
        };

        let r_u = snap_e8(&u_block);
        let r_v = snap_e8(&v_block);

        let d_root = rotor_distance(&r_u, &r_v);
        let d_cont = rotor_distance(&u_block, &v_block);
        let d_snap = 0.5 * (rotor_distance(&u_block, &r_u) + rotor_distance(&v_block, &r_v));
        let d_struct = 0.60 * d_root + 0.30 * d_cont + 0.10 * d_snap;

        intra_sum += d_struct;
        intra_count += 1;
    }

    let d_intra = if intra_count > 0 {
        intra_sum / (intra_count as f64)
    } else {
        0.0
    };

    let mut inter_sum = 0.0;
    let mut inter_count = 0usize;
    if blocks > 1 {
        for j in 0..(blocks - 1) {
            let (Some(a_j), Some(a_j1)) = (u_blocks[j], u_blocks[j + 1]) else {
                continue;
            };
            let (Some(b_j), Some(b_j1)) = (v_blocks[j], v_blocks[j + 1]) else {
                continue;
            };

            let bu = wedge8(&a_j, &a_j1);
            let bv = wedge8(&b_j, &b_j1);
            let d_cont = match (
                bivector_normalize(&bu, BIVECTOR_EPS),
                bivector_normalize(&bv, BIVECTOR_EPS),
            ) {
                (None, None) => 0.0,
                (None, Some(_)) | (Some(_), None) => 1.0,
                (Some(bu_n), Some(bv_n)) => bivector_angle_dist(&bu_n, &bv_n),
            };

            let ru_j = snap_e8(&a_j);
            let ru_j1 = snap_e8(&a_j1);
            let rv_j = snap_e8(&b_j);
            let rv_j1 = snap_e8(&b_j1);
            let bru = wedge8(&ru_j, &ru_j1);
            let brv = wedge8(&rv_j, &rv_j1);
            let d_root = match (
                bivector_normalize(&bru, BIVECTOR_EPS),
                bivector_normalize(&brv, BIVECTOR_EPS),
            ) {
                (None, None) => 0.0,
                (None, Some(_)) | (Some(_), None) => 1.0,
                (Some(bru_n), Some(brv_n)) => bivector_angle_dist(&bru_n, &brv_n),
            };

            let d_inter = 0.4 * d_root + 0.6 * d_cont;
            inter_sum += d_inter;
            inter_count += 1;
        }
    }

    let d_inter = if inter_count > 0 {
        inter_sum / (inter_count as f64)
    } else {
        0.0
    };

    let d_struct = 0.6 * d_intra + 0.4 * d_inter;
    let d = (1.0 - alpha_weight) * semantic_dist + alpha_weight * d_struct;
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
    fn symmetry() {
        let u: Vec<f64> = (0..16).map(|i| (i as f64 + 1.0).sin()).collect();
        let v: Vec<f64> = (0..16).map(|i| (i as f64 + 1.0).cos()).collect();
        let d1 = spin3_distance(u.clone(), v.clone(), Some(0.25)).unwrap();
        let d2 = spin3_distance(v, u, Some(0.25)).unwrap();
        assert!((d1 - d2).abs() < 1e-12);
    }

    #[test]
    fn bounds_in_unit_interval() {
        let mut u = vec![0.0; 16];
        let mut v = vec![0.0; 16];
        for i in 0..16 {
            u[i] = (i as f64).sin();
            v[i] = (i as f64).cos();
        }
        let d = spin3_distance(u, v, Some(0.10)).unwrap();
        assert!(d >= 0.0 && d <= 1.0);
    }

    #[test]
    fn identical_vectors_near_zero() {
        let u = vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let d = spin3_distance(u.clone(), u, Some(0.5)).unwrap();
        assert!(d >= 0.0 && d <= 1.0);
        assert!(d.abs() < 1e-8);
    }

    #[test]
    fn opposite_vectors_near_one() {
        let u = vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v: Vec<f64> = u.iter().map(|x| -x).collect();
        let d = spin3_distance(u, v, None).unwrap();
        assert!(d >= 0.90 && d <= 1.0);
    }

    #[test]
    fn invalid_length_errors() {
        let u = vec![0.1; 10];
        let v = vec![0.1; 10];
        assert!(spin3_distance(u, v, None).is_err());
    }

    #[test]
    fn block_shuffle_sensitivity() {
        let blocks = 48;
        let mut u = Vec::with_capacity(blocks * ROOT_DIM);
        for block in 0..blocks {
            for i in 0..ROOT_DIM {
                let seed = (block * 37 + i * 19 + block * i * 3 + 11) % 251;
                let val = (seed as f64 - 125.0) / 50.0;
                u.push(val);
            }
        }

        let mut u_shift = vec![0.0; u.len()];
        for block in 0..blocks {
            let src = block * ROOT_DIM;
            let dst = ((block + 1) % blocks) * ROOT_DIM;
            u_shift[dst..dst + ROOT_DIM].copy_from_slice(&u[src..src + ROOT_DIM]);
        }

        let d = spin3_distance(u, u_shift, Some(1.0)).unwrap();
        assert!(d > 0.15);
    }
}
