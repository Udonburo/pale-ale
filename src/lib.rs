use once_cell::sync::Lazy;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::f64::consts::PI;

const ROOT_DIM: usize = 8;
const BIV_DIM: usize = ROOT_DIM * (ROOT_DIM - 1) / 2;
const ROTOR_DIM: usize = 1 + BIV_DIM;
const EPS_NORM: f64 = 1e-12;
const EPS_BIV: f64 = 1e-12;
const SEMANTIC_EPS: f64 = 1e-9;

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

fn normalize8(x: &[f64; ROOT_DIM]) -> Option<[f64; ROOT_DIM]> {
    let n = norm8(x);
    if n < EPS_NORM {
        return None;
    }
    let inv = 1.0 / n;
    let mut out = [0.0; ROOT_DIM];
    for i in 0..ROOT_DIM {
        out[i] = x[i] * inv;
    }
    Some(out)
}

fn wedge8(u: &[f64; ROOT_DIM], v: &[f64; ROOT_DIM]) -> [f64; BIV_DIM] {
    let mut res = [0.0; BIV_DIM];
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
fn biv_dot(a: &[f64; BIV_DIM], b: &[f64; BIV_DIM]) -> f64 {
    let mut sum = 0.0;
    for i in 0..BIV_DIM {
        sum += a[i] * b[i];
    }
    sum
}

#[inline]
fn biv_norm(a: &[f64; BIV_DIM]) -> f64 {
    biv_dot(a, a).sqrt()
}

fn biv_normalize(a: &[f64; BIV_DIM], eps: f64) -> Option<[f64; BIV_DIM]> {
    let n = biv_norm(a);
    if n < eps {
        return None;
    }
    let inv = 1.0 / n;
    let mut out = [0.0; BIV_DIM];
    for i in 0..BIV_DIM {
        out[i] = a[i] * inv;
    }
    Some(out)
}

fn biv_angle_dist(a: &[f64; BIV_DIM], b: &[f64; BIV_DIM]) -> f64 {
    let dot = clamp(biv_dot(a, b), -1.0, 1.0);
    let wedge_sq = 1.0 - dot * dot;
    let wedge_norm = if wedge_sq <= 0.0 { 0.0 } else { wedge_sq.sqrt() };
    let theta = wedge_norm.atan2(dot);
    theta / PI
}

#[inline]
fn dot29(a: &[f64; ROTOR_DIM], b: &[f64; ROTOR_DIM]) -> f64 {
    let mut sum = 0.0;
    for i in 0..ROTOR_DIM {
        sum += a[i] * b[i];
    }
    sum
}

#[inline]
fn norm29(a: &[f64; ROTOR_DIM]) -> f64 {
    dot29(a, a).sqrt()
}

fn normalize29(a: &[f64; ROTOR_DIM]) -> [f64; ROTOR_DIM] {
    let n = norm29(a).max(EPS_NORM);
    let inv = 1.0 / n;
    let mut out = [0.0; ROTOR_DIM];
    for i in 0..ROTOR_DIM {
        out[i] = a[i] * inv;
    }
    out
}

fn to_rotor29(u: &[f64; ROOT_DIM], v: &[f64; ROOT_DIM]) -> [f64; ROTOR_DIM] {
    let c = clamp(dot8(u, v), -1.0, 1.0);
    let b = wedge8(u, v);
    if biv_norm(&b) < EPS_BIV {
        let mut out = [0.0; ROTOR_DIM];
        out[0] = 1.0;
        return out;
    }

    let mut out = [0.0; ROTOR_DIM];
    out[0] = c;
    for i in 0..BIV_DIM {
        out[i + 1] = b[i];
    }
    let n = norm29(&out).max(EPS_NORM);
    let inv = 1.0 / n;
    for i in 0..ROTOR_DIM {
        out[i] *= inv;
    }
    out
}

fn rotor_dist_29(a: &[f64; ROTOR_DIM], b: &[f64; ROTOR_DIM]) -> f64 {
    let a_n = normalize29(a);
    let b_n = normalize29(b);
    let dot = clamp(dot29(&a_n, &b_n), -1.0, 1.0);
    let wedge_sq = 1.0 - dot * dot;
    let wedge_norm = if wedge_sq <= 0.0 { 0.0 } else { wedge_sq.sqrt() };
    let theta = wedge_norm.atan2(dot);
    theta / PI
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

fn hct_distance(
    u_blocks: &[Option<[f64; ROOT_DIM]>],
    v_blocks: &[Option<[f64; ROOT_DIM]>],
) -> f64 {
    let blocks = u_blocks.len().min(v_blocks.len());
    if blocks < 2 {
        return 0.0;
    }

    let mut cont_sum = 0.0;
    let mut cont_count = 0usize;
    let mut root_sum = 0.0;
    let mut root_count = 0usize;

    let mut prev_cont: Option<f64> = None;
    let mut prev_root: Option<f64> = None;

    for j in 0..(blocks - 1) {
        let (Some(u_j), Some(u_j1)) = (u_blocks[j], u_blocks[j + 1]) else {
            prev_cont = None;
            prev_root = None;
            continue;
        };
        let (Some(v_j), Some(v_j1)) = (v_blocks[j], v_blocks[j + 1]) else {
            prev_cont = None;
            prev_root = None;
            continue;
        };

        let qu = to_rotor29(&u_j, &u_j1);
        let qv = to_rotor29(&v_j, &v_j1);
        let m_cont = rotor_dist_29(&qu, &qv);

        let ru_j = snap_e8(&u_j);
        let ru_j1 = snap_e8(&u_j1);
        let rv_j = snap_e8(&v_j);
        let rv_j1 = snap_e8(&v_j1);
        let qu_root = to_rotor29(&ru_j, &ru_j1);
        let qv_root = to_rotor29(&rv_j, &rv_j1);
        let m_root = rotor_dist_29(&qu_root, &qv_root);

        if let Some(prev) = prev_cont {
            cont_sum += (m_cont - prev).abs();
            cont_count += 1;
        }
        if let Some(prev) = prev_root {
            root_sum += (m_root - prev).abs();
            root_count += 1;
        }

        prev_cont = Some(m_cont);
        prev_root = Some(m_root);
    }

    let d_cont = if cont_count > 0 {
        cont_sum / cont_count as f64
    } else {
        0.0
    };
    let d_root = if root_count > 0 {
        root_sum / root_count as f64
    } else {
        0.0
    };

    if cont_count == 0 && root_count == 0 {
        0.0
    } else {
        0.6 * d_cont + 0.4 * d_root
    }
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
        let mut u_block = [0.0; ROOT_DIM];
        let mut v_block = [0.0; ROOT_DIM];
        u_block.copy_from_slice(&u[start..start + ROOT_DIM]);
        v_block.copy_from_slice(&v[start..start + ROOT_DIM]);
        u_blocks.push(normalize8(&u_block));
        v_blocks.push(normalize8(&v_block));
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
                biv_normalize(&bu, EPS_BIV),
                biv_normalize(&bv, EPS_BIV),
            ) {
                (None, None) => 0.0,
                (None, Some(_)) | (Some(_), None) => 1.0,
                (Some(bu_n), Some(bv_n)) => biv_angle_dist(&bu_n, &bv_n),
            };

            let ru_j = snap_e8(&a_j);
            let ru_j1 = snap_e8(&a_j1);
            let rv_j = snap_e8(&b_j);
            let rv_j1 = snap_e8(&b_j1);
            let bru = wedge8(&ru_j, &ru_j1);
            let brv = wedge8(&rv_j, &rv_j1);
            let d_root = match (
                biv_normalize(&bru, EPS_BIV),
                biv_normalize(&brv, EPS_BIV),
            ) {
                (None, None) => 0.0,
                (None, Some(_)) | (Some(_), None) => 1.0,
                (Some(bru_n), Some(brv_n)) => biv_angle_dist(&bru_n, &brv_n),
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

    let d_hct = hct_distance(&u_blocks, &v_blocks);
    let d_struct = 0.5 * d_intra + 0.3 * d_inter + 0.2 * d_hct;
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

    fn blocks_from_vec(v: &[f64]) -> Vec<Option<[f64; ROOT_DIM]>> {
        let blocks = v.len() / ROOT_DIM;
        let mut out = Vec::with_capacity(blocks);
        for block in 0..blocks {
            let start = block * ROOT_DIM;
            let mut raw = [0.0; ROOT_DIM];
            raw.copy_from_slice(&v[start..start + ROOT_DIM]);
            out.push(normalize8(&raw));
        }
        out
    }

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

    #[test]
    fn test_hct_smooth_vs_leap() {
        let blocks = 12;
        let mut u = Vec::with_capacity(blocks * ROOT_DIM);
        for block in 0..blocks {
            let phase = block as f64 * 0.18;
            for i in 0..ROOT_DIM {
                let t = phase + i as f64 * 0.31;
                let val = t.sin() + 0.35 * (t * 0.7).cos();
                u.push(val);
            }
        }

        let mut v = u.clone();
        let kink_block = 6;
        let start = kink_block * ROOT_DIM;
        for i in 0..ROOT_DIM {
            v[start + i] = -v[start + i];
        }

        let u_blocks = blocks_from_vec(&u);
        let v_blocks = blocks_from_vec(&v);
        let d_same = hct_distance(&u_blocks, &u_blocks);
        let d_kink = hct_distance(&u_blocks, &v_blocks);

        assert!(d_same <= 1e-12);
        assert!(d_kink > 0.05);
    }

    #[test]
    fn test_hct_shuffle_sensitivity() {
        let blocks = 16;
        let mut u = Vec::with_capacity(blocks * ROOT_DIM);
        for block in 0..blocks {
            let base = block as f64 * 0.12;
            for i in 0..ROOT_DIM {
                let t = base + i as f64 * 0.27;
                let val = t.sin() + 0.2 * (t * 0.9).cos();
                u.push(val);
            }
        }

        let mut u_rot = vec![0.0; u.len()];
        for block in 0..blocks {
            let src = block * ROOT_DIM;
            let dst = ((block + 1) % blocks) * ROOT_DIM;
            u_rot[dst..dst + ROOT_DIM].copy_from_slice(&u[src..src + ROOT_DIM]);
        }

        let u_blocks = blocks_from_vec(&u);
        let v_blocks = blocks_from_vec(&u_rot);
        let d_hct = hct_distance(&u_blocks, &v_blocks);

        assert!(d_hct > 0.02);
    }
}
