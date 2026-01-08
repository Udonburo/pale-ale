use once_cell::sync::Lazy;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
#[cfg(feature = "inspect")]
use pyo3::types::PyDict;
use std::f64::consts::PI;

const ROOT_DIM: usize = 8;
const BIV_DIM: usize = ROOT_DIM * (ROOT_DIM - 1) / 2;
const ROTOR_DIM: usize = 1 + BIV_DIM;
const EPS_NORM: f64 = 1e-12;
const EPS_BIV: f64 = 1e-12;
const SEMANTIC_EPS: f64 = 1e-9;
const SNAP_SOFT_K: usize = 3;
const SNAP_SOFT_BETA: f64 = 12.0;

#[allow(dead_code)]
struct ComponentStats {
    d_sem: f64,
    d_intra: f64,
    d_inter: f64,
    d_hct: f64,
    intra_root: f64,
    intra_cont: f64,
    intra_snap: f64,
    inter_root: f64,
    inter_cont: f64,
    hct_root: f64,
    hct_cont: f64,
    anchor_u_mean: f64,
    anchor_v_mean: f64,
    anchor_delta: f64,
    valid_blocks: usize,
    valid_pairs: usize,
    valid_triplets: usize,
}

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
fn resolve_alpha(alpha: Option<f64>) -> f64 {
    clamp(alpha.unwrap_or(0.15), 0.0, 1.0)
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

fn snap_soft(u_unit: &[f64; ROOT_DIM], k: usize, beta: f64) -> [f64; ROOT_DIM] {
    let mut dots: Vec<(f64, usize)> = Vec::with_capacity(E8_ROOTS.len());
    let mut best_idx = 0;
    let mut best_dot = dot8(u_unit, &E8_ROOTS[0]);
    dots.push((best_dot, 0));
    for (idx, root) in E8_ROOTS.iter().enumerate().skip(1) {
        let d = dot8(u_unit, root);
        if d > best_dot {
            best_dot = d;
            best_idx = idx;
        }
        dots.push((d, idx));
    }

    let k = k.max(1).min(dots.len());
    dots.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let top = &dots[..k];
    let max_dot = top[0].0;

    let mut weight_sum = 0.0;
    let mut weights: Vec<f64> = Vec::with_capacity(k);
    for (d, _) in top.iter() {
        let w = (beta * (d - max_dot)).exp();
        weight_sum += w;
        weights.push(w);
    }

    if weight_sum > 0.0 && weight_sum.is_finite() {
        let mut acc = [0.0; ROOT_DIM];
        let inv_sum = 1.0 / weight_sum;
        for (w, (_, idx)) in weights.iter().zip(top.iter()) {
            let root = E8_ROOTS[*idx];
            let scaled = w * inv_sum;
            for i in 0..ROOT_DIM {
                acc[i] += scaled * root[i];
            }
        }
        if let Some(normed) = normalize8(&acc) {
            return normed;
        }
    }

    E8_ROOTS[best_idx]
}

fn rotor_distance(a: &[f64; ROOT_DIM], b: &[f64; ROOT_DIM]) -> f64 {
    let dot = clamp(dot8(a, b), -1.0, 1.0);
    let wedge_sq = 1.0 - dot * dot;
    let wedge_norm = if wedge_sq <= 0.0 { 0.0 } else { wedge_sq.sqrt() };
    let theta = wedge_norm.atan2(dot);
    theta / PI
}

fn hct_distance_stats(
    u_blocks: &[Option<[f64; ROOT_DIM]>],
    v_blocks: &[Option<[f64; ROOT_DIM]>],
) -> (f64, f64, f64, usize) {
    let blocks = u_blocks.len().min(v_blocks.len());
    if blocks < 2 {
        return (0.0, 0.0, 0.0, 0);
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

        let ru_j = snap_soft(&u_j, SNAP_SOFT_K, SNAP_SOFT_BETA);
        let ru_j1 = snap_soft(&u_j1, SNAP_SOFT_K, SNAP_SOFT_BETA);
        let rv_j = snap_soft(&v_j, SNAP_SOFT_K, SNAP_SOFT_BETA);
        let rv_j1 = snap_soft(&v_j1, SNAP_SOFT_K, SNAP_SOFT_BETA);
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

    let d_hct = if cont_count == 0 && root_count == 0 {
        0.0
    } else {
        0.6 * d_cont + 0.4 * d_root
    };

    (d_hct, d_root, d_cont, cont_count)
}

fn hct_distance(
    u_blocks: &[Option<[f64; ROOT_DIM]>],
    v_blocks: &[Option<[f64; ROOT_DIM]>],
) -> f64 {
    hct_distance_stats(u_blocks, v_blocks).0
}

fn calculate_components(u: &[f64], v: &[f64]) -> Result<ComponentStats, String> {
    if u.len() != v.len() {
        return Err("u and v must have the same length".to_string());
    }
    let dim = u.len();
    if dim == 0 {
        return Ok(ComponentStats {
            d_sem: 0.0,
            d_intra: 0.0,
            d_inter: 0.0,
            d_hct: 0.0,
            intra_root: 0.0,
            intra_cont: 0.0,
            intra_snap: 0.0,
            inter_root: 0.0,
            inter_cont: 0.0,
            hct_root: 0.0,
            hct_cont: 0.0,
            anchor_u_mean: 0.0,
            anchor_v_mean: 0.0,
            anchor_delta: 0.0,
            valid_blocks: 0,
            valid_pairs: 0,
            valid_triplets: 0,
        });
    }
    if dim % ROOT_DIM != 0 {
        return Err("vector length must be a multiple of 8".to_string());
    }

    let nu = norm(u);
    let nv = norm(v);
    let d_sem = if nu < SEMANTIC_EPS || nv < SEMANTIC_EPS {
        1.0
    } else {
        let sim = clamp(dot(u, v) / (nu * nv), -1.0, 1.0);
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

    let mut anchor_u_sum = 0.0;
    let mut anchor_u_count = 0usize;
    let mut anchor_v_sum = 0.0;
    let mut anchor_v_count = 0usize;

    let mut intra_root_sum = 0.0;
    let mut intra_cont_sum = 0.0;
    let mut intra_snap_sum = 0.0;
    let mut intra_count = 0usize;
    for block in 0..blocks {
        let u_opt = u_blocks[block];
        let v_opt = v_blocks[block];

        let mut r_u_soft_opt: Option<[f64; ROOT_DIM]> = None;
        let mut r_v_soft_opt: Option<[f64; ROOT_DIM]> = None;
        let mut u_anchor = None;
        let mut v_anchor = None;

        if let Some(u_block) = u_opt {
            let r_u_hard = snap_e8(&u_block);
            let u_anchor_val = rotor_distance(&u_block, &r_u_hard);
            anchor_u_sum += u_anchor_val;
            anchor_u_count += 1;
            u_anchor = Some(u_anchor_val);
            r_u_soft_opt = Some(snap_soft(&u_block, SNAP_SOFT_K, SNAP_SOFT_BETA));
        }

        if let Some(v_block) = v_opt {
            let r_v_hard = snap_e8(&v_block);
            let v_anchor_val = rotor_distance(&v_block, &r_v_hard);
            anchor_v_sum += v_anchor_val;
            anchor_v_count += 1;
            v_anchor = Some(v_anchor_val);
            r_v_soft_opt = Some(snap_soft(&v_block, SNAP_SOFT_K, SNAP_SOFT_BETA));
        }

        let (Some(u_block), Some(v_block), Some(r_u), Some(r_v), Some(u_anchor), Some(v_anchor)) = (
            u_opt,
            v_opt,
            r_u_soft_opt,
            r_v_soft_opt,
            u_anchor,
            v_anchor,
        )
        else {
            continue;
        };

        let d_root = rotor_distance(&r_u, &r_v);
        let d_cont = rotor_distance(&u_block, &v_block);
        let d_snap = 0.5 * (u_anchor + v_anchor);

        intra_root_sum += d_root;
        intra_cont_sum += d_cont;
        intra_snap_sum += d_snap;
        intra_count += 1;
    }

    let intra_root = if intra_count > 0 {
        intra_root_sum / (intra_count as f64)
    } else {
        0.0
    };
    let intra_cont = if intra_count > 0 {
        intra_cont_sum / (intra_count as f64)
    } else {
        0.0
    };
    let intra_snap = if intra_count > 0 {
        intra_snap_sum / (intra_count as f64)
    } else {
        0.0
    };
    let d_intra = if intra_count > 0 {
        0.6 * intra_root + 0.4 * intra_cont
    } else {
        0.0
    };

    let anchor_u_mean = if anchor_u_count > 0 {
        anchor_u_sum / (anchor_u_count as f64)
    } else {
        0.0
    };
    let anchor_v_mean = if anchor_v_count > 0 {
        anchor_v_sum / (anchor_v_count as f64)
    } else {
        0.0
    };
    let anchor_delta = (anchor_u_mean - anchor_v_mean).abs();

    let mut inter_sum = 0.0;
    let mut inter_root_sum = 0.0;
    let mut inter_cont_sum = 0.0;
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

            let ru_j = snap_soft(&a_j, SNAP_SOFT_K, SNAP_SOFT_BETA);
            let ru_j1 = snap_soft(&a_j1, SNAP_SOFT_K, SNAP_SOFT_BETA);
            let rv_j = snap_soft(&b_j, SNAP_SOFT_K, SNAP_SOFT_BETA);
            let rv_j1 = snap_soft(&b_j1, SNAP_SOFT_K, SNAP_SOFT_BETA);
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
            inter_root_sum += d_root;
            inter_cont_sum += d_cont;
            inter_sum += d_inter;
            inter_count += 1;
        }
    }

    let d_inter = if inter_count > 0 {
        inter_sum / (inter_count as f64)
    } else {
        0.0
    };

    let inter_root = if inter_count > 0 {
        inter_root_sum / (inter_count as f64)
    } else {
        0.0
    };
    let inter_cont = if inter_count > 0 {
        inter_cont_sum / (inter_count as f64)
    } else {
        0.0
    };

    let (d_hct, hct_root, hct_cont, valid_triplets) = hct_distance_stats(&u_blocks, &v_blocks);

    Ok(ComponentStats {
        d_sem,
        d_intra,
        d_inter,
        d_hct,
        intra_root,
        intra_cont,
        intra_snap,
        inter_root,
        inter_cont,
        hct_root,
        hct_cont,
        anchor_u_mean,
        anchor_v_mean,
        anchor_delta,
        valid_blocks: intra_count,
        valid_pairs: inter_count,
        valid_triplets,
    })
}

#[pyfunction]
fn spin3_distance(u: Vec<f64>, v: Vec<f64>, alpha: Option<f64>) -> PyResult<f64> {
    let stats = calculate_components(&u, &v).map_err(PyValueError::new_err)?;
    if u.is_empty() {
        return Ok(0.0);
    }

    let alpha_weight = resolve_alpha(alpha);
    let d_struct = 0.5 * stats.d_intra + 0.3 * stats.d_inter + 0.2 * stats.d_hct;
    let d = (1.0 - alpha_weight) * stats.d_sem + alpha_weight * d_struct;
    Ok(clamp(d, 0.0, 1.0))
}

#[cfg(feature = "inspect")]
#[pyfunction]
fn spin3_inspect(
    py: Python<'_>,
    u: Vec<f64>,
    v: Vec<f64>,
    alpha: Option<f64>,
) -> PyResult<Bound<'_, PyDict>> {
    let stats = calculate_components(&u, &v).map_err(PyValueError::new_err)?;
    let alpha_weight = resolve_alpha(alpha);
    let d_struct = 0.5 * stats.d_intra + 0.3 * stats.d_inter + 0.2 * stats.d_hct;
    let total = if u.is_empty() {
        0.0
    } else {
        clamp(
            (1.0 - alpha_weight) * stats.d_sem + alpha_weight * d_struct,
            0.0,
            1.0,
        )
    };

    let dict = PyDict::new_bound(py);
    dict.set_item("semantic", stats.d_sem)?;
    dict.set_item("intra", stats.d_intra)?;
    dict.set_item("inter", stats.d_inter)?;
    dict.set_item("hct", stats.d_hct)?;
    dict.set_item("intra_root", stats.intra_root)?;
    dict.set_item("intra_cont", stats.intra_cont)?;
    dict.set_item("intra_snap", stats.intra_snap)?;
    dict.set_item("inter_root", stats.inter_root)?;
    dict.set_item("inter_cont", stats.inter_cont)?;
    dict.set_item("hct_root", stats.hct_root)?;
    dict.set_item("hct_cont", stats.hct_cont)?;
    dict.set_item("anchor_u_mean", stats.anchor_u_mean)?;
    dict.set_item("anchor_v_mean", stats.anchor_v_mean)?;
    dict.set_item("anchor_delta", stats.anchor_delta)?;
    dict.set_item("structural", d_struct)?;
    dict.set_item("total", total)?;
    dict.set_item("alpha", alpha_weight)?;
    dict.set_item("valid_blocks", stats.valid_blocks)?;
    dict.set_item("valid_pairs", stats.valid_pairs)?;
    dict.set_item("valid_triplets", stats.valid_triplets)?;
    Ok(dict)
}

#[pymodule]
fn pale_ale_core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(spin3_distance, m)?)?;
    #[cfg(feature = "inspect")]
    m.add_function(wrap_pyfunction!(spin3_inspect, m)?)?;
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
    fn test_identity_is_zero() {
        let u = vec![
            0.2, -0.1, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, 0.2, -0.3, 0.4, -0.5, 0.6,
            -0.7, 0.8,
        ];
        let d = spin3_distance(u.clone(), u, Some(1.0)).unwrap();
        assert!(d < 1e-5);
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

        assert!(d_same <= 1e-6);
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
