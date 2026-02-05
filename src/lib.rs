#[cfg(feature = "numpy-support")]
use numpy::PyReadonlyArray1;
use once_cell::sync::Lazy;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::f64::consts::PI;
use std::marker::PhantomData;

const ROOT_DIM: usize = 8;
const ROOT_COUNT: usize = 240;
const BIV_DIM: usize = ROOT_DIM * (ROOT_DIM - 1) / 2;
const ROTOR_DIM: usize = 1 + BIV_DIM;
const EPS_NORM: f64 = 1e-12;
const EPS_BIV: f64 = 1e-12;
const SEMANTIC_EPS: f64 = 1e-9;
const SNAP_SOFT_K: usize = 3;
const SNAP_SOFT_BETA: f64 = 12.0;

struct SnapMeta {
    soft: [f64; ROOT_DIM],
    #[allow(dead_code)]
    hard_idx: usize,
    hard_dot: f64,
}

#[derive(Debug)]
struct PrecomputedBlock {
    soft: [f64; ROOT_DIM],
    anchor: f64,
}

enum DataView<'a> {
    Owned(Vec<f64>, PhantomData<&'a ()>),
    #[cfg(feature = "numpy-support")]
    Numpy(PyReadonlyArray1<'a, f64>),
}

impl<'a> DataView<'a> {
    fn as_slice(&self) -> &[f64] {
        match self {
            DataView::Owned(v, _) => v.as_slice(),
            #[cfg(feature = "numpy-support")]
            DataView::Numpy(array) => array.as_slice().expect("numpy array should be contiguous"),
        }
    }
}

fn extract_view<'py>(obj: &Bound<'py, PyAny>) -> PyResult<DataView<'py>> {
    #[cfg(feature = "numpy-support")]
    {
        if let Ok(array) = obj.extract::<PyReadonlyArray1<'py, f64>>() {
            if array.as_slice().is_ok() {
                return Ok(DataView::Numpy(array));
            }
        }
    }

    let owned = obj.extract::<Vec<f64>>()?;
    Ok(DataView::Owned(owned, PhantomData))
}

#[derive(Clone, Copy, Debug)]
pub struct Spin3Components {
    pub d_intra: f64,
    pub d_inter: f64,
    pub d_hct: f64,
    pub d_struct: f64,
    pub intra_root: f64,
    pub intra_cont: f64,
    pub intra_snap: f64,
    pub inter_root: f64,
    pub inter_cont: f64,
    pub hct_root: f64,
    pub hct_cont: f64,
    pub anchor_u_mean: f64,
    pub anchor_v_mean: f64,
    pub anchor_delta: f64,
    pub valid_blocks: usize,
    pub valid_pairs: usize,
    pub valid_triplets: usize,
}

impl Spin3Components {
    pub fn structural_distance(&self) -> f64 {
        self.d_struct
    }
}

static E8_ROOTS: Lazy<Vec<[f64; ROOT_DIM]>> = Lazy::new(|| {
    let mut roots = Vec::with_capacity(ROOT_COUNT);
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
        for (bit, val) in v.iter_mut().enumerate() {
            let sign = if (mask >> bit) & 1 == 1 { -0.5 } else { 0.5 };
            *val = sign * inv_sqrt2;
        }
        roots.push(v);
    }

    debug_assert_eq!(roots.len(), ROOT_COUNT);
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
fn angle_dist_from_dot(dot: f64) -> f64 {
    let d = clamp(dot, -1.0, 1.0);
    let wedge = (1.0 - d * d).max(0.0).sqrt();
    wedge.atan2(d) / PI
}

#[inline]
fn resolve_alpha(alpha: Option<f64>) -> f64 {
    clamp(alpha.unwrap_or(0.15), 0.0, 1.0)
}

fn validate_alpha(alpha: Option<f64>) -> Result<(), String> {
    if let Some(a) = alpha {
        if !a.is_finite() {
            return Err("alpha must be a finite number".to_string());
        }
    }
    Ok(())
}

fn resolve_k_beta(k: Option<usize>, beta: Option<f64>) -> Result<(usize, f64), String> {
    let k = k.unwrap_or(SNAP_SOFT_K).clamp(1, ROOT_COUNT);
    let beta = beta.unwrap_or(SNAP_SOFT_BETA);
    if !beta.is_finite() || beta <= 0.0 {
        return Err("beta must be a positive finite number".to_string());
    }
    Ok((k, beta))
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline]
fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn semantic_distance(u: &[f64], v: &[f64]) -> f64 {
    if u.is_empty() {
        return 0.0;
    }

    let nu = norm(u);
    let nv = norm(v);
    if nu < SEMANTIC_EPS || nv < SEMANTIC_EPS {
        1.0
    } else {
        let sim = clamp(dot(u, v) / (nu * nv), -1.0, 1.0);
        0.5 * (1.0 - sim)
    }
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
    let wedge_norm = if wedge_sq <= 0.0 {
        0.0
    } else {
        wedge_sq.sqrt()
    };
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
    out[1..].copy_from_slice(&b);
    let n = norm29(&out).max(EPS_NORM);
    let inv = 1.0 / n;
    for val in out.iter_mut() {
        *val *= inv;
    }
    out
}

fn rotor_dist_29(a: &[f64; ROTOR_DIM], b: &[f64; ROTOR_DIM]) -> f64 {
    let a_n = normalize29(a);
    let b_n = normalize29(b);
    let dot = clamp(dot29(&a_n, &b_n), -1.0, 1.0);
    let wedge_sq = 1.0 - dot * dot;
    let wedge_norm = if wedge_sq <= 0.0 {
        0.0
    } else {
        wedge_sq.sqrt()
    };
    let theta = wedge_norm.atan2(dot);
    theta / PI
}

#[allow(dead_code)]
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

fn snap_soft_meta_dyn(u_unit: &[f64; ROOT_DIM], k: usize, beta: f64) -> SnapMeta {
    let k = k.clamp(1, ROOT_COUNT);
    let mut top_dots = [f64::NEG_INFINITY; ROOT_COUNT];
    let mut top_idx = [usize::MAX; ROOT_COUNT];
    let mut weights = [0.0; ROOT_COUNT];

    for (idx, root) in E8_ROOTS.iter().enumerate() {
        let d = dot8(u_unit, root);

        let mut insert_pos = None;
        for i in 0..k {
            let cur_dot = top_dots[i];
            let cur_idx = top_idx[i];
            if d > cur_dot || (d == cur_dot && idx < cur_idx) {
                insert_pos = Some(i);
                break;
            }
        }

        if let Some(pos) = insert_pos {
            for j in (pos + 1..k).rev() {
                top_dots[j] = top_dots[j - 1];
                top_idx[j] = top_idx[j - 1];
            }
            top_dots[pos] = d;
            top_idx[pos] = idx;
        }
    }

    let hard_dot = top_dots[0];
    let hard_idx = top_idx[0];

    let mut soft = E8_ROOTS[hard_idx];
    let max_dot = hard_dot;
    let mut weight_sum = 0.0;
    for i in 0..k {
        let w = (beta * (top_dots[i] - max_dot)).exp();
        weights[i] = w;
        weight_sum += w;
    }

    if weight_sum > 0.0 && weight_sum.is_finite() {
        let mut acc = [0.0; ROOT_DIM];
        let inv_sum = 1.0 / weight_sum;
        for i in 0..k {
            let w = weights[i] * inv_sum;
            let root = E8_ROOTS[top_idx[i]];
            for j in 0..ROOT_DIM {
                acc[j] += w * root[j];
            }
        }
        if let Some(normed) = normalize8(&acc) {
            soft = normed;
        }
    }

    SnapMeta {
        soft,
        hard_idx,
        hard_dot,
    }
}

fn snap_soft_meta_fixed<const K: usize>(u_unit: &[f64; ROOT_DIM], beta: f64) -> SnapMeta {
    let mut top_dots = [f64::NEG_INFINITY; K];
    let mut top_idx = [usize::MAX; K];

    for (idx, root) in E8_ROOTS.iter().enumerate() {
        let d = dot8(u_unit, root);

        let mut insert_pos = None;
        for i in 0..K {
            let cur_dot = top_dots[i];
            let cur_idx = top_idx[i];
            if d > cur_dot || (d == cur_dot && idx < cur_idx) {
                insert_pos = Some(i);
                break;
            }
        }

        if let Some(pos) = insert_pos {
            for j in (pos + 1..K).rev() {
                top_dots[j] = top_dots[j - 1];
                top_idx[j] = top_idx[j - 1];
            }
            top_dots[pos] = d;
            top_idx[pos] = idx;
        }
    }

    let hard_dot = top_dots[0];
    let hard_idx = top_idx[0];

    let mut soft = E8_ROOTS[hard_idx];
    let max_dot = hard_dot;
    let mut weights = [0.0; K];
    let mut weight_sum = 0.0;
    for i in 0..K {
        let w = (beta * (top_dots[i] - max_dot)).exp();
        weights[i] = w;
        weight_sum += w;
    }

    if weight_sum > 0.0 && weight_sum.is_finite() {
        let mut acc = [0.0; ROOT_DIM];
        let inv_sum = 1.0 / weight_sum;
        for i in 0..K {
            let w = weights[i] * inv_sum;
            let root = E8_ROOTS[top_idx[i]];
            for j in 0..ROOT_DIM {
                acc[j] += w * root[j];
            }
        }
        if let Some(normed) = normalize8(&acc) {
            soft = normed;
        }
    }

    SnapMeta {
        soft,
        hard_idx,
        hard_dot,
    }
}

fn snap_soft_meta(u_unit: &[f64; ROOT_DIM], k: usize, beta: f64) -> SnapMeta {
    let k = k.clamp(1, ROOT_COUNT);
    match k {
        1 => snap_soft_meta_fixed::<1>(u_unit, beta),
        2 => snap_soft_meta_fixed::<2>(u_unit, beta),
        3 => snap_soft_meta_fixed::<3>(u_unit, beta),
        _ => snap_soft_meta_dyn(u_unit, k, beta),
    }
}

#[allow(dead_code)]
fn snap_soft(u_unit: &[f64; ROOT_DIM], k: usize, beta: f64) -> [f64; ROOT_DIM] {
    snap_soft_meta(u_unit, k, beta).soft
}

fn precompute_soft_and_anchor(
    blocks: &[Option<[f64; ROOT_DIM]>],
    k: usize,
    beta: f64,
) -> Vec<Option<PrecomputedBlock>> {
    let mut precomputed = Vec::with_capacity(blocks.len());
    for block in blocks.iter() {
        if let Some(u_unit) = block {
            let meta = snap_soft_meta(u_unit, k, beta);
            precomputed.push(Some(PrecomputedBlock {
                soft: meta.soft,
                anchor: angle_dist_from_dot(meta.hard_dot),
            }));
        } else {
            precomputed.push(None);
        }
    }
    precomputed
}

fn rotor_distance(a: &[f64; ROOT_DIM], b: &[f64; ROOT_DIM]) -> f64 {
    let dot = clamp(dot8(a, b), -1.0, 1.0);
    let wedge_sq = 1.0 - dot * dot;
    let wedge_norm = if wedge_sq <= 0.0 {
        0.0
    } else {
        wedge_sq.sqrt()
    };
    let theta = wedge_norm.atan2(dot);
    theta / PI
}

fn hct_distance_stats(
    u_blocks: &[Option<[f64; ROOT_DIM]>],
    v_blocks: &[Option<[f64; ROOT_DIM]>],
    u_pre: &[Option<PrecomputedBlock>],
    v_pre: &[Option<PrecomputedBlock>],
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
        let (Some(u_j), Some(u_j1)) = (u_blocks[j].as_ref(), u_blocks[j + 1].as_ref()) else {
            prev_cont = None;
            prev_root = None;
            continue;
        };
        let (Some(v_j), Some(v_j1)) = (v_blocks[j].as_ref(), v_blocks[j + 1].as_ref()) else {
            prev_cont = None;
            prev_root = None;
            continue;
        };

        let qu = to_rotor29(u_j, u_j1);
        let qv = to_rotor29(v_j, v_j1);
        let m_cont = rotor_dist_29(&qu, &qv);

        let (Some(ru_j), Some(ru_j1)) = (u_pre[j].as_ref(), u_pre[j + 1].as_ref()) else {
            prev_cont = None;
            prev_root = None;
            continue;
        };
        let (Some(rv_j), Some(rv_j1)) = (v_pre[j].as_ref(), v_pre[j + 1].as_ref()) else {
            prev_cont = None;
            prev_root = None;
            continue;
        };
        let qu_root = to_rotor29(&ru_j.soft, &ru_j1.soft);
        let qv_root = to_rotor29(&rv_j.soft, &rv_j1.soft);
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

#[allow(dead_code)]
fn hct_distance(u_blocks: &[Option<[f64; ROOT_DIM]>], v_blocks: &[Option<[f64; ROOT_DIM]>]) -> f64 {
    let u_pre = precompute_soft_and_anchor(u_blocks, SNAP_SOFT_K, SNAP_SOFT_BETA);
    let v_pre = precompute_soft_and_anchor(v_blocks, SNAP_SOFT_K, SNAP_SOFT_BETA);
    hct_distance_stats(u_blocks, v_blocks, &u_pre, &v_pre).0
}

fn calculate_components(
    u: &[f64],
    v: &[f64],
    k: usize,
    beta: f64,
) -> Result<Spin3Components, String> {
    if u.len() != v.len() {
        return Err("u and v must have the same length".to_string());
    }
    let dim = u.len();
    if dim == 0 {
        return Ok(Spin3Components {
            d_intra: 0.0,
            d_inter: 0.0,
            d_hct: 0.0,
            d_struct: 0.0,
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
    #[allow(clippy::manual_is_multiple_of)]
    if dim % ROOT_DIM != 0 {
        return Err("vector length must be a multiple of 8".to_string());
    }

    let blocks = dim / ROOT_DIM;
    let mut u_blocks = Vec::with_capacity(blocks);
    let mut v_blocks = Vec::with_capacity(blocks);
    for block in 0..blocks {
        let start = block * ROOT_DIM;
        let mut u_block = [0.0; ROOT_DIM];
        let mut v_block = [0.0; ROOT_DIM];
        for i in 0..ROOT_DIM {
            let u_val = u[start + i];
            let v_val = v[start + i];
            if !u_val.is_finite() || !v_val.is_finite() {
                return Err("u and v must contain only finite f64 values".to_string());
            }
            u_block[i] = u_val;
            v_block[i] = v_val;
        }
        u_blocks.push(normalize8(&u_block));
        v_blocks.push(normalize8(&v_block));
    }

    let u_pre = precompute_soft_and_anchor(&u_blocks, k, beta);
    let v_pre = precompute_soft_and_anchor(&v_blocks, k, beta);

    let mut anchor_u_sum = 0.0;
    let mut anchor_u_count = 0usize;
    let mut anchor_v_sum = 0.0;
    let mut anchor_v_count = 0usize;

    let mut intra_root_sum = 0.0;
    let mut intra_cont_sum = 0.0;
    let mut intra_snap_sum = 0.0;
    let mut intra_count = 0usize;
    for block in 0..blocks {
        let u_opt = u_blocks[block].as_ref();
        let v_opt = v_blocks[block].as_ref();

        let u_pre_opt = u_pre[block].as_ref();
        let v_pre_opt = v_pre[block].as_ref();

        if let Some(pre) = u_pre_opt {
            anchor_u_sum += pre.anchor;
            anchor_u_count += 1;
        }

        if let Some(pre) = v_pre_opt {
            anchor_v_sum += pre.anchor;
            anchor_v_count += 1;
        }

        let (Some(u_block), Some(v_block), Some(u_pre_block), Some(v_pre_block)) =
            (u_opt, v_opt, u_pre_opt, v_pre_opt)
        else {
            continue;
        };

        let d_root = rotor_distance(&u_pre_block.soft, &v_pre_block.soft);
        let d_cont = rotor_distance(u_block, v_block);
        let d_snap = 0.5 * (u_pre_block.anchor + v_pre_block.anchor);

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
            let (Some(a_j), Some(a_j1)) = (u_blocks[j].as_ref(), u_blocks[j + 1].as_ref()) else {
                continue;
            };
            let (Some(b_j), Some(b_j1)) = (v_blocks[j].as_ref(), v_blocks[j + 1].as_ref()) else {
                continue;
            };

            let bu = wedge8(a_j, a_j1);
            let bv = wedge8(b_j, b_j1);
            let d_cont = match (biv_normalize(&bu, EPS_BIV), biv_normalize(&bv, EPS_BIV)) {
                (None, None) => 0.0,
                (None, Some(_)) | (Some(_), None) => 1.0,
                (Some(bu_n), Some(bv_n)) => biv_angle_dist(&bu_n, &bv_n),
            };

            let (Some(ru_j), Some(ru_j1)) = (u_pre[j].as_ref(), u_pre[j + 1].as_ref()) else {
                continue;
            };
            let (Some(rv_j), Some(rv_j1)) = (v_pre[j].as_ref(), v_pre[j + 1].as_ref()) else {
                continue;
            };
            let bru = wedge8(&ru_j.soft, &ru_j1.soft);
            let brv = wedge8(&rv_j.soft, &rv_j1.soft);
            let d_root = match (biv_normalize(&bru, EPS_BIV), biv_normalize(&brv, EPS_BIV)) {
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

    let (d_hct, hct_root, hct_cont, valid_triplets) =
        hct_distance_stats(&u_blocks, &v_blocks, &u_pre, &v_pre);
    let d_struct = 0.5 * d_intra + 0.3 * d_inter + 0.2 * d_hct;

    Ok(Spin3Components {
        d_intra,
        d_inter,
        d_hct,
        d_struct,
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

/// Pure Rust API: computes the mixed semantic/structural distance between two vectors.
///
/// Returns a score in [0.0, 1.0] where 0.0 is identical and 1.0 is maximally different.
///
/// # Arguments
///
/// * `u` - First vector (length must be multiple of 8)
/// * `v` - Second vector (length must be multiple of 8)
/// * `alpha` - Mixing factor [0.0, 1.0].
///     * `Some(0.0)` -> pure semantic distance (approx 1 - cosine)
///     * `Some(1.0)` -> pure structural distance
///     * `None` -> defaults to 0.15
///
/// Prefer `spin3_struct` + Python-side semantic mixing for new integrations.
pub fn spin3_distance(u: &[f64], v: &[f64], alpha: Option<f64>) -> Result<f64, String> {
    validate_alpha(alpha)?;
    let components = calculate_components(u, v, SNAP_SOFT_K, SNAP_SOFT_BETA)?;
    if u.is_empty() {
        return Ok(0.0);
    }

    let alpha_weight = resolve_alpha(alpha);
    let d_sem = semantic_distance(u, v);
    let d_struct = components.d_struct;
    let d = (1.0 - alpha_weight) * d_sem + alpha_weight * d_struct;
    Ok(clamp(d, 0.0, 1.0))
}

/// Pure Rust API: computes the structural-only distance between two vectors.
pub fn spin3_struct(
    u: &[f64],
    v: &[f64],
    k: Option<usize>,
    beta: Option<f64>,
) -> Result<f64, String> {
    let (k, beta) = resolve_k_beta(k, beta)?;
    let components = calculate_components(u, v, k, beta)?;
    if u.is_empty() {
        return Ok(0.0);
    }

    Ok(clamp(components.d_struct, 0.0, 1.0))
}

/// Pure Rust API: computes the structural-only distance between two vectors (default k/beta).
pub fn spin3_struct_distance(u: &[f64], v: &[f64]) -> Result<f64, String> {
    spin3_struct(u, v, None, None)
}

/// Pure Rust API: returns structural component breakdown (default k/beta).
pub fn spin3_components(u: &[f64], v: &[f64]) -> Result<Spin3Components, String> {
    spin3_components_with(u, v, None, None)
}

fn spin3_components_with(
    u: &[f64],
    v: &[f64],
    k: Option<usize>,
    beta: Option<f64>,
) -> Result<Spin3Components, String> {
    let (k, beta) = resolve_k_beta(k, beta)?;
    calculate_components(u, v, k, beta)
}

/// Python API wrapper for spin3_distance
#[pyfunction]
#[pyo3(name = "spin3_distance")]
fn spin3_distance_py(
    u: &Bound<'_, PyAny>,
    v: &Bound<'_, PyAny>,
    alpha: Option<f64>,
) -> PyResult<f64> {
    let u_view = extract_view(u)?;
    let v_view = extract_view(v)?;
    spin3_distance(u_view.as_slice(), v_view.as_slice(), alpha).map_err(PyValueError::new_err)
}

/// Python API wrapper for spin3_struct
#[pyfunction]
#[pyo3(name = "spin3_struct")]
fn spin3_struct_py(
    u: &Bound<'_, PyAny>,
    v: &Bound<'_, PyAny>,
    k: Option<usize>,
    beta: Option<f64>,
) -> PyResult<f64> {
    let u_view = extract_view(u)?;
    let v_view = extract_view(v)?;
    spin3_struct(u_view.as_slice(), v_view.as_slice(), k, beta).map_err(PyValueError::new_err)
}

/// Python API wrapper for spin3_struct_distance (legacy alias)
#[pyfunction]
#[pyo3(name = "spin3_struct_distance")]
fn spin3_struct_distance_py(u: &Bound<'_, PyAny>, v: &Bound<'_, PyAny>) -> PyResult<f64> {
    let u_view = extract_view(u)?;
    let v_view = extract_view(v)?;
    spin3_struct(u_view.as_slice(), v_view.as_slice(), None, None).map_err(PyValueError::new_err)
}

/// Python API wrapper for spin3_components
#[pyfunction]
#[pyo3(name = "spin3_components")]
fn spin3_components_py<'py>(
    py: Python<'py>,
    u: &Bound<'py, PyAny>,
    v: &Bound<'py, PyAny>,
    k: Option<usize>,
    beta: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let u_view = extract_view(u)?;
    let v_view = extract_view(v)?;
    let components = spin3_components_with(u_view.as_slice(), v_view.as_slice(), k, beta)
        .map_err(PyValueError::new_err)?;
    build_components_dict(py, &components)
}

fn build_components_dict<'py>(
    py: Python<'py>,
    components: &Spin3Components,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("d_intra", components.d_intra)?;
    dict.set_item("d_inter", components.d_inter)?;
    dict.set_item("d_hct", components.d_hct)?;
    dict.set_item("d_struct", components.d_struct)?;
    dict.set_item("intra", components.d_intra)?;
    dict.set_item("inter", components.d_inter)?;
    dict.set_item("hct", components.d_hct)?;
    dict.set_item("intra_root", components.intra_root)?;
    dict.set_item("intra_cont", components.intra_cont)?;
    dict.set_item("intra_snap", components.intra_snap)?;
    dict.set_item("inter_root", components.inter_root)?;
    dict.set_item("inter_cont", components.inter_cont)?;
    dict.set_item("hct_root", components.hct_root)?;
    dict.set_item("hct_cont", components.hct_cont)?;
    dict.set_item("anchor_u_mean", components.anchor_u_mean)?;
    dict.set_item("anchor_v_mean", components.anchor_v_mean)?;
    dict.set_item("anchor_delta", components.anchor_delta)?;
    dict.set_item("structural", components.d_struct)?;
    dict.set_item("valid_blocks", components.valid_blocks)?;
    dict.set_item("valid_pairs", components.valid_pairs)?;
    dict.set_item("valid_triplets", components.valid_triplets)?;
    Ok(dict)
}

#[cfg(feature = "inspect")]
fn build_inspect_dict<'py>(
    py: Python<'py>,
    components: &Spin3Components,
    d_sem: f64,
    alpha_weight: f64,
    d_struct: f64,
    total: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = build_components_dict(py, components)?;
    dict.set_item("semantic", d_sem)?;
    dict.set_item("total", total)?;
    dict.set_item("alpha", alpha_weight)?;
    dict.set_item("d_struct", d_struct)?;
    Ok(dict)
}

#[cfg(feature = "inspect")]
#[pyfunction]
fn spin3_inspect<'py>(
    py: Python<'py>,
    u: &Bound<'py, PyAny>,
    v: &Bound<'py, PyAny>,
    alpha: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let u_view = extract_view(u)?;
    let v_view = extract_view(v)?;
    validate_alpha(alpha).map_err(PyValueError::new_err)?;
    let components = calculate_components(
        u_view.as_slice(),
        v_view.as_slice(),
        SNAP_SOFT_K,
        SNAP_SOFT_BETA,
    )
    .map_err(PyValueError::new_err)?;
    let alpha_weight = resolve_alpha(alpha);
    let d_sem = semantic_distance(u_view.as_slice(), v_view.as_slice());
    let d_struct = components.structural_distance();
    let total = if u_view.as_slice().is_empty() {
        0.0
    } else {
        clamp(
            (1.0 - alpha_weight) * d_sem + alpha_weight * d_struct,
            0.0,
            1.0,
        )
    };

    build_inspect_dict(py, &components, d_sem, alpha_weight, d_struct, total)
}

#[cfg(feature = "inspect")]
#[pyfunction]
fn spin3_inspect_dev<'py>(
    py: Python<'py>,
    u: &Bound<'py, PyAny>,
    v: &Bound<'py, PyAny>,
    k: usize,
    beta: f64,
    alpha: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let k = k.clamp(1, ROOT_COUNT);
    if !beta.is_finite() || beta <= 0.0 {
        return Err(PyValueError::new_err(
            "beta must be a positive finite number",
        ));
    }

    let u_view = extract_view(u)?;
    let v_view = extract_view(v)?;
    validate_alpha(alpha).map_err(PyValueError::new_err)?;
    let components = calculate_components(u_view.as_slice(), v_view.as_slice(), k, beta)
        .map_err(PyValueError::new_err)?;
    let alpha_weight = resolve_alpha(alpha);
    let d_sem = semantic_distance(u_view.as_slice(), v_view.as_slice());
    let d_struct = components.structural_distance();
    let total = if u_view.as_slice().is_empty() {
        0.0
    } else {
        clamp(
            (1.0 - alpha_weight) * d_sem + alpha_weight * d_struct,
            0.0,
            1.0,
        )
    };

    build_inspect_dict(py, &components, d_sem, alpha_weight, d_struct, total)
}

#[pymodule]
fn pale_ale_core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(spin3_distance_py, m)?)?;
    m.add_function(wrap_pyfunction!(spin3_struct_py, m)?)?;
    m.add_function(wrap_pyfunction!(spin3_struct_distance_py, m)?)?;
    m.add_function(wrap_pyfunction!(spin3_components_py, m)?)?;
    #[cfg(feature = "inspect")]
    m.add_function(wrap_pyfunction!(spin3_inspect, m)?)?;
    #[cfg(feature = "inspect")]
    m.add_function(wrap_pyfunction!(spin3_inspect_dev, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "numpy-support")]
    use numpy::PyArray1;
    use pyo3::types::PyList;
    use pyo3::Python;

    fn assert_unit_interval(value: f64) {
        assert!(value.is_finite());
        assert!((0.0..=1.0).contains(&value));
    }

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
        // Use pure Rust function
        let d1 = spin3_distance(&u, &v, Some(0.25)).unwrap();
        let d2 = spin3_distance(&v, &u, Some(0.25)).unwrap();
        assert!((d1 - d2).abs() < 1e-12);
    }

    #[test]
    fn struct_matches_alpha_one() {
        let cases = vec![
            (
                (0..16)
                    .map(|i| (i as f64 + 1.0).sin())
                    .collect::<Vec<f64>>(),
                (0..16)
                    .map(|i| (i as f64 + 1.0).cos())
                    .collect::<Vec<f64>>(),
            ),
            (
                (0..24)
                    .map(|i| ((i as f64) * 0.17).sin())
                    .collect::<Vec<f64>>(),
                (0..24)
                    .map(|i| ((i as f64) * 0.13).cos())
                    .collect::<Vec<f64>>(),
            ),
        ];

        for (u, v) in cases {
            let d_struct = spin3_struct(&u, &v, None, None).unwrap();
            let d_mix = spin3_distance(&u, &v, Some(1.0)).unwrap();
            assert!((d_struct - d_mix).abs() < 1e-12);
        }
    }

    #[test]
    fn components_symmetry() {
        let u: Vec<f64> = (0..32).map(|i| ((i as f64) * 0.21).sin()).collect();
        let v: Vec<f64> = (0..32).map(|i| ((i as f64) * 0.19).cos()).collect();
        let a = spin3_components_with(&u, &v, None, None).unwrap();
        let b = spin3_components_with(&v, &u, None, None).unwrap();

        assert!((a.d_struct - b.d_struct).abs() < 1e-12);
        assert!((a.d_intra - b.d_intra).abs() < 1e-12);
        assert!((a.d_inter - b.d_inter).abs() < 1e-12);
        assert!((a.d_hct - b.d_hct).abs() < 1e-12);
        assert!((a.intra_root - b.intra_root).abs() < 1e-12);
        assert!((a.intra_cont - b.intra_cont).abs() < 1e-12);
        assert!((a.intra_snap - b.intra_snap).abs() < 1e-12);
        assert!((a.inter_root - b.inter_root).abs() < 1e-12);
        assert!((a.inter_cont - b.inter_cont).abs() < 1e-12);
        assert!((a.hct_root - b.hct_root).abs() < 1e-12);
        assert!((a.hct_cont - b.hct_cont).abs() < 1e-12);
        assert!((a.anchor_u_mean - b.anchor_v_mean).abs() < 1e-12);
        assert!((a.anchor_v_mean - b.anchor_u_mean).abs() < 1e-12);
        assert!((a.anchor_delta - b.anchor_delta).abs() < 1e-12);
    }

    #[test]
    fn component_bounds() {
        let u: Vec<f64> = (0..16).map(|i| (i as f64 * 0.11).sin()).collect();
        let v: Vec<f64> = (0..16).map(|i| (i as f64 * 0.09).cos()).collect();
        let components = spin3_components_with(&u, &v, None, None).unwrap();

        assert_unit_interval(components.d_intra);
        assert_unit_interval(components.d_inter);
        assert_unit_interval(components.d_hct);
        assert_unit_interval(components.d_struct);
        assert_unit_interval(components.intra_root);
        assert_unit_interval(components.intra_cont);
        assert_unit_interval(components.intra_snap);
        assert_unit_interval(components.inter_root);
        assert_unit_interval(components.inter_cont);
        assert_unit_interval(components.hct_root);
        assert_unit_interval(components.hct_cont);
        assert_unit_interval(components.anchor_u_mean);
        assert_unit_interval(components.anchor_v_mean);
        assert_unit_interval(components.anchor_delta);
    }

    #[test]
    fn anchor_from_dot_matches_snap_e8() {
        let mut raw = [0.0; ROOT_DIM];
        for (i, val) in raw.iter_mut().enumerate() {
            *val = ((i as f64) * 0.37 + 0.11).sin();
        }
        let unit = normalize8(&raw).expect("normalize8 should succeed for test input");
        let meta = snap_soft_meta(&unit, SNAP_SOFT_K, SNAP_SOFT_BETA);
        let hard_dot = dot8(&unit, &E8_ROOTS[meta.hard_idx]);
        assert!((hard_dot - meta.hard_dot).abs() < 1e-12);
        let anchor_dot = angle_dist_from_dot(meta.hard_dot);
        let anchor_snap = rotor_distance(&unit, &snap_e8(&unit));
        assert!((anchor_dot - anchor_snap).abs() < 1e-12);
    }

    #[test]
    fn snap_soft_meta_fixed_matches_dyn() {
        let mut raw = [0.0; ROOT_DIM];
        for (i, val) in raw.iter_mut().enumerate() {
            *val = (i as f64 * 0.17 + 0.09).sin();
        }
        let unit = normalize8(&raw).expect("normalize8 should succeed for test input");
        let beta = 7.25;
        for k in [1_usize, 2, 3] {
            let fixed = match k {
                1 => snap_soft_meta_fixed::<1>(&unit, beta),
                2 => snap_soft_meta_fixed::<2>(&unit, beta),
                3 => snap_soft_meta_fixed::<3>(&unit, beta),
                _ => unreachable!("k is restricted to 1..=3 in this test"),
            };
            let dyn_meta = snap_soft_meta_dyn(&unit, k, beta);

            assert_eq!(fixed.hard_idx, dyn_meta.hard_idx);
            assert!((fixed.hard_dot - dyn_meta.hard_dot).abs() < 1e-12);
            for i in 0..ROOT_DIM {
                assert!((fixed.soft[i] - dyn_meta.soft[i]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn bounds_in_unit_interval() {
        let mut u = vec![0.0; 16];
        let mut v = vec![0.0; 16];
        for i in 0..16 {
            u[i] = (i as f64).sin();
            v[i] = (i as f64).cos();
        }
        let d = spin3_distance(&u, &v, Some(0.10)).unwrap();
        assert!((0.0..=1.0).contains(&d));
    }

    #[test]
    fn identical_vectors_near_zero() {
        let u = vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let d = spin3_distance(&u, &u, Some(0.5)).unwrap();
        assert!((0.0..=1.0).contains(&d));
        assert!(d.abs() < 1e-8);
    }

    #[test]
    fn test_identity_is_zero() {
        let u = vec![
            0.2, -0.1, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8,
        ];
        let d = spin3_distance(&u, &u, Some(1.0)).unwrap();
        assert!(d < 1e-5);
    }

    #[test]
    fn opposite_vectors_near_one() {
        let u = vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v: Vec<f64> = u.iter().map(|x| -x).collect();
        let d = spin3_distance(&u, &v, None).unwrap();
        assert!((0.90..=1.0).contains(&d));
    }

    #[test]
    fn invalid_length_errors() {
        let u = vec![0.1; 10];
        let v = vec![0.1; 10];
        assert!(spin3_distance(&u, &v, None).is_err());
    }

    #[test]
    fn rejects_nonfinite_inputs() {
        let base_u = vec![0.1; 8];
        let base_v = vec![0.2; 8];

        let mut u_nan = base_u.clone();
        u_nan[0] = f64::NAN;
        assert!(spin3_struct(&u_nan, &base_v, None, None).is_err());
        assert!(spin3_distance(&u_nan, &base_v, None).is_err());

        let mut v_nan = base_v.clone();
        v_nan[0] = f64::NAN;
        assert!(spin3_struct(&base_u, &v_nan, None, None).is_err());
        assert!(spin3_distance(&base_u, &v_nan, None).is_err());

        let mut u_inf = base_u.clone();
        u_inf[0] = f64::INFINITY;
        assert!(spin3_struct(&u_inf, &base_v, None, None).is_err());
        assert!(spin3_distance(&u_inf, &base_v, None).is_err());

        let mut v_inf = base_v.clone();
        v_inf[0] = f64::NEG_INFINITY;
        assert!(spin3_struct(&base_u, &v_inf, None, None).is_err());
        assert!(spin3_distance(&base_u, &v_inf, None).is_err());
    }

    #[test]
    fn rejects_nonfinite_alpha() {
        let u = vec![0.1; 8];
        let v = vec![0.2; 8];
        assert!(spin3_distance(&u, &v, Some(f64::NAN)).is_err());
        assert!(spin3_distance(&u, &v, Some(f64::INFINITY)).is_err());
        assert!(spin3_distance(&u, &v, Some(f64::NEG_INFINITY)).is_err());
    }

    #[cfg(feature = "inspect")]
    #[test]
    fn rejects_nonfinite_alpha_in_inspect() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let u = PyList::new_bound(py, vec![1.0_f64; 8]);
            let v = PyList::new_bound(py, vec![2.0_f64; 8]);
            assert!(spin3_inspect(py, u.as_any(), v.as_any(), Some(f64::NAN)).is_err());
            assert!(spin3_inspect(py, u.as_any(), v.as_any(), Some(f64::INFINITY)).is_err());
            assert!(
                spin3_inspect_dev(py, u.as_any(), v.as_any(), 3, 12.0, Some(f64::NAN)).is_err()
            );
            assert!(
                spin3_inspect_dev(py, u.as_any(), v.as_any(), 3, 12.0, Some(f64::INFINITY))
                    .is_err()
            );
        });
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

        let d = spin3_distance(&u, &u_shift, Some(1.0)).unwrap();
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

    #[test]
    fn extract_view_vec_fallback() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let list = PyList::new_bound(py, vec![1.0_f64, 2.0, 3.0]);
            let view = extract_view(list.as_any()).unwrap();
            match view {
                DataView::Owned(values, _) => assert_eq!(values, vec![1.0, 2.0, 3.0]),
                #[cfg(feature = "numpy-support")]
                DataView::Numpy(_) => panic!("expected owned fallback for list input"),
            }
        });
    }

    #[cfg(feature = "numpy-support")]
    #[test]
    fn extract_view_numpy_borrow() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let array = PyArray1::from_vec_bound(py, vec![1.0_f64, 2.0, 3.0]);
            let view = extract_view(array.as_any()).unwrap();
            match view {
                DataView::Numpy(array) => {
                    assert_eq!(array.as_slice().unwrap(), &[1.0, 2.0, 3.0])
                }
                DataView::Owned(_, _) => panic!("expected numpy borrow"),
            }
        });
    }
}
