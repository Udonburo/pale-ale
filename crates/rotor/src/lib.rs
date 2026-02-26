mod even128;

pub use even128::{
    basis_blade_even128, blade_mul_masks, blade_sign_swapcount_popcount_v1,
    embed_simple29_to_even128, even_index_of_mask, even_masks_v1,
    gate1_lex_bivector_index_for_pair, gate1_lex_bivector_index_to_even128_grade2_index,
    gate1_lex_bivector_index_to_even128_index, gate1_lex_pair_from_bivector_index,
    grade2_index_of_mask, grade_of_mask, inner, inner_via_scalar_part,
    left_fold_mul_time_reversed_normalize_once, mul_even128, n2, n2_sum_sq, n2_via_scalar_part,
    normalize, reverse, scalar_part, Even128, EvenError, ALGEBRA_ID, BLADE_SIGN_ID, COMPOSITION_ID,
    EMBED_ID, EVEN128_DIM, GRADE2_BLOCK_LEN, GRADE2_BLOCK_OFFSET, NORMALIZE_ID, REVERSE_ID,
};

pub const ROOT_DIM: usize = 8;
pub const BIV_DIM: usize = ROOT_DIM * (ROOT_DIM - 1) / 2;
pub const ROTOR_DIM: usize = 1 + BIV_DIM;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RotorConfig {
    pub tau_wedge: f64,
    pub tau_antipodal_dot: f64,
}

impl Default for RotorConfig {
    fn default() -> Self {
        Self {
            tau_wedge: 1e-6,
            tau_antipodal_dot: 1.0 - 1e-6,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vec8Error {
    NonFiniteComponent,
    ZeroOrNonFiniteNorm,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotorError {
    Vec8(Vec8Error),
    NonFiniteTheta,
    RenormFailure,
}

impl From<Vec8Error> for RotorError {
    fn from(value: Vec8Error) -> Self {
        RotorError::Vec8(value)
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RotorStep {
    Materialized {
        r29: [f64; ROTOR_DIM],
        theta: f64,
        is_collinear: bool,
    },
    AntipodalAngleOnly {
        theta: f64,
    },
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

#[inline]
fn norm29(a: &[f64; ROTOR_DIM]) -> f64 {
    let mut sum = 0.0;
    for value in a {
        sum += value * value;
    }
    sum.sqrt()
}

pub fn normalize_vec8(x: [f64; ROOT_DIM]) -> Result<[f64; ROOT_DIM], Vec8Error> {
    if x.iter().any(|v| !v.is_finite()) {
        return Err(Vec8Error::NonFiniteComponent);
    }
    let n = norm8(&x);
    if !n.is_finite() || n == 0.0 {
        return Err(Vec8Error::ZeroOrNonFiniteNorm);
    }
    let inv = 1.0 / n;
    let mut out = [0.0; ROOT_DIM];
    for i in 0..ROOT_DIM {
        out[i] = x[i] * inv;
    }
    Ok(out)
}

pub fn wedge_lex_i_lt_j(u: &[f64; ROOT_DIM], v: &[f64; ROOT_DIM]) -> [f64; BIV_DIM] {
    let mut wedge = [0.0; BIV_DIM];
    let mut k = 0;
    for i in 0..ROOT_DIM {
        for j in (i + 1)..ROOT_DIM {
            wedge[k] = u[i] * v[j] - u[j] * v[i];
            k += 1;
        }
    }
    wedge
}

#[inline]
pub fn wedge_norm(wedge: &[f64; BIV_DIM]) -> f64 {
    let mut sum = 0.0;
    for value in wedge {
        sum += value * value;
    }
    sum.sqrt()
}

pub fn simple_rotor29_doc_to_ans(
    doc_vec8: [f64; ROOT_DIM],
    ans_vec8: [f64; ROOT_DIM],
    config: RotorConfig,
) -> Result<RotorStep, RotorError> {
    let u = normalize_vec8(doc_vec8)?;
    let v = normalize_vec8(ans_vec8)?;

    let dot_raw = dot8(&u, &v);
    let dot = clamp(dot_raw, -1.0, 1.0);
    let wedge = wedge_lex_i_lt_j(&u, &v);
    let wedge_n = wedge_norm(&wedge);
    let theta_uv = wedge_n.atan2(dot);
    if !theta_uv.is_finite() {
        return Err(RotorError::NonFiniteTheta);
    }

    // Branch order is fixed by SSOT: antipodal -> collinear -> normal.
    if dot <= -config.tau_antipodal_dot {
        return Ok(RotorStep::AntipodalAngleOnly { theta: theta_uv });
    }

    if wedge_n <= config.tau_wedge && dot >= 0.0 {
        let mut r29 = [0.0; ROTOR_DIM];
        r29[0] = 1.0;
        return Ok(RotorStep::Materialized {
            r29,
            theta: theta_uv,
            is_collinear: true,
        });
    }

    let s = ((1.0 + dot) * 0.5).max(0.0).sqrt();
    let sin_half = ((1.0 - dot) * 0.5).max(0.0).sqrt();
    let mut r_pre = [0.0; ROTOR_DIM];
    r_pre[0] = s;

    if wedge_n > 0.0 {
        let inv_wedge = 1.0 / wedge_n;
        for i in 0..BIV_DIM {
            r_pre[1 + i] = wedge[i] * inv_wedge * sin_half;
        }
    }

    let r_norm = norm29(&r_pre);
    if !r_norm.is_finite() || r_norm == 0.0 {
        return Err(RotorError::RenormFailure);
    }
    let inv_r = 1.0 / r_norm;
    let mut r29 = [0.0; ROTOR_DIM];
    for i in 0..ROTOR_DIM {
        r29[i] = r_pre[i] * inv_r;
    }

    Ok(RotorStep::Materialized {
        r29,
        theta: theta_uv,
        is_collinear: false,
    })
}

pub fn proj_chordal_v1(r1: &[f64; ROTOR_DIM], r2: &[f64; ROTOR_DIM]) -> f64 {
    let mut inner = 0.0;
    for i in 0..ROTOR_DIM {
        inner += r1[i] * r2[i];
    }
    let a = inner.abs().min(1.0);
    let d2 = (2.0 * (1.0 - a)).max(0.0);
    d2.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_vec8_rejects_non_finite() {
        let mut x = [0.0; ROOT_DIM];
        x[0] = f64::NAN;
        assert_eq!(normalize_vec8(x), Err(Vec8Error::NonFiniteComponent));
    }

    #[test]
    fn normalize_vec8_rejects_zero_norm() {
        assert_eq!(
            normalize_vec8([0.0; ROOT_DIM]),
            Err(Vec8Error::ZeroOrNonFiniteNorm)
        );
    }

    #[test]
    fn wedge_basis_starts_at_01() {
        let mut u = [0.0; ROOT_DIM];
        let mut v = [0.0; ROOT_DIM];
        u[0] = 1.0;
        v[1] = 1.0;
        let wedge = wedge_lex_i_lt_j(&u, &v);
        assert_eq!(wedge[0], 1.0);
    }
}
