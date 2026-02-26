use crate::{BIV_DIM, ROOT_DIM, ROTOR_DIM};

pub const EVEN128_DIM: usize = 128;
pub const GRADE2_BLOCK_OFFSET: usize = 1;
pub const GRADE2_BLOCK_LEN: usize = 28;

pub const ALGEBRA_ID: &str = "cl8_even128_mask_grade_order_v1";
pub const BLADE_SIGN_ID: &str = "swapcount_popcount_v1";
pub const REVERSE_ID: &str = "reverse_grade_sign_v1";
pub const NORMALIZE_ID: &str = "scalar_part_a_mul_rev_a_v1";
pub const COMPOSITION_ID: &str = "strict_left_fold_time_reversed_normalize_once_v1";
pub const EMBED_ID: &str = "embed_simple29_to_even128_v1";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Even128 {
    pub coeffs: [f64; EVEN128_DIM],
}

impl Even128 {
    pub const fn zero() -> Self {
        Self {
            coeffs: [0.0; EVEN128_DIM],
        }
    }

    pub const fn identity() -> Self {
        let mut coeffs = [0.0; EVEN128_DIM];
        coeffs[0] = 1.0;
        Self { coeffs }
    }
}

impl Default for Even128 {
    fn default() -> Self {
        Self::zero()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvenError {
    NonFiniteNormSquared,
    NonPositiveNormSquared,
}

pub fn grade_of_mask(mask: u8) -> u32 {
    mask.count_ones()
}

pub fn even_masks_v1() -> [u8; EVEN128_DIM] {
    let mut out = [0u8; EVEN128_DIM];
    let mut idx = 0usize;
    for target_grade in [0u32, 2, 4, 6, 8] {
        let mut mask = 0u16;
        while mask <= u8::MAX as u16 {
            let mask_u8 = mask as u8;
            if grade_of_mask(mask_u8) == target_grade {
                out[idx] = mask_u8;
                idx += 1;
            }
            mask += 1;
        }
    }
    debug_assert_eq!(idx, EVEN128_DIM);
    out
}

pub fn even_index_of_mask(mask: u8) -> Option<usize> {
    if grade_of_mask(mask) % 2 == 1 {
        return None;
    }
    let masks = even_masks_v1();
    masks.iter().position(|&candidate| candidate == mask)
}

pub fn grade2_index_of_mask(mask: u8) -> Option<usize> {
    if grade_of_mask(mask) != 2 {
        return None;
    }
    let masks = even_masks_v1();
    masks
        .iter()
        .skip(GRADE2_BLOCK_OFFSET)
        .take(GRADE2_BLOCK_LEN)
        .position(|&candidate| candidate == mask)
}

pub fn gate1_lex_pair_from_bivector_index(index: usize) -> Option<(usize, usize)> {
    if index >= BIV_DIM {
        return None;
    }
    let mut cursor = 0usize;
    for i in 0..ROOT_DIM {
        for j in (i + 1)..ROOT_DIM {
            if cursor == index {
                return Some((i, j));
            }
            cursor += 1;
        }
    }
    None
}

pub fn gate1_lex_bivector_index_for_pair(i: usize, j: usize) -> Option<usize> {
    if i >= ROOT_DIM || j >= ROOT_DIM || i >= j {
        return None;
    }
    let mut cursor = 0usize;
    for ii in 0..ROOT_DIM {
        for jj in (ii + 1)..ROOT_DIM {
            if ii == i && jj == j {
                return Some(cursor);
            }
            cursor += 1;
        }
    }
    None
}

pub fn gate1_lex_bivector_index_to_even128_grade2_index(index: usize) -> Option<usize> {
    let (i, j) = gate1_lex_pair_from_bivector_index(index)?;
    let mask = ((1u16 << i) | (1u16 << j)) as u8;
    grade2_index_of_mask(mask)
}

pub fn gate1_lex_bivector_index_to_even128_index(index: usize) -> Option<usize> {
    let grade2_idx = gate1_lex_bivector_index_to_even128_grade2_index(index)?;
    Some(GRADE2_BLOCK_OFFSET + grade2_idx)
}

pub fn basis_blade_even128(mask: u8) -> Option<Even128> {
    let idx = even_index_of_mask(mask)?;
    let mut out = Even128::zero();
    out.coeffs[idx] = 1.0;
    Some(out)
}

pub fn blade_sign_swapcount_popcount_v1(a_mask: u8, b_mask: u8) -> i8 {
    let mut swaps = 0u32;
    let mut a = a_mask;
    while a != 0 {
        let bit = a.trailing_zeros();
        let lower_bits_mask = if bit == 0 {
            0u8
        } else {
            ((1u16 << bit) - 1) as u8
        };
        swaps += (b_mask & lower_bits_mask).count_ones();
        a &= a - 1;
    }
    if swaps % 2 == 0 {
        1
    } else {
        -1
    }
}

pub fn blade_mul_masks(a_mask: u8, b_mask: u8) -> (i8, u8) {
    let sign = blade_sign_swapcount_popcount_v1(a_mask, b_mask);
    (sign, a_mask ^ b_mask)
}

fn even_index_map_v1(masks: &[u8; EVEN128_DIM]) -> [i16; 256] {
    let mut index_map = [-1i16; 256];
    for (idx, &mask) in masks.iter().enumerate() {
        index_map[mask as usize] = idx as i16;
    }
    index_map
}

pub fn mul_even128(a: &Even128, b: &Even128) -> Even128 {
    let masks = even_masks_v1();
    let index_map = even_index_map_v1(&masks);
    let mut out = Even128::zero();

    for (i, &a_coeff) in a.coeffs.iter().enumerate() {
        if a_coeff == 0.0 {
            continue;
        }
        let a_mask = masks[i];
        for (j, &b_coeff) in b.coeffs.iter().enumerate() {
            if b_coeff == 0.0 {
                continue;
            }
            let b_mask = masks[j];
            let (sign, out_mask) = blade_mul_masks(a_mask, b_mask);
            let out_idx = index_map[out_mask as usize];
            debug_assert!(out_idx >= 0);
            out.coeffs[out_idx as usize] += a_coeff * b_coeff * f64::from(sign);
        }
    }

    out
}

fn reverse_sign_from_grade(grade: u32) -> f64 {
    let g = grade as i32;
    let parity = (g * (g - 1) / 2) & 1;
    if parity == 0 {
        1.0
    } else {
        -1.0
    }
}

pub fn reverse(a: &Even128) -> Even128 {
    let masks = even_masks_v1();
    let mut out = Even128::zero();
    for (idx, &coeff) in a.coeffs.iter().enumerate() {
        let grade = grade_of_mask(masks[idx]);
        out.coeffs[idx] = coeff * reverse_sign_from_grade(grade);
    }
    out
}

pub fn scalar_part(a: &Even128) -> f64 {
    a.coeffs[0]
}

pub fn n2_via_scalar_part(a: &Even128) -> f64 {
    scalar_part(&mul_even128(a, &reverse(a)))
}

pub fn n2_sum_sq(a: &Even128) -> f64 {
    a.coeffs.iter().map(|v| v * v).sum()
}

pub fn n2(a: &Even128) -> f64 {
    n2_sum_sq(a)
}

pub fn normalize(a: &Even128) -> Result<Even128, EvenError> {
    let norm2 = n2(a);
    if !norm2.is_finite() {
        return Err(EvenError::NonFiniteNormSquared);
    }
    if norm2 <= 0.0 {
        return Err(EvenError::NonPositiveNormSquared);
    }
    let inv = 1.0 / norm2.sqrt();
    let mut out = Even128::zero();
    for (idx, &coeff) in a.coeffs.iter().enumerate() {
        out.coeffs[idx] = coeff * inv;
    }
    Ok(out)
}

pub fn inner_via_scalar_part(a: &Even128, b: &Even128) -> f64 {
    scalar_part(&mul_even128(&reverse(a), b))
}

pub fn inner(a: &Even128, b: &Even128) -> f64 {
    let mut acc = 0.0;
    for idx in 0..EVEN128_DIM {
        acc += a.coeffs[idx] * b.coeffs[idx];
    }
    acc
}

pub fn left_fold_mul_time_reversed_normalize_once(
    rotors_time_order: &[Even128],
) -> Result<Even128, EvenError> {
    if rotors_time_order.is_empty() {
        return Ok(Even128::identity());
    }

    let mut acc = rotors_time_order[rotors_time_order.len() - 1];
    for rotor in rotors_time_order[..rotors_time_order.len() - 1]
        .iter()
        .rev()
    {
        acc = mul_even128(&acc, rotor);
    }
    normalize(&acc)
}

pub fn embed_simple29_to_even128(simple29: &[f64; ROTOR_DIM]) -> Even128 {
    let mut out = Even128::zero();
    out.coeffs[0] = simple29[0];
    for gate1_idx in 0..BIV_DIM {
        let even_idx = gate1_lex_bivector_index_to_even128_index(gate1_idx)
            .expect("all Gate1 bivector indices must map to Even128 grade-2 slots");
        out.coeffs[even_idx] = simple29[1 + gate1_idx];
    }
    out
}
