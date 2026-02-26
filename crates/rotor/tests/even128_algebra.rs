use approx::assert_abs_diff_eq;
use pale_ale_rotor::{
    basis_blade_even128, blade_mul_masks, embed_simple29_to_even128, even_masks_v1,
    gate1_lex_bivector_index_for_pair, gate1_lex_bivector_index_to_even128_grade2_index,
    gate1_lex_pair_from_bivector_index, grade2_index_of_mask, grade_of_mask, inner,
    inner_via_scalar_part, left_fold_mul_time_reversed_normalize_once, mul_even128, n2, n2_sum_sq,
    n2_via_scalar_part, normalize, reverse, BIV_DIM, COMPOSITION_ID, EMBED_ID, EVEN128_DIM,
    NORMALIZE_ID, REVERSE_ID, ROTOR_DIM,
};
use std::f64::consts::FRAC_1_SQRT_2;

#[derive(Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn next_f64_sym(&mut self) -> f64 {
        let x = self.next_u64() >> 11;
        let unit = (x as f64) / ((1u64 << 53) as f64);
        (2.0 * unit) - 1.0
    }
}

fn random_sparse_even128(rng: &mut SplitMix64, nonzeros: usize) -> pale_ale_rotor::Even128 {
    let mut out = pale_ale_rotor::Even128::zero();
    for _ in 0..nonzeros {
        let idx = (rng.next_u64() as usize) % EVEN128_DIM;
        let mut v = rng.next_f64_sym();
        if v == 0.0 {
            v = 1.0;
        }
        out.coeffs[idx] += v;
    }
    out
}

fn plane_rotor_simple29(i: usize, j: usize, theta: f64) -> [f64; ROTOR_DIM] {
    let mut out = [0.0; ROTOR_DIM];
    let idx = gate1_lex_bivector_index_for_pair(i, j).expect("valid plane pair");
    out[0] = (0.5 * theta).cos();
    out[1 + idx] = (0.5 * theta).sin();
    out
}

fn assert_even_close(a: &pale_ale_rotor::Even128, b: &pale_ale_rotor::Even128, eps: f64) {
    for idx in 0..EVEN128_DIM {
        assert_abs_diff_eq!(a.coeffs[idx], b.coeffs[idx], epsilon = eps);
    }
}

fn l2_distance(a: &pale_ale_rotor::Even128, b: &pale_ale_rotor::Even128) -> f64 {
    let mut acc = 0.0;
    for idx in 0..EVEN128_DIM {
        let d = a.coeffs[idx] - b.coeffs[idx];
        acc += d * d;
    }
    acc.sqrt()
}

#[test]
fn exported_gate2_ids_are_exact() {
    assert_eq!(
        pale_ale_rotor::ALGEBRA_ID,
        "cl8_even128_mask_grade_order_v1"
    );
    assert_eq!(pale_ale_rotor::BLADE_SIGN_ID, "swapcount_popcount_v1");
    assert_eq!(REVERSE_ID, "reverse_grade_sign_v1");
    assert_eq!(NORMALIZE_ID, "scalar_part_a_mul_rev_a_v1");
    assert_eq!(
        COMPOSITION_ID,
        "strict_left_fold_time_reversed_normalize_once_v1"
    );
    assert_eq!(EMBED_ID, "embed_simple29_to_even128_v1");
}

#[test]
fn basis_counts_and_ordering_are_canonical() {
    let masks = even_masks_v1();
    assert_eq!(masks.len(), EVEN128_DIM);

    let mut counts = [0usize; 9];
    for &mask in &masks {
        counts[grade_of_mask(mask) as usize] += 1;
    }
    assert_eq!(counts[0], 1);
    assert_eq!(counts[2], 28);
    assert_eq!(counts[4], 70);
    assert_eq!(counts[6], 28);
    assert_eq!(counts[8], 1);
    assert_eq!(counts.iter().sum::<usize>(), EVEN128_DIM);

    let grade_blocks = [(0u32, 1usize), (2, 28), (4, 70), (6, 28), (8, 1)];
    let mut start = 0usize;
    for (grade, block_len) in grade_blocks {
        let end = start + block_len;
        let block = &masks[start..end];
        for &mask in block {
            assert_eq!(grade_of_mask(mask), grade);
        }
        for pair in block.windows(2) {
            assert!(pair[0] < pair[1]);
        }
        start = end;
    }
}

#[test]
fn grade2_remap_is_correct_and_divergence_is_fixed() {
    let mut seen = [false; BIV_DIM];
    for gate1_idx in 0..BIV_DIM {
        let (i, j) = gate1_lex_pair_from_bivector_index(gate1_idx).expect("pair exists");
        let mask = ((1u16 << i) | (1u16 << j)) as u8;
        let g2_idx =
            gate1_lex_bivector_index_to_even128_grade2_index(gate1_idx).expect("grade2 map");
        assert!(g2_idx < BIV_DIM);
        assert!(
            !seen[g2_idx],
            "duplicate mapping for grade2 index {}",
            g2_idx
        );
        seen[g2_idx] = true;
        assert_eq!(grade2_index_of_mask(mask), Some(g2_idx));
    }
    assert!(seen.iter().all(|v| *v));

    let (i2, j2) = gate1_lex_pair_from_bivector_index(2).expect("gate1 index 2");
    assert_eq!((i2, j2), (0, 3));
    let mask_gate1_k2 = ((1u16 << i2) | (1u16 << j2)) as u8;
    assert_eq!(mask_gate1_k2, 9);

    let masks = even_masks_v1();
    let grade2_slice = &masks[1..(1 + BIV_DIM)];
    assert_eq!(grade2_slice[2], 6);
    assert_eq!(grade2_index_of_mask(6), Some(2));
    assert_eq!(gate1_lex_bivector_index_to_even128_grade2_index(2), Some(3));
}

#[test]
fn sign_anticommutation_and_bivector_square_sanity() {
    for i in 0..8 {
        for j in 0..8 {
            if i == j {
                continue;
            }
            let ei = 1u8 << i;
            let ej = 1u8 << j;
            let (s_ij, m_ij) = blade_mul_masks(ei, ej);
            let (s_ji, m_ji) = blade_mul_masks(ej, ei);
            assert_eq!(m_ij, m_ji);
            assert_eq!(s_ij, -s_ji);
        }
    }

    for i in 0..8 {
        for j in (i + 1)..8 {
            let biv_mask = ((1u16 << i) | (1u16 << j)) as u8;
            let (sign, out_mask) = blade_mul_masks(biv_mask, biv_mask);
            assert_eq!(out_mask, 0);
            assert_eq!(sign, -1);

            let biv = basis_blade_even128(biv_mask).expect("even bivector basis");
            let sq = mul_even128(&biv, &biv);
            assert_abs_diff_eq!(sq.coeffs[0], -1.0, epsilon = 1e-12);
            for &coeff in sq.coeffs.iter().skip(1) {
                assert_abs_diff_eq!(coeff, 0.0, epsilon = 1e-12);
            }
        }
    }
}

#[test]
fn reverse_is_involutive_and_grade_signs_are_correct() {
    let mut rng = SplitMix64::new(0xD1CE_1234_5678_9ABC);
    for _ in 0..64 {
        let a = random_sparse_even128(&mut rng, 10);
        let rr = reverse(&reverse(&a));
        assert_even_close(&rr, &a, 1e-12);
    }

    let scalar = basis_blade_even128(0).expect("scalar basis");
    let rev_scalar = reverse(&scalar);
    assert_even_close(&rev_scalar, &scalar, 0.0);

    let biv = basis_blade_even128(0b0000_0011).expect("e01");
    let rev_biv = reverse(&biv);
    assert_abs_diff_eq!(rev_biv.coeffs[1], -1.0, epsilon = 1e-12);
    for &coeff in rev_biv.coeffs.iter().skip(2) {
        assert_abs_diff_eq!(coeff, 0.0, epsilon = 1e-12);
    }
}

#[test]
fn norm_lemma_holds_and_embedded_d1_is_unit_norm() {
    let mut rng = SplitMix64::new(0xACE0_BAAD_F00D_9999);
    for _ in 0..128 {
        let a = random_sparse_even128(&mut rng, 12);
        let n2_prod = n2_via_scalar_part(&a);
        let n2_dot = n2_sum_sq(&a);
        assert_abs_diff_eq!(n2_prod, n2_dot, epsilon = 1e-10);
        assert_abs_diff_eq!(n2(&a), n2_dot, epsilon = 1e-12);
    }

    let mut d1 = [0.0; ROTOR_DIM];
    d1[0] = FRAC_1_SQRT_2;
    d1[1] = FRAC_1_SQRT_2;
    let embedded = embed_simple29_to_even128(&d1);
    let n2_embedded = n2(&embedded);
    assert_abs_diff_eq!(n2_embedded, 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(n2_via_scalar_part(&embedded), 1.0, epsilon = 1e-12);
}

#[test]
fn inner_lemma_matches_dot_product() {
    let mut rng = SplitMix64::new(0x1122_3344_5566_7788);
    for _ in 0..128 {
        let a = random_sparse_even128(&mut rng, 11);
        let b = random_sparse_even128(&mut rng, 11);
        let inner_prod = inner_via_scalar_part(&a, &b);
        let dot = inner(&a, &b);
        assert_abs_diff_eq!(inner_prod, dot, epsilon = 1e-10);
    }
}

#[test]
fn associativity_holds_for_random_sparse_and_basis_blades() {
    let mut rng = SplitMix64::new(0xBEEF_BABE_CAFE_D00D);
    for _ in 0..64 {
        let a = random_sparse_even128(&mut rng, 8);
        let b = random_sparse_even128(&mut rng, 8);
        let c = random_sparse_even128(&mut rng, 8);
        let left = mul_even128(&mul_even128(&a, &b), &c);
        let right = mul_even128(&a, &mul_even128(&b, &c));
        assert_even_close(&left, &right, 1e-10);
    }

    let masks = even_masks_v1();
    for &a in masks.iter().take(16) {
        for &b in masks.iter().skip(8).take(16) {
            for &c in masks.iter().skip(24).take(16) {
                let (s_ab, m_ab) = blade_mul_masks(a, b);
                let (s_left, m_left) = blade_mul_masks(m_ab, c);
                let left_sign = s_ab * s_left;

                let (s_bc, m_bc) = blade_mul_masks(b, c);
                let (s_right, m_right) = blade_mul_masks(a, m_bc);
                let right_sign = s_bc * s_right;

                assert_eq!(m_left, m_right);
                assert_eq!(left_sign, right_sign);
            }
        }
    }
}

#[test]
fn time_reversed_left_fold_normalize_once_matches_spec() {
    let r0 = embed_simple29_to_even128(&plane_rotor_simple29(0, 1, 0.41));
    let r1 = embed_simple29_to_even128(&plane_rotor_simple29(0, 2, -0.29));
    let r2 = embed_simple29_to_even128(&plane_rotor_simple29(1, 2, 0.63));

    let got = left_fold_mul_time_reversed_normalize_once(&[r0, r1, r2]).expect("compose");

    let expected = normalize(&mul_even128(&mul_even128(&r2, &r1), &r0)).expect("expected");
    assert_even_close(&got, &expected, 1e-12);

    let wrong = normalize(&mul_even128(&mul_even128(&r0, &r1), &r2)).expect("wrong order");
    let delta = l2_distance(&got, &wrong);
    assert!(
        delta > 1e-6,
        "time-reversed and time-forward products must diverge for this fixture (delta={delta})"
    );
}
