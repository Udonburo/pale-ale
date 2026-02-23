use approx::assert_abs_diff_eq;
use pale_ale_rotor::{
    proj_chordal_v1, simple_rotor29_doc_to_ans, RotorConfig, RotorStep, BIV_DIM, ROOT_DIM,
    ROTOR_DIM,
};
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

fn e(i: usize) -> [f64; ROOT_DIM] {
    let mut out = [0.0; ROOT_DIM];
    out[i] = 1.0;
    out
}

#[test]
fn d1_orthogonal_axis() {
    let u = e(0);
    let v = e(1);
    let step = simple_rotor29_doc_to_ans(u, v, RotorConfig::default()).expect("must build");

    match step {
        RotorStep::Materialized {
            r29,
            theta,
            is_collinear,
        } => {
            assert!(!is_collinear);
            assert_abs_diff_eq!(theta, FRAC_PI_2, epsilon = 1e-12);
            let half = (0.5_f64).sqrt();
            assert_abs_diff_eq!(r29[0], half, epsilon = 1e-12);
            assert_abs_diff_eq!(r29[1], half, epsilon = 1e-12);
            for value in r29.iter().skip(2) {
                assert_abs_diff_eq!(*value, 0.0, epsilon = 1e-12);
            }
        }
        RotorStep::AntipodalAngleOnly { .. } => panic!("must be materialized"),
    }
}

#[test]
fn d2_collinear_identity_fallback() {
    let u = e(0);
    let v = e(0);
    let step = simple_rotor29_doc_to_ans(u, v, RotorConfig::default()).expect("must build");

    match step {
        RotorStep::Materialized {
            r29,
            theta,
            is_collinear,
        } => {
            assert!(is_collinear);
            assert_abs_diff_eq!(theta, 0.0, epsilon = 1e-12);
            assert_abs_diff_eq!(r29[0], 1.0, epsilon = 1e-12);
            for value in r29.iter().skip(1) {
                assert_abs_diff_eq!(*value, 0.0, epsilon = 1e-12);
            }
        }
        RotorStep::AntipodalAngleOnly { .. } => panic!("must be materialized identity"),
    }
}

#[test]
fn d3_antipodal_angle_only_no_materialization() {
    let u = e(0);
    let mut v = e(0);
    v[0] = -1.0;
    let step = simple_rotor29_doc_to_ans(u, v, RotorConfig::default()).expect("must build");

    match step {
        RotorStep::AntipodalAngleOnly { theta } => {
            assert_abs_diff_eq!(theta, PI, epsilon = 1e-12);
        }
        RotorStep::Materialized { .. } => panic!("D3 must not materialize"),
    }
}

#[test]
fn d4_forty_five_degree_plane() {
    let u = e(0);
    let mut v = [0.0; ROOT_DIM];
    let inv = (0.5_f64).sqrt();
    v[0] = inv;
    v[1] = inv;
    let step = simple_rotor29_doc_to_ans(u, v, RotorConfig::default()).expect("must build");

    match step {
        RotorStep::Materialized {
            r29,
            theta,
            is_collinear,
        } => {
            assert!(!is_collinear);
            assert_abs_diff_eq!(theta, FRAC_PI_4, epsilon = 1e-12);
            let s_expected = ((1.0 + inv) * 0.5).sqrt();
            assert_abs_diff_eq!(r29[0], s_expected, epsilon = 1e-12);
            let sin_half = ((1.0 - inv) * 0.5).sqrt();
            assert_abs_diff_eq!(r29[1], sin_half, epsilon = 1e-12);
            for value in r29.iter().skip(2) {
                assert_abs_diff_eq!(*value, 0.0, epsilon = 1e-12);
            }
        }
        RotorStep::AntipodalAngleOnly { .. } => panic!("must be materialized"),
    }
}

#[test]
fn proj_chordal_identity_is_zero() {
    let mut r = [0.0; ROTOR_DIM];
    r[0] = 1.0;
    let d = proj_chordal_v1(&r, &r);
    assert_abs_diff_eq!(d, 0.0, epsilon = 1e-12);
}

#[test]
fn proj_chordal_is_projective_under_global_sign_flip() {
    let u = e(0);
    let v = e(1);
    let step = simple_rotor29_doc_to_ans(u, v, RotorConfig::default()).expect("must build");
    let r = match step {
        RotorStep::Materialized { r29, .. } => r29,
        RotorStep::AntipodalAngleOnly { .. } => panic!("must be materialized"),
    };
    let mut neg_r = [0.0; ROTOR_DIM];
    for i in 0..ROTOR_DIM {
        neg_r[i] = -r[i];
    }
    let d = proj_chordal_v1(&r, &neg_r);
    assert_abs_diff_eq!(d, 0.0, epsilon = 1e-12);
}

#[test]
fn dimensions_are_fixed() {
    assert_eq!(BIV_DIM, 28);
    assert_eq!(ROTOR_DIM, 29);
}

// --- Section 1.5 DoD tests: Vec8 acceptance ---

#[test]
fn vec8_already_unit_passes() {
    let u = e(0); // already unit
    let result = pale_ale_rotor::normalize_vec8(u);
    assert!(result.is_ok());
    let out = result.unwrap();
    assert_abs_diff_eq!(out[0], 1.0, epsilon = 1e-15);
    for &v in &out[1..] {
        assert_abs_diff_eq!(v, 0.0, epsilon = 1e-15);
    }
}

#[test]
fn vec8_non_unit_normalizes_to_unit() {
    let x = [3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // norm = 5
    let out = pale_ale_rotor::normalize_vec8(x).expect("must normalize");
    let norm_sq: f64 = out.iter().map(|v| v * v).sum();
    assert_abs_diff_eq!(norm_sq, 1.0, epsilon = 1e-14);
    assert_abs_diff_eq!(out[0], 0.6, epsilon = 1e-14);
    assert_abs_diff_eq!(out[1], 0.8, epsilon = 1e-14);
}

// --- Section 1.5 DoD tests: distance NaN guard ---

#[test]
fn proj_chordal_no_nan_on_boundary_rounding() {
    // Craft two rotor vectors whose inner product slightly exceeds 1.0
    // after floating-point accumulation, verifying the min(1,abs(inner))
    // and max(0,...) guards prevent NaN.
    let mut r1 = [0.0; ROTOR_DIM];
    let mut r2 = [0.0; ROTOR_DIM];
    // Both are unit vectors along scalar axis
    r1[0] = 1.0;
    r2[0] = 1.0;
    // Sprinkle tiny perturbations that could cause abs(inner) > 1.0
    r1[1] = 1e-16;
    r2[1] = 1e-16;
    let d = proj_chordal_v1(&r1, &r2);
    assert!(d.is_finite(), "distance must not be NaN");
    assert!(d >= 0.0, "distance must be non-negative");
}
