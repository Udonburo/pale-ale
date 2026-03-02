use pale_ale_rotor::{even_masks_v1, grade_of_mask, inner, Even128};

pub(crate) fn projective_chordal_distance(left: &Even128, right: &Even128) -> f64 {
    let inn = inner(left, right);
    let a = inn.abs().min(1.0);
    (2.0 * (1.0 - a)).max(0.0).sqrt()
}

pub(crate) fn higher_grade_energy_ratio(raw: &Even128) -> f64 {
    let masks = even_masks_v1();
    let mut e_total = 0.0;
    let mut e_high = 0.0;
    for (idx, &coeff) in raw.coeffs.iter().enumerate() {
        let energy = coeff * coeff;
        e_total += energy;
        let grade = grade_of_mask(masks[idx]);
        if grade >= 4 {
            e_high += energy;
        }
    }
    if e_total > 0.0 {
        e_high / e_total
    } else {
        0.0
    }
}
