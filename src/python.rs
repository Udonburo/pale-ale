#![cfg(feature = "python")]

#[cfg(feature = "numpy-support")]
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::marker::PhantomData;

use crate::{spin3_components_with, spin3_distance, spin3_struct, Spin3Components};

#[cfg(feature = "python-inspect")]
use crate::{
    calculate_components, clamp, resolve_alpha, semantic_distance, validate_alpha, ROOT_COUNT,
    SNAP_SOFT_BETA, SNAP_SOFT_K,
};

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

#[cfg(feature = "python-inspect")]
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

#[cfg(feature = "python-inspect")]
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

#[cfg(feature = "python-inspect")]
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
    #[cfg(feature = "python-inspect")]
    m.add_function(wrap_pyfunction!(spin3_inspect, m)?)?;
    #[cfg(feature = "python-inspect")]
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

    #[cfg(feature = "python-inspect")]
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
