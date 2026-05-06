"""Conformance test: dcmrtstruct2nii vs RTMaskConformanceTest analytic ground truth.

Runs only when the `conformance` extra is installed:

    pip install -e .[conformance]
    pytest dcmrtstruct2nii/tests/test_conformance.py -v

Without the extra, the module is skipped (importorskip), so default test
runs are unaffected.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# If `rtmask_conformance` isn't installed, skip the whole module so the
# default `pytest -vv -s` run continues to pass on CI without the extra.
rtmask_conformance = pytest.importorskip(  # noqa: F841
    "rtmask_conformance",
    reason="install the `conformance` extra: pip install -e .[conformance]",
)

from rtmask_conformance import CONFORMANCE_ROIS, generate_fixture, load_config  # noqa: E402
from rtmask_conformance.generate import GenerateOptions  # noqa: E402
from rtmask_conformance.verify import Status, evaluate_one  # noqa: E402

from dcmrtstruct2nii import dcmrtstruct2nii  # noqa: E402

_CONFIG_YAML = Path(__file__).with_name("conformance.yaml")


@pytest.fixture(scope="session")
def conformance_fixture(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate the synthetic CT + RTSTRUCT + analytic GT NIfTIs once per session.

    ``n_quadrature=2`` keeps the fixture build under ~30 s; the published
    default of 8 is overkill for a CI gate.
    """
    out = tmp_path_factory.mktemp("conformance_fixture")
    generate_fixture(out, options=GenerateOptions(n_quadrature=2))
    return out


@pytest.fixture(scope="session")
def predictions(
    conformance_fixture: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Run dcmrtstruct2nii against the conformance fixture once per session.

    dcmrtstruct2nii writes ``mask_<roi>.nii.gz`` per ROI; the rtmask
    verifier expects ``<roi>.nii.gz``. We rename in place after conversion
    rather than asking the verifier to glob, because the rename is cheap
    and keeps the rtmask side of the contract narrow.
    """
    pred_dir = tmp_path_factory.mktemp("preds")

    dcmrtstruct2nii(
        rtstruct_file=str(conformance_fixture / "rtstruct" / "primitives_planar.dcm"),
        dicom_file=str(conformance_fixture / "refct"),
        output_path=str(pred_dir),
        structures=None,                # convert every ROI in the RTSTRUCT
        gzip=True,
        convert_original_dicom=False,   # skip image.nii.gz; the gate doesn't use it
        maskname_pattern=["ROIName"],   # filenames: mask_<roi>.nii.gz (no ROINumber)
    )

    for src in pred_dir.glob("mask_*.nii.gz"):
        roi = src.name[len("mask_"):]
        src.rename(pred_dir / roi)

    return pred_dir


@pytest.fixture(scope="session")
def conformance_config():
    """Resolve thresholds: env var > tests/conformance.yaml > package defaults."""
    config_path = os.environ.get("RTMASK_CONFORMANCE_CONFIG")
    if config_path is None and _CONFIG_YAML.is_file():
        config_path = str(_CONFIG_YAML)
    return load_config(config_path)


@pytest.mark.parametrize("roi", CONFORMANCE_ROIS)
def test_conformance(
    roi: str, conformance_fixture: Path, predictions: Path, conformance_config
) -> None:
    pred = predictions / f"{roi}.nii.gz"
    gt = conformance_fixture / "groundtruth" / f"{roi}.nii.gz"
    result = evaluate_one(roi, pred, gt, conformance_config)
    if result.status != Status.PASS:
        pytest.fail(
            f"{roi}: {result.status.value}\n"
            f"  violations: {result.violations}\n"
            f"  metrics:    {result.metrics}\n"
            f"  thresholds: {result.thresholds}"
        )
