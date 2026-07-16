"""Unit coverage for NISAR GUNW metadata parsing in ``mintpy.prep_nisar``.

The tests build minimal synthetic GUNW HDF5 files with ``h5py`` (a few KB,
no GDAL warp and no network) and exercise the pure-Python / h5py parsing
layer of ``prep_nisar``: frequency resolution, required-path discovery and
metadata extraction.

They lock in regression coverage for the metadata reader, most importantly
issue #1485: the reader must read ``.../radarGrid/referenceSlantRange`` and
not a bare ``slantRange`` dataset (which does not exist in real GUNW files
and used to raise ``KeyError``).

The GDAL-backed data path (``read_subset`` -> ``_warp_to_grid_mem`` and the
radarGrid interpolation) is intentionally out of scope here; it is validated
against a real sample GUNW product separately.
"""

import datetime

import h5py
import numpy as np
import pytest

from mintpy import prep_nisar
from mintpy.constants import EARTH_RADIUS, SPEED_OF_LIGHT

GUNW_ROOT = "/science/LSAR/GUNW"
IDENT = "/science/LSAR/identification"
RADARGRID = f"{GUNW_ROOT}/metadata/radarGrid"


def _unw_root(freq):
    return f"{GUNW_ROOT}/grids/{freq}/unwrappedInterferogram"


def _params_root(freq):
    return (
        f"{GUNW_ROOT}/metadata/processingInformation/parameters/"
        f"unwrappedInterferogram/{freq}"
    )


def _build_gunw(
    path,
    *,
    frequencies=("frequencyA",),
    polarization="HH",
    epsg=32612,
    x_start=400000.0,
    y_start=4300000.0,
    x_step=80.0,
    y_step=-80.0,
    nx=5,
    ny=4,
    center_frequency=1.2575e9,
    slant_range_name="referenceSlantRange",
    orbit_direction="ASCENDING",
    mission_id="NISAR",
    start_time="2008-10-12T06:09:11.000000",
    end_time="2008-10-12T06:09:25.000000",
    secondary_start_time="2008-11-27T06:10:00.000000",
    range_looks=20,
    azimuth_looks=4,
    bperp=50.0,
    include_ifgram=True,
    include_ion=True,
    include_tropo=True,
    include_set=True,
    include_mask=False,
):
    """Write a minimal synthetic NISAR GUNW HDF5 product to ``path``.

    Datasets mirror the paths that the GDAL-free readers touch. String fields
    are stored as byte strings to exercise the ``.decode('utf-8')`` path, and
    spacing / looks / EPSG are stored as scalar datasets on purpose.
    """
    xcoord = x_start + np.arange(nx) * x_step
    ycoord = y_start + np.arange(ny) * y_step
    shape = (ny, nx)

    with h5py.File(path, "w") as f:
        # identification group -- byte strings exercise .decode('utf-8')
        f[f"{IDENT}/orbitPassDirection"] = np.bytes_(orbit_direction)
        f[f"{IDENT}/missionId"] = np.bytes_(mission_id)
        f[f"{IDENT}/referenceZeroDopplerStartTime"] = np.bytes_(start_time)
        f[f"{IDENT}/referenceZeroDopplerEndTime"] = np.bytes_(end_time)
        f[f"{IDENT}/secondaryZeroDopplerStartTime"] = np.bytes_(secondary_start_time)

        # radarGrid cube: coords + reference slant range (issue #1485 dataset)
        f[f"{RADARGRID}/xCoordinates"] = xcoord
        f[f"{RADARGRID}/yCoordinates"] = ycoord
        slant_range = np.full(shape, 8.0e5, dtype=np.float64) + np.arange(nx)
        f[f"{RADARGRID}/{slant_range_name}"] = slant_range
        # extra radarGrid fields touched by the interpolation/baseline readers
        f[f"{RADARGRID}/heightAboveEllipsoid"] = np.zeros(shape, np.float64)
        f[f"{RADARGRID}/incidenceAngle"] = np.full(shape, 30.0, np.float64)
        f[f"{RADARGRID}/losUnitVectorX"] = np.full(shape, 0.6, np.float64)
        f[f"{RADARGRID}/losUnitVectorY"] = np.full(shape, 0.8, np.float64)
        f[f"{RADARGRID}/perpendicularBaseline"] = np.full(shape, bperp, np.float64)
        if include_tropo:
            f[f"{RADARGRID}/wetTroposphericPhaseScreen"] = np.zeros(shape, np.float32)
            f[f"{RADARGRID}/hydrostaticTroposphericPhaseScreen"] = np.zeros(shape, np.float32)
        if include_set:
            f[f"{RADARGRID}/slantRangeSolidEarthTidesPhase"] = np.zeros(shape, np.float32)

        for freq in frequencies:
            grids = f"{GUNW_ROOT}/grids/{freq}"
            # scalar center frequency
            f[f"{grids}/centerFrequency"] = np.float64(center_frequency)

            unw = f"{_unw_root(freq)}/{polarization}"
            f[f"{unw}/xCoordinates"] = xcoord
            f[f"{unw}/yCoordinates"] = ycoord
            # scalar spacing datasets
            f[f"{unw}/xCoordinateSpacing"] = np.float64(x_step)
            f[f"{unw}/yCoordinateSpacing"] = np.float64(y_step)
            # scalar integer EPSG code
            f[f"{unw}/projection"] = np.int32(epsg)
            if include_ifgram:
                f[f"{unw}/unwrappedPhase"] = np.zeros(shape, np.float32)
                f[f"{unw}/coherenceMagnitude"] = np.ones(shape, np.float32)
                f[f"{unw}/connectedComponents"] = np.zeros(shape, np.int32)
            if include_ion:
                f[f"{unw}/ionospherePhaseScreen"] = np.zeros(shape, np.float32)
            if include_mask:
                # mask sits at the unwrappedInterferogram level (not per-pol).
                # value 11 -> ref/sec subswath 1, not water -> land & valid.
                grids_root = f"{GUNW_ROOT}/grids/{freq}/unwrappedInterferogram"
                f[f"{grids_root}/mask"] = np.full(shape, 11, np.int32)

            params = _params_root(freq)
            # scalar look counts
            f[f"{params}/numberOfRangeLooks"] = np.int32(range_looks)
            f[f"{params}/numberOfAzimuthLooks"] = np.int32(azimuth_looks)

    return str(path)


# ---------------------------------------------------------------------------
# Tier A -- pure helpers (no HDF5)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value, expected",
    [
        (None, "frequencyA"),
        ("auto", "frequencyA"),
        ("A", "frequencyA"),
        ("B", "frequencyB"),
        ("frequencyA", "frequencyA"),
        ("frequencyB", "frequencyB"),
    ],
)
def test_normalize_frequency_valid(value, expected):
    assert prep_nisar._normalize_frequency(value) == expected


def test_normalize_frequency_invalid():
    with pytest.raises(ValueError, match="auto, A, B"):
        prep_nisar._normalize_frequency("C")


def test_dataset_path_builders():
    assert prep_nisar._dataset_root_unw("frequencyA").endswith(
        "grids/frequencyA/unwrappedInterferogram"
    )
    assert prep_nisar._center_frequency_path("frequencyB").endswith(
        "grids/frequencyB/centerFrequency"
    )
    datasets = prep_nisar._datasets_for_pol("HH", "frequencyA")
    assert datasets["unw"].endswith("frequencyA/unwrappedInterferogram/HH/unwrappedPhase")
    assert datasets["cor"].endswith("/HH/coherenceMagnitude")
    assert datasets["connComp"].endswith("/HH/connectedComponents")
    assert datasets["epsg"].endswith("/HH/projection")


@pytest.mark.parametrize(
    "stack_type, tail",
    [
        ("ifgram", "unwrappedPhase"),
        ("ion", "ionospherePhaseScreen"),
        ("tropo", "wetTroposphericPhaseScreen"),
        ("set", "slantRangeSolidEarthTidesPhase"),
    ],
)
def test_required_paths_for_stack_type(stack_type, tail):
    paths = prep_nisar._required_paths_for_stack_type(stack_type, "HH", "frequencyA")
    assert any(p.endswith(tail) for p in paths)


def test_required_paths_for_stack_type_unknown():
    with pytest.raises(ValueError, match="Unsupported stack_type"):
        prep_nisar._required_paths_for_stack_type("bogus", "HH", "frequencyA")


def test_processinfo_reads_reference_slant_range():
    """Regression sentinel for #1485: the reader targets referenceSlantRange."""
    slant_path = prep_nisar.PROCESSINFO["rdr_slant_range"]
    assert slant_path.endswith("/referenceSlantRange")
    # guard against a regression back to the bare (non-existent) dataset name
    assert not slant_path.endswith("/slantRange")


# ---------------------------------------------------------------------------
# Tier B -- HDF5 traversal: frequency resolution and required-path discovery
# ---------------------------------------------------------------------------
def test_resolve_frequency_auto_and_explicit(tmp_path):
    gunw = _build_gunw(tmp_path / "gunw_A.h5", frequencies=("frequencyA",))
    assert prep_nisar._resolve_frequency(gunw, None, "HH") == "frequencyA"
    assert prep_nisar._resolve_frequency(gunw, "auto", "HH") == "frequencyA"
    assert prep_nisar._resolve_frequency(gunw, "A", "HH") == "frequencyA"


def test_resolve_frequency_missing_b_raises(tmp_path):
    gunw = _build_gunw(tmp_path / "gunw_A.h5", frequencies=("frequencyA",))
    with pytest.raises(ValueError, match="frequencyB"):
        prep_nisar._resolve_frequency(gunw, "B", "HH")


def test_resolve_frequency_missing_polarization_raises(tmp_path):
    gunw = _build_gunw(tmp_path / "gunw_hh.h5", polarization="HH")
    with pytest.raises(ValueError, match="VV"):
        prep_nisar._resolve_frequency(gunw, "A", "VV")


def test_missing_required_paths_complete(tmp_path):
    gunw = _build_gunw(tmp_path / "gunw_full.h5")
    for stack_type in ["ifgram", "ion", "tropo", "set"]:
        missing = prep_nisar._missing_required_paths(
            [gunw], stack_type, "HH", "frequencyA"
        )
        assert missing == [], f"{stack_type} should have no missing paths"


def test_missing_required_paths_detects_missing(tmp_path):
    # build a file whose ifgram layer lacks connectedComponents
    gunw = tmp_path / "gunw_partial.h5"
    _build_gunw(gunw, include_ifgram=True)
    with h5py.File(gunw, "a") as f:
        del f[f"{_unw_root('frequencyA')}/HH/connectedComponents"]

    missing = prep_nisar._missing_required_paths(
        [str(gunw)], "ifgram", "HH", "frequencyA"
    )
    assert len(missing) == 1
    assert missing[0][1].endswith("/HH/connectedComponents")


# ---------------------------------------------------------------------------
# Tier C -- extract_metadata (the #1485 regression target)
# ---------------------------------------------------------------------------
def test_extract_metadata_utm(tmp_path):
    gunw = _build_gunw(tmp_path / "gunw_utm.h5", epsg=32612, nx=5, ny=4)
    meta, bounds = prep_nisar.extract_metadata([gunw])

    assert meta["EPSG"] == 32612
    assert meta["X_UNIT"] == "meters"
    assert meta["Y_UNIT"] == "meters"
    assert meta["UTM_ZONE"] == "12N"
    assert meta["ORBIT_DIRECTION"] == "ASCENDING"
    assert meta["POLARIZATION"] == "HH"
    assert meta["PLATFORM"] == "NISAR"
    assert int(meta["RLOOKS"]) == 20
    assert int(meta["ALOOKS"]) == 4
    assert int(meta["LENGTH"]) == 4
    assert int(meta["WIDTH"]) == 5
    assert meta["X_STEP"] == pytest.approx(80.0)
    assert meta["Y_STEP"] == pytest.approx(-80.0)
    assert meta["HEIGHT"] == 747000
    assert float(meta["EARTH_RADIUS"]) == pytest.approx(EARTH_RADIUS)
    assert float(meta["WAVELENGTH"]) == pytest.approx(SPEED_OF_LIGHT / 1.2575e9)
    # STARTING_RANGE comes from radarGrid/referenceSlantRange (min = 8.0e5)
    assert float(meta["STARTING_RANGE"]) == pytest.approx(8.0e5)
    # mid of 06:09:11 -> 06:09:25 is 06:09:18 = 22158 s past midnight
    assert float(meta["CENTER_LINE_UTC"]) == pytest.approx(22158.0)
    assert len(bounds) == 4


def test_extract_metadata_southern_utm_zone(tmp_path):
    gunw = _build_gunw(tmp_path / "gunw_utm_s.h5", epsg=32712)
    meta, _ = prep_nisar.extract_metadata([gunw])
    assert meta["UTM_ZONE"] == "12S"


def test_extract_metadata_geographic_4326(tmp_path):
    gunw = _build_gunw(
        tmp_path / "gunw_geo.h5",
        epsg=4326,
        x_start=-78.0,
        y_start=0.98,
        x_step=0.001,
        y_step=-0.001,
    )
    meta, _ = prep_nisar.extract_metadata([gunw])
    assert meta["EPSG"] == 4326
    assert meta["X_UNIT"] == "degree"
    assert meta["Y_UNIT"] == "degree"
    assert "UTM_ZONE" not in meta


def test_extract_metadata_ascending_y(tmp_path):
    # ascending y coordinates (positive y_step) must still parse cleanly
    gunw = _build_gunw(tmp_path / "gunw_asc_y.h5", y_start=4200000.0, y_step=80.0)
    meta, _ = prep_nisar.extract_metadata([gunw])
    assert meta["Y_STEP"] == pytest.approx(80.0)
    assert int(meta["LENGTH"]) == 4
    assert int(meta["WIDTH"]) == 5


def test_extract_metadata_requires_reference_slant_range(tmp_path):
    """Guard #1485: a file with a bare ``slantRange`` (and no
    ``referenceSlantRange``) must fail -- the old buggy behavior."""
    gunw = _build_gunw(
        tmp_path / "gunw_bad_slant.h5", slant_range_name="slantRange"
    )
    # match on the dataset name so an unrelated KeyError cannot pass this guard
    with pytest.raises(KeyError, match="referenceSlantRange"):
        prep_nisar.extract_metadata([gunw])


def test_extract_metadata_center_line_utc_matches_manual(tmp_path):
    start = "2020-01-01T00:00:00.000000"
    end = "2020-01-01T00:01:00.000000"
    gunw = _build_gunw(
        tmp_path / "gunw_utc.h5", start_time=start, end_time=end
    )
    meta, _ = prep_nisar.extract_metadata([gunw])
    t0 = datetime.datetime.fromisoformat(start)
    t1 = datetime.datetime.fromisoformat(end)
    t_mid = t0 + (t1 - t0) / 2.0
    expected = (t_mid - datetime.datetime(2020, 1, 1)).total_seconds()
    assert float(meta["CENTER_LINE_UTC"]) == pytest.approx(expected)


def test_extract_metadata_origin_and_bounds_values(tmp_path):
    """Pin the half-pixel origin correction and common bounds numerically."""
    gunw = _build_gunw(tmp_path / "gunw_origin.h5")  # default 5x4 grid
    meta, bounds = prep_nisar.extract_metadata([gunw])
    # X_FIRST = min(x) - x_step/2 = 400000 - 40; Y_FIRST = max(y) - y_step/2
    assert float(meta["X_FIRST"]) == pytest.approx(399960.0)
    assert float(meta["Y_FIRST"]) == pytest.approx(4300040.0)
    # bounds = (west, south, east, north) at pixel centers
    assert tuple(pytest.approx(v) for v in bounds) == (
        pytest.approx(400000.0),
        pytest.approx(4299760.0),
        pytest.approx(400320.0),
        pytest.approx(4300000.0),
    )


# ---------------------------------------------------------------------------
# frequencyB positive path (not just normalization / missing-B)
# ---------------------------------------------------------------------------
def test_resolve_frequency_b_positive(tmp_path):
    gunw = _build_gunw(
        tmp_path / "gunw_ab.h5", frequencies=("frequencyA", "frequencyB")
    )
    assert prep_nisar._resolve_frequency(gunw, "B", "HH") == "frequencyB"
    # auto still prefers frequencyA even when B exists
    assert prep_nisar._resolve_frequency(gunw, "auto", "HH") == "frequencyA"


def test_extract_metadata_frequency_b(tmp_path):
    gunw = _build_gunw(
        tmp_path / "gunw_ab.h5", frequencies=("frequencyA", "frequencyB")
    )
    meta, _ = prep_nisar.extract_metadata([gunw], frequency="frequencyB")
    assert meta["POLARIZATION"] == "HH"
    assert int(meta["WIDTH"]) == 5
    assert float(meta["STARTING_RANGE"]) == pytest.approx(8.0e5)


# ---------------------------------------------------------------------------
# Multi-file common bounds
# ---------------------------------------------------------------------------
def test_get_raster_corners(tmp_path):
    gunw = _build_gunw(tmp_path / "gunw_corners.h5")
    west, south, east, north = prep_nisar.get_raster_corners(gunw)
    assert (west, south, east, north) == (
        pytest.approx(400000.0),
        pytest.approx(4299760.0),
        pytest.approx(400320.0),
        pytest.approx(4300000.0),
    )


def test_common_raster_bound_intersection(tmp_path):
    # second scene shifted east by two pixels (160 m) -> overlapping range
    f1 = _build_gunw(tmp_path / "gunw_a.h5", x_start=400000.0)
    f2 = _build_gunw(tmp_path / "gunw_b.h5", x_start=400160.0)
    west, south, east, north = prep_nisar.common_raster_bound([f1, f2])
    assert west == pytest.approx(400160.0)  # max of the two wests
    assert east == pytest.approx(400320.0)  # min of the two easts
    assert south == pytest.approx(4299760.0)
    assert north == pytest.approx(4300000.0)


def test_common_raster_bound_no_overlap_raises(tmp_path):
    f1 = _build_gunw(tmp_path / "gunw_a.h5", x_start=400000.0)
    f2 = _build_gunw(tmp_path / "gunw_far.h5", x_start=500000.0)
    with pytest.raises(ValueError, match="No common overlap"):
        prep_nisar.common_raster_bound([f1, f2])


# ---------------------------------------------------------------------------
# Tier D -- HDF5 data-array readers (subset, masks, baseline, target grid)
# These stay GDAL-free: they read h5py arrays but never warp a raster.
# ---------------------------------------------------------------------------
def test_read_subset_full(tmp_path):
    gunw = _build_gunw(tmp_path / "gunw_rs.h5")  # unw=0, cor=1, cc=0, ion=0
    data = prep_nisar.read_subset(gunw, None)
    assert data["unw_data"].shape == (4, 5)
    assert data["unw_data"].dtype == np.float32
    assert np.all(data["cor_data"] == 1.0)
    assert data["xybbox"] == (0, 0, 5, 4)


def test_read_subset_geometry_only(tmp_path):
    gunw = _build_gunw(tmp_path / "gunw_geo.h5")
    out = prep_nisar.read_subset(gunw, None, geometry=True)
    assert set(out) == {"xybbox"}
    assert out["xybbox"] == (0, 0, 5, 4)


def test_read_subset_bbox_window(tmp_path):
    gunw = _build_gunw(tmp_path / "gunw_win.h5")
    ramp = np.arange(20, dtype=np.float32).reshape(4, 5)
    with h5py.File(gunw, "a") as f:
        f[f"{_unw_root('frequencyA')}/HH/unwrappedPhase"][...] = ramp
    # bbox selects x in [400000, 400160] (cols 0-2) and y in [4299840, 4300000]
    # (rows 0-2) -> a 3x3 top-left window.
    bbox = (400000.0, 4299840.0, 400160.0, 4300000.0)
    sub = prep_nisar.read_subset(gunw, bbox)
    assert sub["unw_data"].shape == (3, 3)
    assert np.array_equal(sub["unw_data"], ramp[0:3, 0:3])


def test_read_subset_fillvalue_to_nan(tmp_path):
    gunw = _build_gunw(tmp_path / "gunw_fill.h5")
    with h5py.File(gunw, "a") as f:
        dset = f[f"{_unw_root('frequencyA')}/HH/unwrappedPhase"]
        dset[0, 0] = -9999.0
        dset.attrs["_FillValue"] = np.float32(-9999.0)
    data = prep_nisar.read_subset(gunw, None)
    assert np.isnan(data["unw_data"][0, 0])
    assert np.isfinite(data["unw_data"][1, 1])


def test_read_perpendicular_baseline_mean(tmp_path):
    gunw = _build_gunw(tmp_path / "gunw_bp.h5", bperp=50.0)
    assert float(prep_nisar._read_perpendicular_baseline(gunw)) == pytest.approx(50.0)


def test_read_perpendicular_baseline_skips_fill(tmp_path):
    gunw = _build_gunw(tmp_path / "gunw_bp2.h5", bperp=50.0)
    with h5py.File(gunw, "a") as f:
        dset = f[f"{RADARGRID}/perpendicularBaseline"]
        dset[0, 0] = -9999.0
        dset.attrs["_FillValue"] = -9999.0
    # the fill pixel is dropped; the remaining values are all 50 -> mean 50
    assert float(prep_nisar._read_perpendicular_baseline(gunw)) == pytest.approx(50.0)


def test_read_perpendicular_baseline_all_fill_raises(tmp_path):
    gunw = _build_gunw(tmp_path / "gunw_bp3.h5")
    with h5py.File(gunw, "a") as f:
        dset = f[f"{RADARGRID}/perpendicularBaseline"]
        dset[...] = -9999.0
        dset.attrs["_FillValue"] = -9999.0
    with pytest.raises(ValueError, match="No finite perpendicular baseline"):
        prep_nisar._read_perpendicular_baseline(gunw)


def test_read_target_grid(tmp_path):
    gunw = _build_gunw(tmp_path / "gunw_tg.h5", epsg=32612)
    epsg, xsub, ysub = prep_nisar._read_target_grid(gunw, (0, 0, 5, 4), "HH", "frequencyA")
    assert epsg == 32612
    assert len(xsub) == 5
    assert len(ysub) == 4
    assert xsub[0] == pytest.approx(400000.0)


def test_read_unwrapped_phase_valid_mask(tmp_path):
    gunw = _build_gunw(tmp_path / "gunw_vm.h5")
    with h5py.File(gunw, "a") as f:
        dset = f[f"{_unw_root('frequencyA')}/HH/unwrappedPhase"]
        dset[0, 0] = -9999.0
        dset.attrs["_FillValue"] = np.float32(-9999.0)
    valid = prep_nisar._read_unwrapped_phase_valid_mask(
        gunw, (0, 0, 5, 4), "HH", "frequencyA"
    )
    assert not valid[0, 0]
    assert valid[1, 1]


def test_read_is_land_and_valid_mask_bits(tmp_path):
    # value 11 -> ref/sec subswath 1, not water -> land & valid everywhere,
    # then poke the pathological encodings.
    gunw = _build_gunw(tmp_path / "gunw_mask.h5", include_mask=True)
    with h5py.File(gunw, "a") as f:
        m = f[f"{_unw_root('frequencyA')}/mask"]
        m[0, 0] = 111  # (//100 == 1) -> water -> excluded
        m[0, 1] = 0    # ref/sec subswath 0 -> invalid
        m[0, 2] = 10   # secondary subswath 0 -> invalid
    keep = prep_nisar._read_is_land_and_valid_mask(
        gunw, (0, 0, 5, 4), "HH", "frequencyA"
    )
    assert not keep[0, 0]
    assert not keep[0, 1]
    assert not keep[0, 2]
    assert keep[1, 1]


def test_read_is_land_and_valid_mask_fallback(tmp_path):
    # no mask dataset -> fall back to finite-unwrappedPhase validity
    gunw = _build_gunw(tmp_path / "gunw_nomask.h5", include_mask=False)
    keep = prep_nisar._read_is_land_and_valid_mask(
        gunw, (0, 0, 5, 4), "HH", "frequencyA"
    )
    assert keep.all()  # zeros are finite -> all valid


def test_read_common_is_land_and_valid_mask(tmp_path):
    f1 = _build_gunw(tmp_path / "gunw_c1.h5", include_mask=True)
    f2 = _build_gunw(tmp_path / "gunw_c2.h5", include_mask=True)
    with h5py.File(f2, "a") as f:
        f[f"{_unw_root('frequencyA')}/mask"][0, 0] = 111  # water in f2 only
    common = prep_nisar._read_common_is_land_and_valid_mask(
        [f1, f2], None, "HH", "frequencyA"
    )
    assert not common[0, 0]  # excluded because f2 flags it water
    assert common[1, 1]      # both land & valid


# ---------------------------------------------------------------------------
# Tier E -- pure numpy / scipy / string logic (no HDF5, no GDAL)
# ---------------------------------------------------------------------------
def test_grid_bounds_from_xy():
    x = np.array([400000.0, 400080.0, 400160.0, 400240.0, 400320.0])
    y = np.array([4300000.0, 4299920.0, 4299840.0, 4299760.0])
    bounds, dx, dy = prep_nisar._grid_bounds_from_xy(x, y)
    assert dx == pytest.approx(80.0)
    assert dy == pytest.approx(-80.0)
    # pixel edges extend half a step beyond the outer centers
    assert bounds == pytest.approx((399960.0, 4299720.0, 400360.0, 4300040.0))


def test_grid_bounds_from_xy_too_short_raises():
    with pytest.raises(ValueError, match="at least 2"):
        prep_nisar._grid_bounds_from_xy(np.array([1.0]), np.array([1.0]))


def test_make_rgi_descending_axis():
    # descending axis must be flipped so lookups stay correct
    rgi = prep_nisar._make_rgi(([2.0, 1.0, 0.0],), np.array([0.0, 10.0, 20.0]))
    assert rgi(np.array([[1.0]]))[0] == pytest.approx(10.0)
    # out-of-bounds -> NaN (bounds_error=False, fill_value=nan)
    assert np.isnan(rgi(np.array([[5.0]]))[0])


def test_prepare_valid_interp_points():
    xcoord = np.array([100.0, 200.0])
    ycoord = np.array([10.0, 20.0])
    y_2d, x_2d = np.meshgrid(ycoord, xcoord, indexing="ij")
    dem = np.zeros((2, 2))
    mask = np.array([[True, False], [False, True]])
    shape, ii, jj, pts = prep_nisar._prepare_valid_interp_points(x_2d, y_2d, dem, mask)
    assert shape == (2, 2)
    assert ii.size == 2
    assert pts.shape == (2, 3)  # columns: dem, y, x
    # empty mask -> no points
    _, ii0, _, pts0 = prep_nisar._prepare_valid_interp_points(
        x_2d, y_2d, dem, np.zeros((2, 2), bool)
    )
    assert ii0.size == 0
    assert pts0 is None


def _radar_grid_cube(**fields):
    """Minimal ascending (height, y, x) radar-grid dict with constant fields."""
    ctx = {
        "height_radar_grid": np.array([0.0, 1000.0]),
        "ycoord_radar_grid": np.array([0.0, 100.0]),
        "xcoord_radar_grid": np.array([0.0, 300.0]),
    }
    for name, value in fields.items():
        ctx[name] = np.full((2, 2, 2), value)
    return ctx


def test_interpolate_radar_grid_field_constant():
    rdr = _radar_grid_cube(foo=9.0)
    pts = np.array([[0.0, 50.0, 150.0]])
    out = prep_nisar._interpolate_radar_grid_field(rdr, "foo", pts)
    assert out[0] == pytest.approx(9.0)


def test_interpolate_geometry_constant_fields():
    xcoord = np.array([100.0, 200.0])
    ycoord = np.array([10.0, 20.0])
    y_2d, x_2d = np.meshgrid(ycoord, xcoord, indexing="ij")
    dem = np.zeros((2, 2))
    valid_mask = np.array([[True, False], [True, True]])
    rdr = _radar_grid_cube(
        slant_range=7.0e5, incidence_angle=30.0, los_x=0.6, los_y=0.8
    )
    sl, inc, az = prep_nisar.interpolate_geometry(x_2d, y_2d, dem, rdr, valid_mask)
    assert sl[0, 0] == pytest.approx(7.0e5)
    assert np.isnan(sl[0, 1])  # masked-out pixel stays NaN
    assert inc[1, 1] == pytest.approx(30.0)
    # azimuth angle from the horizontal LOS unit vector
    assert az[0, 0] == pytest.approx(np.degrees(np.arctan2(-0.8, -0.6)), abs=1e-3)


def test_interpolate_geometry_empty_mask():
    xcoord = np.array([100.0, 200.0])
    ycoord = np.array([10.0, 20.0])
    y_2d, x_2d = np.meshgrid(ycoord, xcoord, indexing="ij")
    dem = np.zeros((2, 2))
    rdr = _radar_grid_cube(
        slant_range=7.0e5, incidence_angle=30.0, los_x=0.6, los_y=0.8
    )
    sl, inc, az = prep_nisar.interpolate_geometry(
        x_2d, y_2d, dem, rdr, np.zeros((2, 2), bool)
    )
    assert np.isnan(sl).all()
    assert np.isnan(inc).all()
    assert np.isnan(az).all()


def test_interpolate_troposphere_sum():
    xcoord = np.array([100.0, 200.0])
    ycoord = np.array([10.0, 20.0])
    y_2d, x_2d = np.meshgrid(ycoord, xcoord, indexing="ij")
    dem = np.zeros((2, 2))
    rdr = _radar_grid_cube(hydrostatic_tropo=2.0, wet_tropo=3.0)
    out = prep_nisar.interpolate_troposphere(
        x_2d, y_2d, dem, rdr, np.ones((2, 2), bool)
    )
    assert np.allclose(out, 5.0)  # total = hydrostatic + wet


def test_interpolate_set_constant():
    xcoord = np.array([100.0, 200.0])
    ycoord = np.array([10.0, 20.0])
    y_2d, x_2d = np.meshgrid(ycoord, xcoord, indexing="ij")
    dem = np.zeros((2, 2))
    rdr = _radar_grid_cube(rdr_SET=4.0)
    out = prep_nisar.interpolate_set(x_2d, y_2d, dem, rdr, np.ones((2, 2), bool))
    assert np.allclose(out, 4.0)


def test_empty_interp_array():
    a = prep_nisar._empty_interp_array((2, 3))
    assert a.shape == (2, 3)
    assert a.dtype == np.float32
    assert np.isnan(a).all()


def test_apply_common_mask_to_observation():
    obs = {
        "unwrap_phase": np.ones((2, 2), np.float32),
        "connect_component": np.ones((2, 2), np.int32),
        "pbase": np.float32(5.0),  # scalar, must pass through untouched
    }
    mask = np.array([[True, False], [True, True]])
    out = prep_nisar._apply_common_mask_to_observation(obs, mask)
    assert np.isnan(out["unwrap_phase"][0, 1])   # float invalid -> NaN
    assert out["connect_component"][0, 1] == 0   # int invalid -> 0
    assert out["pbase"] == np.float32(5.0)
    # None mask is a no-op passthrough
    assert prep_nisar._apply_common_mask_to_observation(obs, None) is obs


def test_apply_common_mask_shape_mismatch_raises():
    obs = {"unwrap_phase": np.ones((3, 3), np.float32)}
    mask = np.array([[True, False], [True, True]])
    with pytest.raises(ValueError, match="common mask"):
        prep_nisar._apply_common_mask_to_observation(obs, mask)


def test_resolve_stack_type_explicit_and_legacy():
    assert prep_nisar._resolve_stack_type("ifgram", "whatever.h5") == "ifgram"
    assert prep_nisar._resolve_stack_type(None, "inputs/ionStack.h5") == "ion"
    assert prep_nisar._resolve_stack_type(None, "out/inputs/setStack.h5") == "set"


def test_resolve_stack_type_invalid_raises():
    with pytest.raises(ValueError, match="Unsupported stack_type"):
        prep_nisar._resolve_stack_type("bogus", "x.h5")


def test_resolve_stack_type_uninferable_raises():
    with pytest.raises(ValueError, match="Unable to infer"):
        prep_nisar._resolve_stack_type(None, "random_output.h5")


def test_get_date_pairs_from_identification(tmp_path):
    gunw = _build_gunw(tmp_path / "gunw_dp.h5")  # ref 2008-10-12 / sec 2008-11-27
    assert prep_nisar._get_date_pairs([gunw]) == ["20081012_20081127"]


def test_get_date_pairs_filename_fallback(tmp_path):
    name = (
        "NISAR_L2_PR_GUNW_001_030_A_019_002_2000_SH_20081012T060911_"
        "20081012T060925_20081127T061000_20081127T061014_D00404_N_F_J_001.h5"
    )
    gunw = _build_gunw(tmp_path / name)
    with h5py.File(gunw, "a") as f:
        del f[f"{IDENT}/secondaryZeroDopplerStartTime"]
    assert prep_nisar._get_date_pairs([gunw]) == ["20081012_20081127"]


def test_get_date_pairs_unresolvable_raises(tmp_path):
    gunw = _build_gunw(tmp_path / "short.h5")
    with h5py.File(gunw, "a") as f:
        del f[f"{IDENT}/secondaryZeroDopplerStartTime"]
    with pytest.raises(ValueError, match="Could not determine"):
        prep_nisar._get_date_pairs([gunw])


def test_bbox_to_utm_same_epsg_passthrough():
    # same src/dst EPSG returns sorted (xmin, ymin, xmax, ymax) unchanged
    out = prep_nisar.bbox_to_utm((10.0, 20.0, 5.0, 25.0), 4326, 4326)
    assert out == (5.0, 20.0, 10.0, 25.0)


def test_bbox_to_utm_transform_orders_bounds():
    out = prep_nisar.bbox_to_utm((-117.5, 33.9, -117.4, 34.0), 32611, 4326)
    assert len(out) == 4
    assert all(np.isfinite(v) for v in out)
    assert out[0] < out[2]
    assert out[1] < out[3]


def test_coerce_subset_metadata_types():
    meta = {
        "LENGTH": "4",
        "WIDTH": "5",
        "SUBSET_XMIN": "1",
        "X_FIRST": "399960.0",
        "Y_STEP": "-80.0",
    }
    out = prep_nisar._coerce_subset_metadata_types(meta)
    assert out["LENGTH"] == 4 and isinstance(out["LENGTH"], int)
    assert out["SUBSET_XMIN"] == 1 and isinstance(out["SUBSET_XMIN"], int)
    assert out["X_FIRST"] == 399960.0 and isinstance(out["X_FIRST"], float)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("auto", False),
        ("None", False),
        (None, False),
        ("no", False),
        ("", False),
        ("/data/mask.tif", True),
    ],
)
def test_external_mask_is_set(value, expected):
    assert prep_nisar._external_mask_is_set(value) is expected
