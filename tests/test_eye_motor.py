"""Geometry of the virtual eye and the motor readout, without the connectome."""

import numpy as np
import pandas as pd
import pytest
import torch

from flysurvivors.columns import EyeColumns
from flysurvivors.eye import EyeParams, VirtualEye
from flysurvivors.motor import Locomotion, LocomotionParams, MotorReadout


def synthetic_columns(step_deg=6.0):
    """A regular grid of columns on both eyes, no cells."""
    rows = []
    for eye, sign in (("R", 1), ("L", -1)):
        for az in np.arange(10, 170, step_deg):
            for el in np.arange(-70, 70, step_deg):
                rows.append((eye, len(rows) + 1, 0, 0, sign * az, el))
    columns = pd.DataFrame(rows, columns=["eye", "col", "p", "q", "az_deg", "el_deg"])
    cells = pd.DataFrame(columns=["idx", "cell_type", "eye", "col"])
    return EyeColumns(columns, cells)


def disc_crop(size, px_per_unit, xy, radius, ground=0.8, value=0.0):
    img = np.full((size, size), ground, dtype=np.float32)
    c = (size - 1) / 2
    yy, xx = np.mgrid[0:size, 0:size]
    img[(xx - (c + xy[0] * px_per_unit)) ** 2 + (yy - (c + xy[1] * px_per_unit)) ** 2 <= (radius * px_per_unit) ** 2] = value
    return img


def test_eye_sees_dark_disc_on_the_right():
    ec = synthetic_columns()
    eye = VirtualEye(ec, EyeParams(crop_px=96, px_per_unit=12.0))
    az = ec.columns["az_deg"].to_numpy()
    el = ec.columns["el_deg"].to_numpy()
    # Disc 2 units to the right (fly frame: +y is the right side).
    I = eye.encode(disc_crop(96, 12.0, (0.0, 2.0), 0.6), heading_deg=0.0)
    dark = (I < 0.4) & eye.sees_ground
    assert dark.any()
    assert np.all(az[dark] > 0), "only right-eye columns should see it"
    assert np.all(el[dark] < 0), "only below-horizon columns see the ground"
    assert abs(np.median(az[dark]) - 90) < 15
    # Uniform ground: all ground-seeing columns see the ground value.
    I0 = eye.encode(np.full((96, 96), 0.8, dtype=np.float32))
    assert np.allclose(I0[eye.sees_ground], 0.8, atol=1e-3)
    assert np.all(I0[~eye.sees_ground] == eye.p.sky)


def test_eye_heading_rotates_the_world():
    ec = synthetic_columns()
    eye = VirtualEye(ec, EyeParams(crop_px=96, px_per_unit=12.0))
    az = ec.columns["az_deg"].to_numpy()
    crop = disc_crop(96, 12.0, (2.0, 0.0), 0.6)  # disc towards screen +x
    g = eye.sees_ground
    I = eye.encode(crop, heading_deg=0.0)
    assert abs(np.median(az[(I < 0.4) & g])) < 15  # straight ahead
    I = eye.encode(crop, heading_deg=90.0)  # fly now faces screen +y: disc is on its left
    assert np.median(az[(I < 0.4) & g]) < -60


def test_closer_disc_covers_more_columns():
    ec = synthetic_columns()
    eye = VirtualEye(ec, EyeParams(crop_px=96, px_per_unit=12.0))
    g = eye.sees_ground
    far = ((eye.encode(disc_crop(96, 12.0, (0.0, 3.0), 0.5)) < 0.4) & g).sum()
    near = ((eye.encode(disc_crop(96, 12.0, (0.0, 1.2), 0.5)) < 0.4) & g).sum()
    assert near > 2 * far


def test_motor_readout_rates():
    groups = {"a": np.array([0, 1]), "b": np.array([2])}
    r = MotorReadout(groups, "cpu", tau_ms=1e-6)  # no smoothing
    counts = torch.tensor([2.0, 4.0, 1.0])
    rates = r.update(counts, window_ms=100.0)
    assert rates["a"] == pytest.approx(30.0)  # mean of 20 and 40 Hz
    assert rates["b"] == pytest.approx(10.0)


def test_locomotion_turns_right_when_right_dna02_leads():
    loco = Locomotion(LocomotionParams(turn_gain=10.0, forward_gain=0.1))
    h, s = loco.update({"DNa02_R": 5.0, "DNa02_L": 1.0, "DNp09": 3.0}, dt_ms=1000.0)
    assert h == pytest.approx(40.0)
    assert s == pytest.approx(0.3)
    x, y = loco.stick()
    assert x > 0 and y > 0  # clockwise from +x in screen coordinates


def test_locomotion_escape_burst():
    loco = Locomotion(LocomotionParams(escape_threshold=5.0, escape_ms=100.0))
    _, s = loco.update({"GF": 20.0, "DNp09": 50.0}, dt_ms=16.0)
    assert s == pytest.approx(-loco.params.escape_speed)
    for _ in range(10):
        _, s = loco.update({"GF": 0.0, "DNp09": 50.0}, dt_ms=16.0)
    assert s > 0
