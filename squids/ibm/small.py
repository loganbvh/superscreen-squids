"""IBM SQUID susceptometer, 100 nm inner radius pickup loop."""

import os

import numpy as np
from numpy.lib.polynomial import poly
from scipy.io import loadmat

import superscreen as sc
from superscreen import Device, Layer, Polygon, geometry

from .layers import ibm_squid_layers


def make_layout_small_susc_jkr():
    mat_path = os.path.join(os.path.dirname(__file__), "small_susc.mat")
    layout = loadmat(mat_path)
    origin = layout["origin"]
    pl = layout["pl"]
    pl_centers = layout["pl_centers"]
    pl_shield = layout["pl_shield"]
    pl_shield_2 = layout["pl_shield_2"]
    # A = layout['A']
    fc_in = layout["fc_in"]
    fc_out = layout["fc_out"]
    fc_shield = layout["fc_shield"]
    two_micron_scale = layout["two_micron_scale"]

    z0 = 0.0  # microns
    london_lambda = 0.080  # 80 nm London penetration depth for Nb films

    scale_factor = 2 / (two_micron_scale[1, 0] - two_micron_scale[1, 1])

    components = {}

    fc_in[:, 0] = (fc_in[:, 0] - origin[0, 0]) * scale_factor
    fc_in[:, 1] = -(fc_in[:, 1] - origin[0, 1]) * scale_factor
    components["fc_in"] = fc_in
    fc_out[:, 0] = (fc_out[:, 0] - origin[0, 0]) * scale_factor
    fc_out[:, 1] = -(fc_out[:, 1] - origin[0, 1]) * scale_factor
    components["fc_out"] = fc_out
    fc_shield[:, 0] = (fc_shield[:, 0] - origin[0, 0]) * scale_factor
    fc_shield[:, 1] = -(fc_shield[:, 1] - origin[0, 1]) * scale_factor
    components["fc_shield"] = fc_shield
    pl[:, 0] = (pl[:, 0] - origin[0, 0]) * scale_factor
    pl[:, 1] = -(pl[:, 1] - origin[0, 1]) * scale_factor
    components["pl"] = pl
    pl_shield[:, 0] = (pl_shield[:, 0] - origin[0, 0]) * scale_factor
    pl_shield[:, 1] = -(pl_shield[:, 1] - origin[0, 1]) * scale_factor
    components["pl_shield"] = pl_shield
    pl_shield_2[:, 0] = (pl_shield_2[:, 0] - origin[0, 0]) * scale_factor
    pl_shield_2[:, 1] = -(pl_shield_2[:, 1] - origin[0, 1]) * scale_factor
    components["pl_shield2"] = pl_shield_2
    pl_centers[:, 0] = (pl_centers[:, 0] - origin[0, 0]) * scale_factor
    pl_centers[:, 1] = -(pl_centers[:, 1] - origin[0, 1]) * scale_factor


    polygons = {
        "fc": np.concatenate([fc_in[:-2, :], np.flipud(fc_out)]),
        "fc_shield": fc_shield,
        "pl_shield": pl_shield,
        "pl_shield2": pl_shield_2,
        "pl": pl,
    }
    return polygons

def make_squid():

    polygons = make_layout_small_susc_jkr()

    fc0 = geometry.rotate(polygons["fc"], -45)

    fc = np.concatenate(
        [
            fc0[29:-3],
            [[1.80, -0.19]],
            [[1.95, -0.5]],
            [[1.97, -0.8]],
            [[1.97, -1.2]],
            [[1.97, -1.5]],
            [[1.97, -2.0]],
            [[1.90, -2.1]],
            [[1.80, -2.10]],
            [[1.60, -2.10]],
            [[1.40, -2.00]],
            [[1.30, -1.95]],
            [[1.20, -1.80]],
            [[1.15, -1.50]],
            [[1.10, -1.40]],
        ]
    )

    fc_center = np.concatenate([[[1.45, -0.65]], fc0[1:28][::-1]])

    fc_center = np.concatenate(
        [
            fc_center[4:-1],
            [[1.20, -0.20]],
            [[1.40, -0.45]],
            [[1.50, -0.65]],
            [[1.55, -0.75]],
            [[1.60, -0.85]],
            [[1.62, -0.95]],
            [[1.64, -1.00]],
            [[1.66, -1.2]],
            [[1.63, -1.30]],
            [[1.50, -1.30]],
            [[1.40, -1.15]],
            [[1.30, -1.00]],
            [[1.20, -0.85]],
            [[1.10, -0.76]],
        ]
    )

    fc_shield = np.array(
        [
            [0.81450159, -1.62204163],
            [0.65, -1.1],
            [0.75880918, -0.04873086],
            [1.25, 0.45],
            [1.55, 0.25],
            [1.8, 0.05],
            [2.0, -0.13],
            [2.15, -0.31326984],
            [2.15, -3.0],
            [1.3, -3.0],
        ]
    )

    pl_shield = np.array(
        [
            [-0.31326984, -0.10442328],
            [0.30630829, -0.11138483],
            [0.71007831, -1.74038802],
            [1.1, -3],
            [-1.1, -3],
            [-0.70311676, -1.55242611],
        ]
    )

    pl_shield2 = np.array(
        [
            [-0.45250089, -1.44104128],
            [0.42465468, -1.44104128],
            [0.57084727, -1.87961906],
            [0.92, -2.9],
            [-0.92, -2.9],
            [-0.54996261, -1.71950336],
        ]
    )

    pl0 = geometry.rotate(polygons["pl"], -45)
    pl = (
        np.concatenate(
            [
                pl0[:8],
                np.array([[+0.70, -2.7]]),
                np.array([[+0.75, -2.8]]),
                np.array([[-0.75, -2.8]]),
                np.array([[-0.70, -2.7]]),
            ]
        )
        + np.array([0.02, 0])
    )
    y0 = pl[:, 1].max() - 0.15
    pl_center = sc.geometry.box(0.225, 2.6, center=(0, -2.6 / 2 + y0))

    bbox = np.array(
        [
            [-1.25, -3.15],
            [-1.25, 1.05],
            [2.20, 1.05],
            [2.20, -3.15],
        ]
    )

    polygons = {
        "fc": fc,
        "fc_center": fc_center,
        "pl_shield2": pl_shield2,
        "fc_shield": fc_shield,
        "pl": pl,
        "pl_center": pl_center,
        "pl_shield": pl_shield,
        "bounding_box": bbox,
    }

    films = [
        Polygon("fc", layer="BE", points=polygons["fc"]),
        Polygon("pl_shield2", layer="BE", points=polygons["pl_shield2"]),
        Polygon("fc_shield", layer="W1", points=polygons["fc_shield"]),
        Polygon("pl", layer="W1", points=polygons["pl"]),
        Polygon("pl_shield", layer="W2", points=polygons["pl_shield"]),
    ]

    holes = [
        Polygon("fc_center", layer="BE", points=polygons["fc_center"]),
        Polygon("pl_center", layer="W1", points=polygons["pl_center"]).resample(),
    ]

    abstract_regions = [
        Polygon("bounding_box", layer="W1", points=polygons["bounding_box"]),
    ]

    return Device(
        "ibm_100nm",
        layers=ibm_squid_layers(),
        films=films,
        holes=holes,
        abstract_regions=abstract_regions,
    )
