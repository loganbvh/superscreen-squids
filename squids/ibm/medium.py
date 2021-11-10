"""IBM SQUID susceptometer, 300 nm inner radius pickup loop."""

import numpy as np

import superscreen as sc

from .layers import ibm_squid_layers

def squid_geometry(interp_points=101):

    ri_pl = 0.3
    ro_pl = 0.5
    ri_fc = 1.0
    ro_fc = 1.5

    w_pl_outer = 0.45
    w_pl_center = 0.15
    pl_angle = 0
    pl_total_length = 3.5
    y0_pl_leads = -(pl_total_length - ro_pl)

    x0_pl_center = w_pl_center / 2
    theta0_pl_center = np.arcsin(x0_pl_center / ri_pl)
    thetas_pl_center = (
        np.linspace(theta0_pl_center, 2 * np.pi - theta0_pl_center, 101) - np.pi / 2
    )

    x0_pl_outer = w_pl_outer / 2
    theta0_pl_outer = np.arcsin(x0_pl_outer / ro_pl)
    thetas_pl_outer = (
        np.linspace(theta0_pl_outer, 2 * np.pi - theta0_pl_outer, 101) - np.pi / 2
    )

    pl_points = np.concatenate(
        [
            [[-w_pl_outer / 2, y0_pl_leads]],
            ro_pl
            * np.stack([np.cos(thetas_pl_outer), np.sin(thetas_pl_outer)], axis=1)[
                ::-1
            ],
            [[w_pl_outer / 2, y0_pl_leads]],
        ]
    )
    pl_points = sc.geometry.rotate(pl_points, pl_angle)

    pl_center = np.concatenate(
        [
            [[w_pl_center / 2, y0_pl_leads + (ro_pl - ri_pl)]],
            ri_pl
            * np.stack([np.cos(thetas_pl_center), np.sin(thetas_pl_center)], axis=1),
            [[-w_pl_center / 2, y0_pl_leads + (ro_pl - ri_pl)]],
        ]
    )
    pl_center = sc.geometry.rotate(pl_center, pl_angle)

    pl_shield = np.array(
        [
            [-1.30, -3.50],
            [-0.95, -2.80],
            [-0.29, -0.50],
            [0.30, -0.50],
            [1.00, -2.83],
            [1.30, -3.50],
        ]
    )
    pl_shield = pl_shield + np.array([[0, 0.1]])

    pl_shield2 = np.array([[-0.5, -1.75], [0.5, -1.75], [1.0, -3.5], [-1.0, -3.5]])

    w_fc_center = 0.55
    w_fc_outer = 1.30
    fc_angle = 65

    fc_center_length = 1.2
    fc_outer_length = 1.2
    y0_fc_center_leads = -(fc_center_length + ri_fc)
    y0_fc_outer_leads = -(fc_outer_length + ro_fc)

    x0_fc_center = w_fc_center / 2
    theta0_fc_center = np.arcsin(x0_fc_center / ri_fc)
    y0_fc_center = ri_fc * np.cos(theta0_fc_center)
    thetas_fc_center = (
        np.linspace(theta0_fc_center, 2 * np.pi - theta0_fc_center, 101) - np.pi / 2
    )

    fc_center_points = np.concatenate(
        [
            [[-w_fc_center / 2, y0_fc_center_leads]],
            ri_fc
            * np.stack([np.cos(thetas_fc_center), np.sin(thetas_fc_center)], axis=1)[
                ::-1
            ],
            [[w_fc_center / 2, y0_fc_center_leads]],
            [[-w_fc_center / 2, y0_fc_center_leads]],
        ]
    )
    fc_center_points = sc.geometry.rotate(fc_center_points, fc_angle)

    x0_fc_outer = w_fc_outer / 2
    theta0_fc_outer = np.arcsin(x0_fc_outer / ro_fc)
    y0_fc_outer = ri_fc * np.cos(theta0_fc_outer)
    thetas_fc_outer = (
        np.linspace(theta0_fc_outer, 2 * np.pi - theta0_fc_outer, 151) - np.pi / 2
    )

    fc_outer_points = np.concatenate(
        [
            [[-w_fc_outer / 2, y0_fc_outer_leads]],
            ro_fc
            * np.stack([np.cos(thetas_fc_outer), np.sin(thetas_fc_outer)], axis=1)[
                ::-1
            ],
            [[w_fc_outer / 2, y0_fc_outer_leads]],
            [[-w_fc_outer / 2, y0_fc_outer_leads]],
        ]
    )
    fc_outer_points = sc.geometry.rotate(fc_outer_points, fc_angle)

    fc_shield = np.array(
        [
            [1.70, -3.50],
            [0.92, -1.54],
            [1.14, -0.3],
            [1.7, 0.2],
            [2.11, 0],
            [2.75, -0.3],
            [3.25, -1.0],
            [3.25, -3.5],
        ]
    )

    polygons = {
        "pl": pl_points,
        "pl_center": pl_center,
        "pl_shield": pl_shield,
        "pl_shield2": pl_shield2,
        "fc": fc_outer_points,
        "fc_center": fc_center_points,
        "fc_shield": fc_shield,
    }

    if interp_points is not None:
        from scipy.interpolate import splprep, splev

        new_polygons = {}
        for name, points in polygons.items():
            x, y = np.array(points).T
            tck, u = splprep([x, y], s=0, k=1)
            new_points = splev(np.linspace(0, 1, interp_points), tck)
            new_polygons[name] = np.stack(new_points, axis=1)
        polygons = new_polygons

    return polygons


def make_squid(interp_points=121):
    length_units = "um"

    polygons = squid_geometry(interp_points=interp_points)

    films = [
        sc.Polygon("fc", layer="BE", points=polygons["fc"]),
        sc.Polygon("fc_shield", layer="W1", points=polygons["fc_shield"]),
        sc.Polygon("pl", layer="W1", points=polygons["pl"]),
        sc.Polygon("pl_shield", layer="W2", points=polygons["pl_shield"]),
    ]
    holes = [
        sc.Polygon("fc_center", layer="BE", points=polygons["fc_center"]),
        sc.Polygon("pl_center", layer="W1", points=polygons["pl_center"]),
    ]
    abstract_regions = [
        sc.Polygon(
            "bounding_box",
            layer="W1",
            points=sc.geometry.rectangle(6, 6, center=(0.75, -1)),
        ),
    ]

    device = sc.Device(
        "ibm_300nm",
        layers=ibm_squid_layers(),
        films=films,
        holes=holes,
        abstract_regions=abstract_regions,
        length_units=length_units,
    )
    return device
