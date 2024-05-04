from typing import Sequence

import numpy as np
import superscreen as sc
from superscreen.geometry import box
from tqdm import tqdm

LAYER_THICKNESS = dict(
    I0=0.30,
    BE=0.16,
    I1=0.15,
    W1=0.10,
    I2=0.13,
    W2=0.20,
)

SQUID_PARAMS = {
    "small": dict(
        ri_pl=0.1,
        ro_pl=0.3,
        ri_fc=0.5,
        ro_fc=1.0,
        d1=0.1,
        d2=3.1,
        d3=113.0,
    ),
    "medium": dict(
        ri_pl=0.3,
        ro_pl=0.5,
        ri_fc=1.0,
        ro_fc=1.5,
        d1=0.5,
        d2=3.6,
        d3=112.1,
    ),
    "large": dict(
        ri_pl=1.0,
        ro_pl=1.5,
        ri_fc=2.5,
        ro_fc=3.5,
        d1=1.5,
        d2=5.6,
        d3=109.1,
    ),
    "xlarge": dict(
        ri_pl=3.0,
        ro_pl=3.5,
        ri_fc=6.0,
        ro_fc=8.0,
        d1=3.5,
        d2=11.0,
        d3=101.7,
    ),
}


def make_squid_sample_side_view(
    *,
    ri_pl: float,
    ro_pl: float,
    ri_fc: float,
    ro_fc: float,
    d1: float,
    d2: float,
    d3: float,
    squid_height: float = 0.0,
    squid_angle: float = 0.0,
    deep_etch_thickness=10.840,
    name: str = "squid_side_view",
) -> sc.Device:

    squid_layers = []
    squid_polygons = []

    W2 = sc.Layer("Pickup loop shield", Lambda=0, z0=0)
    pl_shield_length = 40
    dx = -(d1 + pl_shield_length / 2)
    dy = -LAYER_THICKNESS["W2"] / 2
    pl_shield = sc.Polygon(
        "pl_shield",
        layer=W2.name,
        points=box(pl_shield_length, LAYER_THICKNESS["W2"]),
    ).translate(dx=dx, dy=dy)
    squid_layers.append(W2)
    squid_polygons.append(pl_shield)

    W1 = sc.Layer("Pickup loop", Lambda=0, z0=0)
    dx = (ri_pl + ro_pl) / 2
    dy = -(LAYER_THICKNESS["W2"] + LAYER_THICKNESS["I2"] + LAYER_THICKNESS["W1"] / 2)
    pl = sc.Polygon(
        "pl1",
        layer=W1.name,
        points=box((ro_pl - ri_pl), LAYER_THICKNESS["W1"]),
    ).translate(dx=dx, dy=dy)
    squid_polygons.append(pl)
    squid_layers.append(W1)

    BE = sc.Layer("Field coil", Lambda=0, z0=0)
    dx = (ri_fc + ro_fc) / 2
    dy = -(
        LAYER_THICKNESS["W2"]
        + LAYER_THICKNESS["I2"]
        + LAYER_THICKNESS["W1"]
        + LAYER_THICKNESS["I1"]
        + LAYER_THICKNESS["BE"] / 2
    )
    fc1 = sc.Polygon(
        "fc1",
        layer=BE.name,
        points=box((ro_fc - ri_fc), LAYER_THICKNESS["BE"]),
    ).translate(dx=dx, dy=dy)
    fc2 = sc.Polygon(
        "fc2",
        layer=BE.name,
        points=box((ro_fc - ri_fc), LAYER_THICKNESS["BE"]),
    ).translate(dx=-dx, dy=dy)
    squid_polygons.extend([fc1, fc2])
    squid_layers.append(BE)

    INS = sc.Layer("SQUID dielectric", Lambda=0, z0=0)
    ins_length = 98
    ins_thick = sum(LAYER_THICKNESS.values()) - LAYER_THICKNESS["W2"]
    dx = d2 - ins_length / 2
    dy = -(LAYER_THICKNESS["W2"] + ins_thick / 2)
    insulator = sc.Polygon(
        "SQUID dielectric",
        layer=INS.name,
        points=box(ins_length, ins_thick),
    ).translate(dx=dx, dy=dy)
    squid_layers.append(INS)
    squid_polygons.append(insulator)

    SUB = sc.Layer("SQUID substrate", Lambda=0, z0=0)
    deep_etch_length = 98
    dx = d2 - deep_etch_length / 2
    dy = -(sum(LAYER_THICKNESS.values()) + deep_etch_thickness / 2)
    deep_etch = sc.Polygon(
        points=box(deep_etch_length, deep_etch_thickness),
    ).translate(dx=dx, dy=dy)

    substrate_length = 200
    substrate_thick = 100
    dx = (d3 + d2) - substrate_length / 2
    dy = -(sum(LAYER_THICKNESS.values()) + deep_etch_thickness + substrate_thick / 2)
    substrate = sc.Polygon(
        "SQUID substrate",
        layer=SUB.name,
        points=box(substrate_length, substrate_thick),
    ).translate(dx=dx, dy=dy)
    substrate = substrate.union(deep_etch)
    squid_layers.append(SUB)
    squid_polygons.append(substrate)

    sub_length = 300
    sub_thick = 10
    dx = 0
    dy = sub_thick / 2
    sub_layer = sc.Layer("Sample substrate", Lambda=0, z0=0)
    sample = sc.Polygon(
        "sample",
        layer=sub_layer.name,
        points=box(sub_length, sub_thick),
    ).translate(dx=dx, dy=dy)

    squid_polygons = [
        p.rotate(squid_angle, origin=(0, 0)).translate(dy=-squid_height)
        for p in squid_polygons
    ]

    side_view = sc.Device(
        name,
        layers=squid_layers[::-1] + [sub_layer],
        films=squid_polygons + [sample],
        length_units="um",
    )

    return side_view


def find_minimum_standoff(
    squid_size: str,
    angles: Sequence[float] = np.linspace(0, 6, 61),
    standoffs: Sequence[float] = np.linspace(0, 2, 401),
) -> np.ndarray:
    kwargs = SQUID_PARAMS[squid_size].copy()
    min_standoffs = []

    for angle in tqdm(angles):
        for z0 in sorted(standoffs):
            kwargs["squid_angle"] = angle
            kwargs["squid_height"] = z0
            device = make_squid_sample_side_view(**kwargs)
            films = device.films
            sample = films["sample"].polygon
            distances = [
                sample.distance(films["SQUID substrate"].polygon),
                sample.distance(films["SQUID dielectric"].polygon),
                sample.distance(films["pl_shield"].polygon),
            ]
            if min(distances) > 0:
                min_standoffs.append(z0)
                break

    min_standoffs = np.array(min_standoffs)
    if len(min_standoffs) != len(angles):
        raise ValueError("Could not find minimum standoff distance for all angles.")
    return min_standoffs


def main():
    import argparse

    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--squid-size",
        type=str,
        choices=list(SQUID_PARAMS),
        help="SQUID size for which to find the minimum standoff distance vs. alignment angle",
    )
    parser.add_argument(
        "--angles",
        type=float,
        nargs=3,
        default=(0, 6, 61),
        help="Start, stop, number for SQUID alignment angle",
    )
    parser.add_argument(
        "--standoffs",
        type=float,
        nargs=3,
        default=(0, 2, 401),
        help="Start, stop, number for SQUID standoff distance",
    )

    args = parser.parse_args()

    squid_size = args.squid_size
    start, stop, num = args.angles
    angles = np.linspace(start, stop, int(num))

    start, stop, num = args.standoffs
    standoffs = np.linspace(start, stop, int(num))

    min_standoffs = find_minimum_standoff(
        squid_size, angles=angles, standoffs=standoffs
    )

    fig, ax = plt.subplots()
    ax.grid(True)
    ax.plot(angles, min_standoffs, ".-")
    ax.set_xlabel("Alignment (pitch) angle [deg.]")
    ax.set_ylabel("Minimum standoff distance [$\\mu$m]")
    ax.set_title(f"ibm.{squid_size}\nDistance to top of W2 layer")
    plt.show()


if __name__ == "__main__":
    main()
