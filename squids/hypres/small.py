import os

import numpy as np
import superscreen as sc
from superscreen.geometry import box

from .layers import hypres_squid_layers


def make_polygons():
    coords = {}
    npz_path = os.path.join(os.path.dirname(__file__), "hypres-400nm.npz")
    with np.load(npz_path) as df:
        for name, array in df.items():
            coords[name] = array
    films = ["fc", "fc_shield", "pl", "pl_shield"]
    holes = ["pl_center", "fc_center"]
    abstract_regions = ["bounding_box"]
    sc_polygons = {name: sc.Polygon(name, points=coords[name]) for name in films}
    sc_holes = {name: sc.Polygon(name, points=coords[name]) for name in holes}
    sc_abstract = {
        name: sc.Polygon(name, points=coords[name]) for name in abstract_regions
    }
    return sc_polygons, sc_holes, sc_abstract


def make_squid(align_layers: str = "middle"):
    polygons, holes, abstract_regions = make_polygons()
    layers = hypres_squid_layers(align=align_layers)
    layer_mapping = {
        "fc": "BE",
        "fc_center": "BE",
        "fc_shield": "W1",
        "pl": "W1",
        "pl_center": "W1",
        "pl_shield": "W2",
    }
    fc_center = holes.pop("fc_center")
    fc_mask = sc.Polygon(points=box(5)).rotate(45).translate(dx=6.5, dy=-5.5)
    polygons["fc"] = polygons["fc"].difference(fc_mask, fc_center)
    for name, poly in polygons.items():
        poly.layer = layer_mapping[name]
        poly.points = poly.resample(151)
    for name, poly in holes.items():
        poly.layer = layer_mapping[name]
        poly.points = poly.resample(151)
    polygons["fc"].points = polygons["fc"].resample(501)
    source = (
        sc.Polygon("source", layer="BE", points=box(2, 0.1))
        .rotate(45)
        .translate(dx=5.5, dy=-2.95)
    )
    drain = (
        sc.Polygon("drain", layer="BE", points=box(2, 0.1))
        .rotate(45)
        .translate(dx=3.95, dy=-4.5)
    )
    return sc.Device(
        "hypres_400nm",
        layers=layers,
        films=list(polygons.values()),
        holes=list(holes.values()),
        terminals={"fc": [source, drain]},
        length_units="um",
    )
