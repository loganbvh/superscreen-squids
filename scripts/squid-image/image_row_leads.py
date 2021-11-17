import os
import sys
import logging
from typing import List

import numpy as np

import superscreen as sc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
import squids
from squids.ibm.layers import ibm_squid_layers


def lambda_bcs(lambda0: float, T: float, Tc: float) -> float:
    t = T / Tc
    return lambda0 / np.sqrt(1 - t ** 4) 


def make_sample(
    film_points=101,
    align_layers="bottom",
    insulator_thickness_multiplier=1.0,
):
    fc_angle = 45

    fc_shield = sc.Polygon(
        name="fc_shield",
        layer="W1",
        points=sc.geometry.box(16, 80, angle=fc_angle),
    )
    fc1 = sc.Polygon(
        "fc_lead1",
        layer="BE",
        points=sc.geometry.box(
            5.5,
            80,
            center=(-(5.5 / 2 + 0.5), 0),
            angle=fc_angle,
        ),
    )
    fc2 = sc.Polygon(
        "fc_lead2",
        layer="BE",
        points=sc.geometry.box(
            5.5,
            80,
            center=(+(5.5 / 2 + 0.5), 0),
            angle=fc_angle,
        ),
    )

    mod_shield1 = sc.Polygon(
        "mod_coil_shield1",
        layer="W1",
        points=sc.geometry.box(40, 21.213, center=(-20, 0)),
    ).difference(fc_shield.buffer(2))

    mod_shield2 = sc.Polygon(
        "mod_coil_shield2",
        layer="W1",
        points=sc.geometry.box(40, 21.213, center=(+20, 0)),
    ).difference(fc_shield.buffer(2))

    mod1 = sc.Polygon(
        "mod_coil_lead1",
        layer="W2",
        points=sc.geometry.box(80, 3, center=(0, -(3 / 2 + 2))),
    )
    mod2 = sc.Polygon(
        "mod_coil_lead2",
        layer="W2",
        points=sc.geometry.box(80, 3, center=(0, +(3 / 2 + 2))),
    )
    films = [
        fc_shield,
        fc1,
        fc2,
        mod_shield1,
        # mod_shield2,
        mod1,
        mod2,
    ]
    
    bounding_box = sc.Polygon(
        "bounding_box",
        layer="W1",
        points=sc.geometry.box(28, 32, center=(0, 0)),
    )

    for film in films:
        film.points = sc.geometry.translate(
            sc.geometry.rotate(film.points, -27),
            7.5, -4.5
        )
    
    films = [
        f.resample(film_points).intersection(bounding_box)
        for f in films
    ]

    sample = sc.Device(
        name="sample",
        layers=ibm_squid_layers(
            align=align_layers,
            insulator_thickness_multiplier=insulator_thickness_multiplier,
        ),
        films=films,
        abstract_regions=[bounding_box],
        length_units="um",
        solve_dtype="float32",
    )
    return sample


def mirror_layers(device, about=0, in_place=False):
    new_layers = []
    for layer in device.layers.values():
        new_layer = layer.copy()
        new_layer.z0 = about - layer.z0
        new_layers.append(new_layer)
    if not in_place:
        device = device.copy()
    device.layers_list = new_layers
    return device


def flip_device(device, about_axis="y"):
    device = device.copy(with_arrays=False)
    assert about_axis in "xy"
    index = 0 if about_axis == "y" else 1
    for polygon in device.polygons.values():
        polygon.points[:, index] *= -1
    return device


def update_origin(device, x0=0, y0=0):
    device = device.copy(with_arrays=True, copy_arrays=True)
    p0 = np.array([[x0, y0]])
    for polygon in device.polygons.values():
        polygon.points += p0
    if getattr(device, "points", None) is not None:
        device.points += p0
    return device


def sample_applied_field(x, y, z, fc_solution=None, field_units="mT"):
    x, y = np.atleast_1d(x, y)
    f = fc_solution.field_at_position(
        np.stack([x, y], axis=1),
        zs=z,
        units=field_units,
        with_units=False,
    )
    return f


def squid_applied_field(x, y, z, sample_solution=None, field_units="mT"):
    x, y = np.atleast_1d(x, y)
    f = sample_solution.field_at_position(
        np.stack([x, y], axis=1),
        zs=z,
        units=field_units,
        with_units=False,
    )
    return f


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        help="output directory",
    )
    parser.add_argument(
        "--sample-min-triangles",
        type=int,
        default=18_000,
        help="Minimum number of triangles to use in the sample mesh.",
    )
    parser.add_argument(
        "--squid-min-triangles",
        type=int,
        default=10_000,
        help="Minimum number of triangles to use in the SQUID mesh.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of solver iterations to perform.",
    )
    parser.add_argument(
        "--squid-height",
        type=float,
        default=0,
        help="Relative distance between the two SQUIDs",
    )
    parser.add_argument(
        "--x_range",
        type=str,
        help="start, stop for x axis in microns",
    )
    parser.add_argument(
        "--y_range",
        type=str,
        help="start, stop for y axis in microns",
    )
    parser.add_argument(
        "-sample-Tc",
        type=float,
        default=9.2,
        help="Sample critical temperature in Kelvin",
    )
    parser.add_argument(
        "--sample-lambda0",
        type=float,
        default=0.08,
        help="Sample T=0 London penetration depth in microns"
    )
    parser.add_argument(
        "--sample-temperature",
        type=float,
        default=9.0,
        help="Sample temperature in Kelvin"
    )
    args = parser.parse_args()

    field_units = "mT"
    outdir = args.outdir
    squid_height = args.squid_height
    x_range = args.x_range
    y_range = args.y_range
    iterations = args.iterations

    job_id = os.environ["SLURM_ARRAY_JOB_ID"]
    array_id = os.environ["SLURM_ARRAY_TASK_ID"]
    num_tasks = float(os.environ["SLURM_ARRAY_TASK_COUNT"])

    outfile = os.path.join(outdir, f"{job_id}_{array_id}_image_squid.npz")

    logging.basicConfig(level=logging.INFO)

    x_range = [
        float(x.strip()) for x in x_range.replace("(", "").replace(")", "").split(",")
    ]
    y_range = [
        float(y.strip()) for y in y_range.replace("(", "").replace(")", "").split(",")
    ]

    xstart, xstop = x_range
    ystart, ystop = y_range
    pixel_size = (ystop - ystart) / num_tasks

    xs = np.linspace(xstart, xstop, int(np.ceil((xstop - xstart) / pixel_size)))
    ys = np.linspace(ystart, ystop, int(np.ceil((ystop - ystart) / pixel_size)))

    squid = squids.ibm.medium.make_squid(align_layers="bottom")
    sample = make_sample()
    sample = flip_device(sample, about_axis="x")

    sample_lambda = lambda_bcs(
        args.sample_lambda0, args.sample_temperature, args.sample_Tc
    )
    for layer in sample.layers_list:
        layer.london_lambda = sample_lambda

    squid.make_mesh(min_triangles=args.squid_min_triangles, optimesh_steps=20)
    sample.make_mesh(min_triangles=args.sample_min_triangles, optimesh_steps=20)

    logging.info("Computing bare mutual inductance...")
    circulating_currents = {"fc_center": "1 mA"}
    I_fc = squid.ureg(circulating_currents["fc_center"])
    fc_solution = sc.solve(
        device=squid,
        circulating_currents=circulating_currents,
        iterations=iterations,
    )[-1]

    pl_fluxoid = sum(fc_solution.hole_fluxoid("pl_center", units="Phi_0"))
    m_no_sample = (pl_fluxoid / I_fc).to("Phi_0/A")
    logging.info(f"\tPhi = {pl_fluxoid:~.3fP}")
    logging.info(f"\tM = {m_no_sample:~.3fP}")

    sample_x0s = xs
    sample_y0 = ys[int(array_id)]

    polygon_flux = []
    flux_part = []
    supercurrent_part = []
    for i, x0 in enumerate(sample_x0s):

        logging.info(
            f"({i + 1} / {len(xs)}) Solving for sample response to field coil..."
        )
        _sample = mirror_layers(sample, about=squid_height)
        _sample = update_origin(_sample, x0=x0, y0=sample_y0)

        applied_field = sc.Parameter(
            sample_applied_field,
            fc_solution=fc_solution,
            field_units=field_units,
        )

        sample_solution = sc.solve(
            device=_sample,
            applied_field=applied_field,
            field_units=field_units,
            iterations=iterations,
            return_solutions=True,
            check_inversion=False,
        )[-1]

        logging.info("\tSolving for squid response to sample...")
        applied_field = sc.Parameter(
            squid_applied_field,
            sample_solution=sample_solution,
            field_units=field_units,
        )

        solution = sc.solve(
            device=squid,
            applied_field=applied_field,
            circulating_currents=None,
            field_units=field_units,
            iterations=iterations,
        )[-1]
        logging.info("\tComputing pickup loop flux...")
        fluxoid = solution.hole_fluxoid("pl_center", units="Phi_0")
        flux_part.append(fluxoid.flux_part.magnitude)
        supercurrent_part.append(fluxoid.supercurrent_part.magnitude)
        pl_flux = solution.polygon_flux(polygons="pl", units="Phi_0", with_units=False)["pl"]
        polygon_flux.append(pl_flux)
        logging.info(
            f"({i + 1} / {len(xs)}) mutual: "
            f"{(sum(fluxoid) / I_fc - m_no_sample).to('Phi_0 / A')}"
        )
        logging.info(f"({i + 1} / {len(xs)}) flux: {flux_part}")

    # Units: Phi_0
    flux = np.array(flux_part)
    supercurrent = np.array(supercurrent_part)
    # Units: Phi_0 / A
    mutual = (flux + supercurrent) / I_fc.to("A").magnitude
    data = dict(
        row=int(array_id),
        I_fc=I_fc.to("A").magnitude,
        current_units="A",
        pl_polygon_flux=np.array(polygon_flux),
        flux=flux,
        supercurrent=supercurrent,
        flux_units="Phi_0",
        mutual=mutual,
        mutual_no_sample=m_no_sample.to("Phi_0 / A").m,
        mutual_units="Phi_0/A",
        xs=xs,
        ys=ys,
        y=sample_y0,
        length_units="um",
    )
    np.savez(outfile, **data)
    logging.info(f"Data saved to {outfile}.")
    logging.info("Done.")
