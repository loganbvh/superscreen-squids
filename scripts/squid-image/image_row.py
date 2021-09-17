import os
import sys
import logging

import numpy as np

import superscreen as sc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
import squids


def mirror_layers(device, about=0, in_place=False):
    new_layers = []
    for layer in device.layers_list:
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
    polygons = device.films_list + device.holes_list + device.abstract_regions_list
    for polygon in polygons:
        polygon.points[:, index] *= -1
    return device


def update_origin(device, x0=0, y0=0):
    device = device.copy(with_arrays=True, copy_arrays=True)
    polygons = device.films_list + device.holes_list + device.abstract_regions_list
    p0 = np.array([[x0, y0]])
    for polygon in polygons:
        polygon.points += p0
    if hasattr(device, "points"):
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
        "--min-triangles",
        type=int,
        default=8000,
        help="Minimum number of triangles to use in the two SQUID meshes.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=2,
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
    args = parser.parse_args()

    field_units = "mT"
    outdir = args.outdir
    squid_height = args.squid_height
    min_triangles = args.min_triangles
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

    squid = squids.ibm.medium.make_squid()
    sample = squids.ibm.large.make_squid()
    sample.layers["BE"].london_lambda = 0.08
    sample = flip_device(sample, about_axis="y")

    squid.make_mesh(min_triangles=args.min_triangles, optimesh_steps=400)
    sample.make_mesh(min_triangles=args.min_triangles, optimesh_steps=400)

    logging.info("Computing bare mutual inductance...")
    circulating_currents = {"fc_center": "1 mA"}
    I_fc = squid.ureg(circulating_currents["fc_center"])
    fc_solution = sc.solve(
        device=squid,
        circulating_currents=circulating_currents,
        iterations=iterations,
        return_solutions=True,
    )[-1]

    flux = fc_solution.polygon_flux()
    m_no_sample = (flux["pl_hull"] / I_fc).to("Phi_0/A")
    logging.info(f"\tPhi = {flux['pl_hull'].to('Phi_0'):.3e~P}")
    logging.info(f"\tM = {m_no_sample:.3f~P}")

    sample_x0s = xs
    sample_y0 = ys[int(array_id)]

    flux = []
    for i, x0 in enumerate(sample_x0s):

        logging.info(
            f"({i + 1} / {len(xs)}) Solving for sample response to field coil..."
        )
        _sample = mirror_layers(sample, about=-abs(squid_height))
        _sample = update_origin(_sample, x0=-x0, y0=-sample_y0)

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
            field_units=field_units,
            return_solutions=True,
            iterations=iterations,
        )[-1]
        logging.info("\tComputing pickup loop flux...")
        flux.append(solution.polygon_flux(units="Phi_0", with_units=False)["pl_hull"])
        logging.info(f"({i + 1} / {len(xs)}) flux: {flux}")

    # Units: Phi_0
    flux = np.array(flux)
    # Units: Phi_0 / A
    susc = flux / I_fc.to("A").magnitude - m_no_sample.magnitude
    data = dict(
        row=int(array_id),
        flux=flux,
        flux_units="Phi_0",
        susc=susc,
        susc_units="Phi_0/A",
        xs=xs,
        ys=ys,
        y=sample_y0,
        length_units="um",
    )
    np.savez(outfile, **data)
    logging.info(f"Data saved to {outfile}.")
    logging.info("Done.")
