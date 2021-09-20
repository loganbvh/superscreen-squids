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


def split_layer(device, layer_name, max_thickness=0.05):
    """Splits a given layer into multiple thinner layers."""
    layers = device.layers
    films = device.films
    holes = device.holes
    abstract_regions = device.abstract_regions

    layer_to_split = layers.pop(layer_name)
    london_lambda = layer_to_split.london_lambda
    d = layer_to_split.thickness

    num_layers, remainder = divmod(d, max_thickness)
    num_layers = int(num_layers)
    new_ds = [max_thickness for _ in range(num_layers)]
    if abs(remainder) / d > 1e-6:
        num_layers += 1
        new_ds.append(remainder)
    new_layers = {}
    for i, new_d in enumerate(new_ds):
        name = f"{layer_name}_{i}"
        z = i * max_thickness + new_d / 2
        new_layers[name] = sc.Layer(
            name, london_lambda=london_lambda, thickness=new_d, z0=z
        )

    new_films = {}
    for name, film in films.items():
        if film.layer == layer_name:
            for i, new_layer_name in enumerate(new_layers):
                film_name = f"{name}_{i}"
                new_film = film.copy()
                new_film.name = film_name
                new_film.layer = new_layer_name
                new_films[film_name] = new_film
        else:
            new_films[name] = film

    new_holes = {}
    for name, hole in holes.items():
        if hole.layer == layer_name:
            for i, new_layer_name in enumerate(new_layers):
                hole_name = f"{name}_{i}"
                new_hole = hole.copy()
                new_hole.name = hole_name
                new_hole.layer = new_layer_name
                new_holes[film_name] = new_hole
        else:
            new_holes[name] = hole

    new_abstract_regions = {}
    for name, region in abstract_regions.items():
        if region.layer == layer_name:
            for i, new_layer_name in enumerate(new_layers):
                region_name = f"{name}_{i}"
                new_region = region.copy()
                new_region.name = region_name
                new_region.layer = new_layer_name
                new_abstract_regions[region_name] = new_region
        else:
            new_abstract_regions[name] = region

    new_layers.update(layers)

    return sc.Device(
        device.name,
        layers=new_layers,
        films=new_films,
        holes=new_holes,
        abstract_regions=new_abstract_regions,
        length_units=device.length_units,
    )


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
    squid = flip_device(squid, about_axis="x")

    films = [
        sc.Polygon(
            "xwide1",
            layer="W1",
            points=sc.geometry.rotate(
                sc.geometry.rectangle(15, 41, center=(6, -1)),
                19,
            )
        ),
        sc.Polygon(
            "xwide2",
            layer="W1",
            points=np.array(
                [
                    [-1, -12],
                    [-15, -5],
                    [-19, -3],
                    [-19, 20],
                    [-12.5, 20],
                    [-11, 16],
                    [-1, -12],
                ]
            ),
        ),
        sc.Polygon(
            "wide1",
            layer="BE",
            points=sc.geometry.rotate(
                sc.geometry.rectangle(3, 41, center=(9.5, -1)),
                19,
            ) 
        ),
        sc.Polygon(
            "wide2",
            layer="BE",
            points=sc.geometry.rotate(
                sc.geometry.rectangle(3, 41, center=(2.5, -1)),
                19,
            ) 
        ),
        sc.Polygon(
            "narrow1",
            layer="W2",
            points=sc.geometry.rotate(
                sc.geometry.rectangle(47, 1.0, center=(-1, 2.75)),
                -27,
            ) 
        ),
        sc.Polygon(
            "narrow2",
            layer="W2",
            points=sc.geometry.rotate(
                sc.geometry.rectangle(47, 1.0, center=(2.5, -4.25)),
                -27,
            ) 
        )
    ]

    abstract_regions = [
        sc.Polygon("bounding_box", layer="BE", points=sc.geometry.square(45, 45, center=(0, 1))),
    ]

    sample  =sc.Device(
        name="sample",
        layers=squids.ibm.large.make_squid().layers_list,
        films=films,
        abstract_regions=abstract_regions,
        length_units="um",
    )

    for layer in sample.layers_list:
        layer.london_lambda = 0.08

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
