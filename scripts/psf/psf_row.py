import os
import sys
import time
import logging

import numpy as np

import superscreen as sc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
import squids


squid_funcs = {
    "ibm-small": squids.ibm.small.make_squid,
    "ibm-medium": squids.ibm.medium.make_squid,
    "ibm-large": squids.ibm.large.make_squid,
    "ibm-xlarge": squids.ibm.xlarge.make_squid,
}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--squid-type",
        type=str,
        choices=("ibm-small", "ibm-medium", "ibm-large", "ibm-xlarge"),
        help="Type of SQUID to simulate.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="output directory",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=4000,
        help="Minimum number of vertices to use in the SQUID mesh.",
    )
    parser.add_argument(
        "--optimesh-steps",
        type=int,
        default=None,
        help="Number of optimesh steps to perform.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of solver iterations to perform.",
    )
    parser.add_argument(
        "--squid-height",
        type=float,
        default=0,
        help="SQUID height in microns",
    )
    parser.add_argument(
        "--align-layers",
        type=str,
        default="middle",
        choices=("bottom", "middle", "top"),
    )
    parser.add_argument(
        "--x-range",
        type=str,
        help="start, stop for x axis in microns",
    )
    parser.add_argument(
        "--y-range",
        type=str,
        help="start, stop for y axis in microns",
    )
    parser.add_argument(
        "--pixel-size", type=float, default=0.05, help="Linear size for square pixels."
    )
    args = parser.parse_args()

    field_units = "Phi_0 / um ** 2"
    squid_type = args.squid_type
    outdir = args.outdir
    squid_height = abs(args.squid_height)
    min_points = args.min_points
    x_range = args.x_range
    y_range = args.y_range
    iterations = args.iterations
    optimesh_steps = args.optimesh_steps
    pixel_size = args.pixel_size
    align_layers = args.align_layers

    x_range = [
        float(x.strip()) for x in x_range.replace("(", "").replace(")", "").split(",")
    ]
    y_range = [
        float(y.strip()) for y in y_range.replace("(", "").replace(")", "").split(",")
    ]

    xstart, xstop = x_range
    ystart, ystop = y_range

    if "SLURM_ARRAY_JOB_ID" in os.environ:
        job_id = os.environ["SLURM_ARRAY_JOB_ID"]
        array_id = os.environ["SLURM_ARRAY_TASK_ID"]
        num_rows = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
        pixel_size = abs(ystop - ystart) / num_rows
    else:
        job_id = time.strfime("%y%m%d_%H%M%S")
        array_id = 0
        num_rows = args.num_rows

    outfile = os.path.join(
        outdir,
        f"{job_id}_{array_id}_{squid_type}_z{1e3 * squid_height:.0f}nm.npz",
    )

    logging.basicConfig(level=logging.INFO)

    xs = np.linspace(xstart, xstop, int(np.ceil((xstop - xstart) / pixel_size)))
    ys = np.linspace(ystart, ystop, int(np.ceil((ystop - ystart) / pixel_size)))

    squid = squid_funcs[squid_type](align_layers=align_layers)
    logging.info(squid)

    squid.make_mesh(min_points=min_points, optimesh_steps=optimesh_steps)

    sample_x0s = xs
    sample_y0 = ys[int(array_id)]

    flux_part = []
    supercurrent_part = []
    for i, x0 in enumerate(sample_x0s):

        logging.info(f"({i + 1} / {len(xs)})")

        applied_field = sc.sources.MonopoleField(r0=(x0, sample_y0, -squid_height))

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
        logging.info(
            f"({i + 1} / {len(xs)}) total fluxoid: {flux_part[-1] + supercurrent_part[-1]}"
        )

    # Units: Phi_0
    flux = np.array(flux_part)
    supercurrent = np.array(supercurrent_part)
    data = dict(
        row=int(array_id),
        current_units="A",
        flux=flux,
        supercurrent=supercurrent,
        fluxoid=(flux + supercurrent),
        flux_units="Phi_0",
        xs=xs,
        ys=ys,
        y=sample_y0,
        length_units="um",
    )
    np.savez(outfile, **data)
    logging.info(f"Data saved to {outfile}.")
    logging.info("Done.")


if __name__ == "__main__":
    main()
