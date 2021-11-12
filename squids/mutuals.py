import argparse
import logging

import matplotlib.pyplot as plt
import superscreen as sc

from . import huber
from . import ibm


def get_mutual(
    make_squid,
    min_triangles,
    iterations,
    optimesh_steps=None,
    fc_lambda=None
):
    squid = make_squid()
    squid.make_mesh(min_triangles=min_triangles, optimesh_steps=optimesh_steps)
    if fc_lambda is not None:
        squid.layers["BE"].london_lambda = fc_lambda
    print(squid)
    fluxoid_polys = sc.make_fluxoid_polygons(squid)
    fig, ax = squid.plot()
    for name, poly in fluxoid_polys.items():
        ax.plot(*sc.geometry.close_curve(poly).T, label=name + "_fluxoid")
    ax.legend(bbox_to_anchor=(1, 1))
    ax.set_title(make_squid.__module__)
    return squid.mutual_inductance_matrix(iterations=iterations, units="Phi_0 / A")


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-triangles",
        type=int,
        default=10_000,
        help="Minimum number of triangles in the mesh.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of solver iterations.",
    )
    parser.add_argument(
        "--optimesh-steps",
        type=int,
        default=None,
        help="Number of optimesh steps to perform."
    )
    parser.add_argument(
        "--fc-lambda",
        type=float,
        default=None,
        help="London penetration depth for the field coil layer."
    )
    args = parser.parse_args()

    squid_funcs = [
        ibm.small.make_squid,
        ibm.medium.make_squid,
        ibm.large.make_squid,
        huber.make_squid,
    ]

    mutuals = {}
    for make_squid in squid_funcs:
        M = get_mutual(
            make_squid,
            args.min_triangles,
            args.iterations,
            optimesh_steps=args.optimesh_steps,
            fc_lambda=args.fc_lambda,
        )
        mutuals[make_squid.__module__] = M
        print(M)

    for label, mutual in mutuals.items():
        print()
        print(label)
        print("-" * len(label))
        print(mutual)
        print(mutual.to("pH"))
        print("-" * len(repr(mutual)))

    plt.show()
