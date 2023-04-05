import argparse

import matplotlib.pyplot as plt

from . import huber, hypres, ibm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--draw",
        action="store_true",
        help="Use Device.draw() instead of Device.plot()",
    )
    parser.add_argument(
        "--same-scale",
        action="store_true",
        help="Whether to plot all devices on the same scale.",
    )
    args = parser.parse_args()

    squid_funcs = [
        hypres.small.make_squid,
        ibm.small.make_squid,
        ibm.medium.make_squid,
        ibm.large.make_squid,
        ibm.xlarge.make_squid,
        huber.make_squid,
    ]

    plt.rcParams["savefig.dpi"] = 200

    fig, axes = plt.subplots(
        1,
        len(squid_funcs),
        figsize=(len(squid_funcs) * 3, 3),
        sharex=args.same_scale,
        sharey=args.same_scale,
        constrained_layout=True,
    )

    for ax, make_squid in zip(axes, squid_funcs):
        squid = make_squid()
        if args.draw:
            squid.draw(ax=ax, legend=False, exclude=["bounding_box", "circle"])
        else:
            squid.plot(ax=ax, legend=False)
        ax.set_title(make_squid.__module__)
    plt.show()
