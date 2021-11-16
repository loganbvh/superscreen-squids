from typing import List

from superscreen import Layer


def ibm_squid_layers(
    align: str = "bottom",
    london_lambda: float = 0.08,
    z0: float = 0.0,
    insulator_thickness_multiplier: float = 1.0
) -> List[Layer]:
    """Return a list of superscreen.Layers representing the superconducting layers
    in IBM SQUID susceptometers.

    See https://arxiv.org/pdf/1605.09483.pdf, Figure 8.

    Args:
        align: Whether to position the 2D model layer at the top, middle, or bottom
            of the phyical 3D metal layer.
        london_lambda: The London penetration depth for the superconducting films,
            in microns.
        z0: The vertical position of the bottom of W2, i.e. the surface of the
            SQUID chip.
        insulator_thickess_multiplier: Scale the insulator layer thicknesses by
            this amount (might be necessary for large SQUIDs to converge).

    Returns:
        A list a Layer objects representing the SQUID wiring layers.

    """
    assert align in ["top", "middle", "bottom"]

    # Layer thicknesses in microns.
    d_W2 = 0.20
    d_I2 = 0.13 * insulator_thickness_multiplier
    d_W1 = 0.10
    d_I1 = 0.15 * insulator_thickness_multiplier
    d_BE = 0.16

    # Metal layer vertical positions in microns.
    if align == "bottom":
        z0_W2 = z0
        z0_W1 = z0 + d_W2 + d_I2
        z0_BE = z0 + d_W2 + d_I2 + d_W1 + d_I1
    elif align == "middle":
        z0_W2 = z0 + d_W2 / 2
        z0_W1 = z0 + d_W2 / 2 + d_I2 + d_W1 / 2
        z0_BE = z0 + d_W2 / 2 + d_I2 + d_W1 / 2 + d_I1 + d_BE / 2
    else:
        z0_W2 = z0 + d_W2
        z0_W1 = z0 + d_W2 + d_I2 + d_W1
        z0_BE = z0 + d_W2 + d_I2 + d_W1 + d_I1 + d_BE

    return [
        Layer("W2", london_lambda=london_lambda, thickness=d_W2, z0=z0_W2),
        Layer("W1", london_lambda=london_lambda, thickness=d_W1, z0=z0_W1),
        Layer("BE", london_lambda=london_lambda, thickness=d_BE, z0=z0_BE),
    ]
