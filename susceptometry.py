from typing import Optional

import numpy as np
import pint
import superscreen as sc
from scipy.spatial.transform import Rotation
from superscreen.geometry import box


def make_sample(
    width: float,
    height: float,
    Lambda: float,
    z0: float = 0.0,
    max_edge_length: float = 0.5,
    smooth: int = 0,
) -> sc.Device:
    """Makes a rectangular sample."""
    layer = sc.Layer("sample", Lambda=Lambda, z0=z0)
    film = sc.Polygon(
        "sample",
        layer="sample",
        points=box(width, height, points=int(1.25 * 2 * (width + height))),
    )
    sample = sc.Device(
        "lead",
        layers=[layer],
        films=[film],
        length_units="um",
    )
    sample.make_mesh(max_edge_length=max_edge_length, smooth=smooth, buffer=0)
    return sample


def get_mutual(
    squid: sc.Device, iterations: int = 5
) -> tuple[sc.Solution, pint.Quantity]:
    """Get the bare mutual inductance for the SQUID."""
    I_fc = "1 mA"
    solution = sc.solve(
        squid,
        terminal_currents={"fc": {"source": f"{I_fc}", "drain": f"-{I_fc}"}},
        iterations=iterations,
    )[-1]
    fluxoid = sum(solution.hole_fluxoid("pl_center"))
    mutual = (fluxoid / sc.ureg(I_fc)).to("Phi_0 / A")
    return solution, mutual


def field_from_solution(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    solution: sc.Solution,
    dr: tuple[float, float, float] = (0.0, 0.0, 0.0),
    units: str = "mT",
    pitch: float = 0.0,
    roll: float = 0.0,
    yaw: float = 0.0,
) -> np.ndarray:
    """Evaluates the z-component of the field from a ``superscreen.Solution``."""
    x = np.squeeze(x)
    y = np.squeeze(y)
    z = np.squeeze(z)
    if z.ndim == 0:
        z = z.item() * np.ones_like(x)
    positions = np.array([x, y, z]).T - np.array([dr])
    if all(angle == 0 for angle in (pitch, roll, yaw)):
        return solution.screening_field_at_position(
            positions, units=units, with_units=False
        )
    # Rotate eval coordinates relative to source coordinates
    rot = Rotation.from_euler("xyz", (pitch, roll, yaw), degrees=True)
    positions = rot.apply(positions)
    field = solution.screening_field_at_position(
        positions,
        vector=True,
        units=units,
        with_units=False,
    )
    # Rotate field back to source coordinates
    field = rot.apply(field, inverse=True)
    return field[:, 2]


def get_susceptibility(
    sample: sc.Device,
    fc_solution: sc.Solution,
    squid_position: tuple[float, float, float],
    squid_model: Optional[sc.FactorizedModel] = None,
    iterations: int = 5,
    pitch: float = 0.0,
    roll: float = 0.0,
    yaw: float = 0.0,
) -> float:
    I_fc = fc_solution.terminal_currents["fc"]["source"]
    current_units = fc_solution.current_units
    I_fc = f"{I_fc} {current_units}"
    squid_position = np.array(squid_position)
    # Calculate screening currents in the sample
    sample_solution = sc.solve(
        sample,
        applied_field=sc.Parameter(
            field_from_solution,
            solution=fc_solution,
            dr=squid_position,
            pitch=pitch,
            roll=roll,
            yaw=yaw,
        ),
        field_units="mT",
    )[-1]

    # Calculate flux through the SQUID
    kwargs = dict(
        applied_field=sc.Parameter(
            field_from_solution,
            solution=sample_solution,
            dr=-squid_position,
            pitch=-pitch,
            roll=-roll,
            yaw=-yaw,
        ),
        field_units="mT",
        iterations=iterations,
        progress_bar=False,
    )

    if squid_model is None:
        kwargs["device"] = fc_solution.device
    else:
        kwargs["model"] = squid_model
        kwargs["current_units"] = None

    squid_solution = sc.solve(**kwargs)[-1]

    # Calculate mutual inductance
    mutual = sum(squid_solution.hole_fluxoid("pl_center")) / sc.ureg(I_fc)
    return mutual.to("Phi_0 / A").magnitude
