"""Runtime-only MJCF variant for matched payload diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET


@dataclass
class ModeledPayloadMjcf:
    """Own the temporary MJCF files used by every dynamics backend."""

    scene_path: Path
    body_name: str
    added_mass_kg: float
    nominal_mass_kg: float
    modeled_mass_kg: float
    _temporary_directory: tempfile.TemporaryDirectory

    def cleanup(self) -> None:
        self._temporary_directory.cleanup()


def create_modeled_payload_mjcf(
    scene_path: str | Path,
    *,
    body_name: str,
    added_mass_kg: float,
) -> ModeledPayloadMjcf:
    """Create a temporary scene whose target body includes the payload.

    The current G1 scene contains one robot ``include``.  The payload body has
    explicit geom masses, so scaling those masses preserves the existing mass
    distribution and matches the former plant-only ``body_mass/inertia``
    scaling.  The returned scene must be kept alive until all native model
    handles have closed.
    """

    source_scene = Path(scene_path).expanduser().resolve()
    added_mass = float(added_mass_kg)
    if not source_scene.is_file():
        raise FileNotFoundError(source_scene)
    if not 0.0 < added_mass <= 0.25:
        raise ValueError("modeled payload mass must be in (0, 0.25] kg")

    scene_tree = ET.parse(source_scene)
    scene_root = scene_tree.getroot()
    includes = scene_root.findall("include")
    if len(includes) != 1 or not includes[0].get("file"):
        raise ValueError("modeled payload requires one scene MJCF include")
    source_robot = (
        source_scene.parent / str(includes[0].get("file"))
    ).resolve()
    if not source_robot.is_file():
        raise FileNotFoundError(source_robot)

    robot_tree = ET.parse(source_robot)
    robot_root = robot_tree.getroot()
    bodies = robot_root.findall(f".//body[@name='{body_name}']")
    if len(bodies) != 1:
        raise ValueError(
            f"expected one payload body {body_name!r}, found {len(bodies)}"
        )
    payload_body = bodies[0]
    if payload_body.find("inertial") is not None:
        raise ValueError(
            "modeled payload helper expects geom-defined body inertia"
        )
    mass_geoms = [
        geom
        for geom in payload_body.findall("geom")
        if geom.get("mass") is not None
    ]
    if not mass_geoms:
        raise ValueError("payload body has no explicit geom masses")
    masses = [float(geom.get("mass")) for geom in mass_geoms]
    nominal_mass = sum(masses)
    if nominal_mass <= 0.0:
        raise ValueError("payload body nominal mass must be positive")
    mass_scale = (nominal_mass + added_mass) / nominal_mass
    for geom, mass in zip(mass_geoms, masses):
        geom.set("mass", format(mass * mass_scale, ".17g"))

    compiler = robot_root.find("compiler")
    if compiler is not None and compiler.get("meshdir"):
        mesh_directory = Path(str(compiler.get("meshdir")))
        if not mesh_directory.is_absolute():
            mesh_directory = (source_robot.parent / mesh_directory).resolve()
        compiler.set("meshdir", str(mesh_directory))

    temporary_directory = tempfile.TemporaryDirectory(
        prefix="disturbance-lab-modeled-payload-"
    )
    temporary_root = Path(temporary_directory.name)
    runtime_robot = temporary_root / "robot.xml"
    runtime_scene = temporary_root / "scene.xml"
    includes[0].set("file", runtime_robot.name)
    robot_tree.write(runtime_robot, encoding="utf-8", xml_declaration=True)
    scene_tree.write(runtime_scene, encoding="utf-8", xml_declaration=True)
    return ModeledPayloadMjcf(
        scene_path=runtime_scene,
        body_name=body_name,
        added_mass_kg=added_mass,
        nominal_mass_kg=nominal_mass,
        modeled_mass_kg=nominal_mass + added_mass,
        _temporary_directory=temporary_directory,
    )
