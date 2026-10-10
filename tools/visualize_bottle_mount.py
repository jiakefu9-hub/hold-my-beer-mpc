#!/usr/bin/env python3
"""Interactively inspect the G1 bottle mount and its center of mass.

This utility only performs forward kinematics.  It never steps simulation,
loads a controller, connects to a robot, or publishes a command.
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = REPO_ROOT / "resources/g1_description/scene.xml"
AXIS_COLORS = (
    np.array([1.0, 0.12, 0.12, 1.0]),  # X: red
    np.array([0.12, 1.0, 0.12, 1.0]),  # Y: green
    np.array([0.15, 0.35, 1.0, 1.0]),  # Z: blue
)
COM_COLOR = np.array([1.0, 0.85, 0.05, 1.0])
ELBOW_AXIS_COLOR = np.array([0.05, 0.95, 0.95, 1.0])
HORIZONTAL_PROJECTION_COLOR = np.array([1.0, 1.0, 1.0, 0.82])
HORIZONTAL_DISTANCE_COLOR = np.array([1.0, 0.05, 0.85, 1.0])


def _object_id(model, object_type, name):
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"MuJoCo model does not contain {name!r}")
    return object_id


def selected_sides(side):
    return ("left", "right") if side == "both" else (side,)


def make_bottles_transparent(model, alpha):
    for side in ("left", "right"):
        for suffix in ("body", "neck"):
            geom_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_bottle_{suffix}"
            )
            if geom_id < 0:
                continue
            model.geom_rgba[geom_id, 3] = alpha


def bottle_dimensions(model, side):
    body_geom_id = _object_id(
        model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_bottle_body"
    )
    neck_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_bottle_neck"
    )
    geom_ids = (body_geom_id,) if neck_geom_id < 0 else (body_geom_id, neck_geom_id)
    axial_min = min(
        model.geom_pos[geom_id, 2] - model.geom_size[geom_id, 1]
        for geom_id in geom_ids
    )
    axial_max = max(
        model.geom_pos[geom_id, 2] + model.geom_size[geom_id, 1]
        for geom_id in geom_ids
    )
    body_diameter = 2.0 * model.geom_size[body_geom_id, 0]
    return axial_max - axial_min, body_diameter


def bottle_frame(model, data, side):
    body_id = _object_id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_bottle")
    # xipos is the world position of the body's combined inertial COM.  It is
    # slightly above the body origin here because the bottle cap has mass.
    com_world = data.xipos[body_id].copy()
    rotation_world = data.xmat[body_id].reshape(3, 3).copy()
    return com_world, rotation_world


def elbow_axis_geometry(model, data, side, com):
    """Return the forearm axis and the COM's horizontal projection onto it.

    The forearm axis is the infinite line through the elbow and wrist joint
    anchors.  For the requested horizontal offset, the line is first projected
    into the world XY plane and then placed at the bottle COM height.
    """
    elbow_id = _object_id(
        model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_elbow_joint"
    )
    wrist_id = _object_id(
        model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_wrist_roll_joint"
    )
    elbow = data.xanchor[elbow_id].copy()
    wrist = data.xanchor[wrist_id].copy()

    direction_3d = wrist - elbow
    direction_3d /= np.linalg.norm(direction_3d)

    direction_xy = direction_3d[:2].copy()
    horizontal_norm = np.linalg.norm(direction_xy)
    if horizontal_norm < 1e-9:
        raise ValueError(f"{side} elbow-wrist axis has no horizontal direction")
    direction_xy /= horizontal_norm

    along_xy = np.dot(com[:2] - elbow[:2], direction_xy)
    closest_xy = elbow[:2] + along_xy * direction_xy
    closest_horizontal = np.array([closest_xy[0], closest_xy[1], com[2]])
    horizontal_distance = np.linalg.norm(com[:2] - closest_xy)

    along_3d = np.dot(com - elbow, direction_3d)
    closest_3d = elbow + along_3d * direction_3d
    spatial_distance = np.linalg.norm(com - closest_3d)
    return (
        elbow,
        wrist,
        direction_3d,
        direction_xy,
        closest_horizontal,
        horizontal_distance,
        spatial_distance,
    )


def _append_sphere(scene, position, radius, color):
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("No free user geometry slots in MuJoCo scene")
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.full(3, radius),
        np.asarray(position, dtype=np.float64),
        np.eye(3).reshape(-1),
        np.asarray(color, dtype=np.float32),
    )
    scene.ngeom += 1


def _append_capsule(scene, start, end, radius, color):
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("No free user geometry slots in MuJoCo scene")
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).reshape(-1),
        np.asarray(color, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        np.asarray(start, dtype=np.float64),
        np.asarray(end, dtype=np.float64),
    )
    scene.ngeom += 1


def append_geometry_markers(scene, model, data, sides, axis_length):
    for side in sides:
        com, rotation = bottle_frame(model, data, side)
        _append_sphere(scene, com, 0.014, COM_COLOR)
        for axis_index, color in enumerate(AXIS_COLORS):
            end = com + axis_length * rotation[:, axis_index]
            _append_capsule(scene, com, end, 0.0045, color)
            _append_sphere(scene, end, 0.007, color)

        (
            elbow,
            wrist,
            direction_3d,
            direction_xy,
            closest_horizontal,
            _,
            _,
        ) = elbow_axis_geometry(model, data, side, com)

        # Cyan: the physical elbow-to-wrist axis, extended in both directions.
        line_start = elbow - 0.10 * direction_3d
        line_end = elbow + 0.55 * direction_3d
        _append_capsule(scene, line_start, line_end, 0.0035, ELBOW_AXIS_COLOR)
        _append_sphere(scene, elbow, 0.009, ELBOW_AXIS_COLOR)
        _append_sphere(scene, wrist, 0.009, ELBOW_AXIS_COLOR)

        # White: that same line projected into the horizontal plane through COM.
        projection_start_xy = elbow[:2] - 0.10 * direction_xy
        projection_end_xy = elbow[:2] + 0.55 * direction_xy
        projection_start = np.array(
            [projection_start_xy[0], projection_start_xy[1], com[2]]
        )
        projection_end = np.array(
            [projection_end_xy[0], projection_end_xy[1], com[2]]
        )
        _append_capsule(
            scene,
            projection_start,
            projection_end,
            0.0025,
            HORIZONTAL_PROJECTION_COLOR,
        )

        # Magenta: the shortest horizontal distance requested by the user.
        _append_capsule(
            scene, com, closest_horizontal, 0.006, HORIZONTAL_DISTANCE_COLOR
        )
        _append_sphere(
            scene, closest_horizontal, 0.010, HORIZONTAL_DISTANCE_COLOR
        )


def print_geometry(model, data, sides):
    print("Legend: COM=yellow, +X=red, +Y=green, +Z=blue")
    print("        elbow-wrist extension=cyan, horizontal projection=white")
    print("        COM horizontal distance=magenta")
    print("The bottle geometries are transparent; axes use the bottle body frame.")
    for side in sides:
        body_id = _object_id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_bottle")
        wrist_id = _object_id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_wrist_roll_joint"
        )
        wrist_body_id = _object_id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_roll_rubber_hand"
        )
        com, _ = bottle_frame(model, data, side)
        (
            elbow,
            wrist,
            _,
            _,
            closest_horizontal,
            horizontal_distance,
            spatial_distance,
        ) = elbow_axis_geometry(model, data, side, com)
        wrist = data.xanchor[wrist_id].copy()
        wrist_rotation = data.xmat[wrist_body_id].reshape(3, 3)
        relative = wrist_rotation.T @ (com - wrist)
        bottle_length, body_diameter = bottle_dimensions(model, side)
        print(f"\n{side.upper()} bottle")
        print(f"  mass                 = {model.body_mass[body_id] * 1000.0:.3f} g")
        print(f"  visible axial length = {bottle_length * 1000.0:.3f} mm")
        print(f"  main-body diameter   = {body_diameter * 1000.0:.3f} mm")
        print(f"  principal inertia    = {model.body_inertia[body_id]} kg m^2")
        print(f"  body inertial offset = {model.body_ipos[body_id] * 1000.0} mm")
        print(f"  COM world            = {com} m")
        print(f"  COM from wrist frame = {relative * 1000.0} mm")
        print(f"  wrist-to-COM distance= {np.linalg.norm(com - wrist) * 1000.0:.3f} mm")
        print(f"  elbow world          = {elbow} m")
        print(f"  wrist world          = {wrist} m")
        print(f"  horizontal foot      = {closest_horizontal} m")
        print(
            "  COM-to-elbow-axis horizontal distance"
            f" = {horizontal_distance * 1000.0:.3f} mm"
        )
        print(
            "  COM-to-elbow-axis 3D distance"
            f"         = {spatial_distance * 1000.0:.3f} mm"
        )


def set_camera(camera, focus, whole_body):
    if whole_body:
        camera.lookat[:] = np.array([0.08, 0.0, 0.90])
        camera.distance = 1.55
        camera.azimuth = 155.0
        camera.elevation = -10.0
    else:
        camera.lookat[:] = focus
        camera.distance = 0.72
        camera.azimuth = 145.0
        camera.elevation = -8.0


def render_snapshot(path, model, data, sides, axis_length, focus, whole_body):
    import imageio.v2 as imageio

    model.vis.global_.offwidth = 1000
    model.vis.global_.offheight = 800
    camera = mujoco.MjvCamera()
    set_camera(camera, focus, whole_body)
    with mujoco.Renderer(model, height=800, width=1000) as renderer:
        renderer.update_scene(data, camera=camera)
        append_geometry_markers(renderer.scene, model, data, sides, axis_length)
        imageio.imwrite(path, renderer.render())
    print(f"Saved preview: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect transparent G1 bottles and visualize their COM frames."
    )
    parser.add_argument(
        "--xml", type=Path, default=DEFAULT_XML, help="MuJoCo scene XML"
    )
    parser.add_argument(
        "--side", choices=("left", "right", "both"), default="right"
    )
    parser.add_argument("--alpha", type=float, default=0.16)
    parser.add_argument("--axis-length", type=float, default=0.12)
    parser.add_argument(
        "--whole-body", action="store_true", help="Start with a whole-body view"
    )
    parser.add_argument(
        "--snapshot", type=Path, default=None, help="Render one image instead of opening a viewer"
    )
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be between 0 and 1")
    if args.axis_length <= 0.0:
        parser.error("--axis-length must be positive")

    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    make_bottles_transparent(model, args.alpha)
    mujoco.mj_forward(model, data)

    sides = selected_sides(args.side)
    print_geometry(model, data, sides)
    focus = np.mean([bottle_frame(model, data, side)[0] for side in sides], axis=0)

    if args.snapshot is not None:
        render_snapshot(
            args.snapshot, model, data, sides, args.axis_length, focus, args.whole_body
        )
        return

    print("\nViewer controls: drag to rotate, right-drag to pan, scroll to zoom.")
    print("Close the window or press Ctrl-C in the terminal to exit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        set_camera(viewer.cam, focus, args.whole_body)
        while viewer.is_running():
            viewer.user_scn.ngeom = 0
            append_geometry_markers(
                viewer.user_scn, model, data, sides, args.axis_length
            )
            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()
