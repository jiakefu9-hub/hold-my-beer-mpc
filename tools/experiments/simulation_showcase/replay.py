"""Open a selected, recorded simulation. Rendering only; never sends commands.

The colored rods illustrate the ACTUAL recorded grasp-site +Z axes. They are
not extra physical payloads. The gray dashed lines are fixed world vertical.
All cases use the same rod length, camera and playback clock; angles are not
amplified. Space pauses, R restarts, G toggles the explanatory rods.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / 'docs/simulation/experiments/20261103_showcase'


def load_case(name):
    manifest = json.loads((EVIDENCE / 'manifest.json').read_text())
    case = manifest['cases'][name]
    path = EVIDENCE / case['replay']
    if hashlib.sha256(path.read_bytes()).hexdigest() != case['replay_sha256']:
        raise ValueError('Replay hash does not match the selected experiment')
    with np.load(path, allow_pickle=False) as archive:
        frames = {k: archive[k] for k in ('time', 'qpos', 'qvel')}
    model = mujoco.MjModel.from_xml_path(str(ROOT / case['xml_path']))
    n = len(frames['time'])
    assert frames['qpos'].shape == (n, model.nq)
    assert frames['qvel'].shape == (n, model.nv)
    assert np.all(np.diff(frames['time']) > 0)
    assert all(np.isfinite(v).all() for v in frames.values())
    assert frames['time'][0] <= .0021 and frames['time'][-1] >= 8.-1e-9
    for filename, expected in manifest['model_sha256'].items():
        if hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Replay model changed: {filename}')
    return case, frames, model


def camera_at(camera, data, reset=False):
    if reset:
        mujoco.mjv_defaultCamera(camera)
        camera.distance = 1.65
        camera.azimuth = 135.
        camera.elevation = -10.
    # Follow translation only. Rotation, zoom and vertical pan remain adjustable.
    camera.lookat[:2] = data.qpos[:2]
    if reset:
        camera.lookat[2] = 1.02


def set_frame(model, data, frames, t):
    i = int(np.clip(np.searchsorted(frames['time'], t, side='right')-1,
                    0, len(frames['time'])-1))
    data.qpos[:] = frames['qpos'][i]
    data.qvel[:] = frames['qvel'][i]
    data.time = float(frames['time'][i])
    mujoco.mj_forward(model, data)


def grasp_tilt_deg(model, data, side):
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                             side+'_grasp_site')
    axis = data.site_xmat[site].reshape(3, 3)[:, 2]
    return np.degrees(np.arccos(np.clip(axis[2], -1, 1)))


def annotate(scene, model, data, enabled=True):
    def line(a, b, width, color):
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                           np.zeros(3), np.zeros(3), np.eye(3).ravel(), color)
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width, a, b)
        scene.ngeom += 1

    up = np.array([0., 0., 1.])
    gray = np.array([.6, .6, .6, 1.])
    for side, color in (
        ('left', np.array([1., .5, .06, 1.])),
        ('right', np.array([.05, .8, 1., 1.])),
    ):
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, side+'_grasp_site')
        point = data.site_xpos[site].copy()
        axis = data.site_xmat[site].reshape(3, 3)[:, 2]
        if enabled:
            for start in np.arange(0., .4, .04):
                line(point+start*up, point+(start+.022)*up, .0018, gray)
            line(point, point+.32*axis, .004, color)


def draw_text(context, model, data, controller_label, width, height):
    """Draw stable screen-space labels with MuJoCo's original smooth font."""
    left = grasp_tilt_deg(model, data, 'left')
    right = grasp_tilt_deg(model, data, 'right')
    if controller_label == 'PID':
        right_name = 'R PID'
    elif controller_label == 'NO-PREVIEW MPC':
        right_name = 'R No-Preview MPC'
    else:
        # Keep the on-screen text below MuJoCo's per-frame text-buffer limit;
        # the manifest and documentation retain the full controller name.
        right_name = 'R Dist.-Preview MPC'
    font = mujoco.mjtFont.mjFONT_NORMAL
    mujoco.mjr_text(font, f'{right_name}: {right:.2f}', context,
                    .30, .75, .05, .8, 1.)
    mujoco.mjr_text(font, f'L PD: {left:.2f}', context,
                    .69, .75, 1., .5, .06)
    # mjr_label changes the active viewport, so draw this after mjr_text.
    time_rect = mujoco.MjrRect(int(.42*width), int(.875*height),
                               int(.16*width), int(.085*height))
    mujoco.mjr_label(time_rect, font, f't = {data.time:.2f} s',
                     0., 0., 0., 0., 1., 1., 1., context)


def interactive_replay(model, data, frames, case, speed, once, guides):
    """Minimal MuJoCo viewer using the original font at 200% scale."""
    import glfw

    if not glfw.init():
        raise mujoco.FatalError('could not initialize GLFW')
    window = glfw.create_window(1280, 720, 'G1 arm-stability replay', None, None)
    if window is None:
        glfw.terminate()
        raise mujoco.FatalError('could not create GLFW window')
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    camera = mujoco.MjvCamera()
    option = mujoco.MjvOption()
    perturb = mujoco.MjvPerturb()
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_200)
    camera_at(camera, data, reset=True)
    state = dict(paused=False, restart=False, guides=guides,
                 last_x=0., last_y=0.)

    def key_callback(_window, key, _scancode, action, _mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_SPACE:
            state['paused'] = not state['paused']
        elif key == glfw.KEY_R:
            state['restart'] = True
        elif key == glfw.KEY_G:
            state['guides'] = not state['guides']
        elif key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)

    def cursor_callback(_window, xpos, ypos):
        dx, dy = xpos-state['last_x'], ypos-state['last_y']
        state['last_x'], state['last_y'] = xpos, ypos
        left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        middle = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        if not (left or middle or right):
            return
        _, height = glfw.get_window_size(window)
        shift = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
                 glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
        if right:
            action = (mujoco.mjtMouse.mjMOUSE_MOVE_H if shift
                      else mujoco.mjtMouse.mjMOUSE_MOVE_V)
        elif left:
            action = (mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift
                      else mujoco.mjtMouse.mjMOUSE_ROTATE_V)
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        mujoco.mjv_moveCamera(model, action, dx/max(height, 1),
                              dy/max(height, 1), scene, camera)

    def scroll_callback(_window, _xoffset, yoffset):
        mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                              0., -.05*yoffset, scene, camera)

    glfw.set_key_callback(window, key_callback)
    glfw.set_cursor_pos_callback(window, cursor_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    t = 0.
    previous = time.monotonic()
    controller_label = case.get('controller_label', 'PID').upper()
    if 'PID' in controller_label:
        controller_label = 'PID'
    try:
        while not glfw.window_should_close(window):
            started = time.monotonic()
            elapsed, previous = started-previous, started
            if state['restart']:
                t, state['restart'] = 0., False
            elif not state['paused']:
                t += elapsed*speed
            if t > 8.:
                if once:
                    break
                t %= 8.
            set_frame(model, data, frames, t)
            camera_at(camera, data)
            mujoco.mjv_updateScene(model, data, option, perturb, camera,
                                   mujoco.mjtCatBit.mjCAT_ALL, scene)
            annotate(scene, model, data, enabled=state['guides'])
            width, height = glfw.get_framebuffer_size(window)
            mujoco.mjr_render(mujoco.MjrRect(0, 0, width, height), scene, context)
            draw_text(context, model, data, controller_label, width, height)
            glfw.swap_buffers(window)
            glfw.poll_events()
            time.sleep(max(0., 1/60-(time.monotonic()-started)))
    finally:
        context.free()
        glfw.destroy_window(window)
        glfw.terminate()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    available_cases = tuple(json.loads(
        (EVIDENCE / 'manifest.json').read_text())['cases'])
    parser.add_argument('case', choices=available_cases)
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--once', action='store_true', help='Exit after one full 8 s task')
    parser.add_argument('--no-guides', action='store_true')
    parser.add_argument('--check', action='store_true', help='Validate files without a GUI')
    parser.add_argument('--render-check', type=Path, help='Save an offscreen frame, then exit')
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error('--speed must be finite and positive')
    case, frames, model = load_case(args.case)
    data = mujoco.MjData(model)
    timing = '6 ms position PID, no preview' if args.case.startswith('D01') else '6 ms MPC, 9 intervals'
    print(f"{args.case}: {case['title']} | {timing} | {args.speed:g}x")
    print('Robot LEFT stays fixed PD. Gray: world up; orange/cyan: actual LEFT/RIGHT axes.')
    print('Recorded 0-8 s, including stop at 6.4 s. No physical angle amplification.')
    if args.check:
        for t in (0., 3.2, 6.4, 8.):
            set_frame(model, data, frames, t)
            assert np.isfinite(data.site_xmat).all()
        print('PASS: hash, model, finite full trajectory and forward kinematics')
        return
    if args.render_check:
        from PIL import Image
        set_frame(model, data, frames, 3.2)
        camera = mujoco.MjvCamera()
        camera_at(camera, data, reset=True)
        model.vis.global_.offwidth = max(1280, model.vis.global_.offwidth)
        model.vis.global_.offheight = max(720, model.vis.global_.offheight)
        with mujoco.Renderer(
                model, height=720, width=1280,
                font_scale=mujoco.mjtFontScale.mjFONTSCALE_200) as renderer:
            renderer.update_scene(data, camera=camera)
            controller_label = case.get('controller_label', 'PID').upper()
            if 'PID' in controller_label:
                controller_label = 'PID'
            annotate(renderer.scene, model, data, enabled=not args.no_guides)
            args.render_check.parent.mkdir(parents=True, exist_ok=True)
            renderer._gl_context.make_current()
            mujoco.mjr_render(renderer._rect, renderer.scene,
                              renderer._mjr_context)
            draw_text(renderer._mjr_context, model, data, controller_label,
                      1280, 720)
            mujoco.mjr_finish()
            pixels = np.empty((720, 1280, 3), dtype=np.uint8)
            mujoco.mjr_readPixels(pixels, None, renderer._rect,
                                  renderer._mjr_context)
            Image.fromarray(np.flipud(pixels)).save(args.render_check)
        print(f'Saved actual replay frame: {args.render_check}')
        return

    print('Looping replay. SPACE: pause/resume; R: restart; G: guides; close window: exit.')
    print('Mouse: left rotate, right pan, middle/scroll zoom. Tracking follows translation.')
    set_frame(model, data, frames, 0.)
    try:
        interactive_replay(model, data, frames, case, args.speed, args.once,
                           not args.no_guides)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
