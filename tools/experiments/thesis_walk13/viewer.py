"""Open a selected, recorded simulation. Rendering only; never sends commands.

The colored rods illustrate the ACTUAL recorded grasp-site +Z axes. They are
not extra physical payloads. The gray dashed lines are fixed world vertical.
All cases use the same rod length, camera and playback clock; angles are not
amplified. Space pauses, R restarts, G toggles the explanatory rods.
"""
from pathlib import Path
import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import time
from zoneinfo import ZoneInfo

import mujoco
import numpy as np

from presentation import CameraPreset, ControlPanel, load_camera, save_camera

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / 'docs/simulation/experiments/20261117_walk10_stop3'
PRESENTATION = ROOT / 'evaluation/replay_presentation'


def replay_duration(case, frames):
    """Use the selected recording's duration for validation and playback."""
    duration = float(case.get('duration_s', frames['time'][-1]))
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError('Replay duration must be finite and positive')
    if frames['time'][-1] < duration-1e-9:
        raise ValueError('Saved trajectory ends before the declared replay duration')
    return duration


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
    assert frames['time'][0] <= .0021
    replay_duration(case, frames)
    for filename, expected in manifest['model_sha256'].items():
        if hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Replay model changed: {filename}')
    return case, frames, model


def camera_at(camera, data, reset=False, offset=None):
    if reset:
        mujoco.mjv_defaultCamera(camera)
        camera.distance = 1.65
        camera.azimuth = 135.
        camera.elevation = -10.
    # Follow translation only. Rotation, zoom and vertical pan remain adjustable.
    offset = (0., 0., 1.02) if offset is None else offset
    camera.lookat[:2] = data.qpos[:2]+np.asarray(offset[:2])
    camera.lookat[2] = offset[2]
    if reset:
        camera.lookat[2] = 1.02


def save_screenshot(path, pixels, case_name, case, task_time, preset):
    """Save the rendered framebuffer without window chrome or control panels."""
    from PIL import Image
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.flipud(pixels)).save(path)
    metadata = dict(case=case_name, controller=case['title'],
                    task_time_s=float(task_time), camera=asdict(preset),
                    image_size_px=[int(pixels.shape[1]), int(pixels.shape[0])],
                    replay_sha256=case['replay_sha256'],
                    replay=case['replay'],
                    created_at=datetime.now(ZoneInfo('Europe/Berlin')).isoformat())
    path.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return path


def set_frame(model, data, frames, t):
    i = int(np.clip(np.searchsorted(frames['time'], t+1e-10, side='right')-1,
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


def draw_text(context, model, data, controller_label):
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
        right_name = 'R Template MPC'
    font = mujoco.mjtFont.mjFONT_NORMAL
    mujoco.mjr_text(font, f'{right_name}: {right:.2f}', context,
                    .30, .82, .05, .8, 1.)
    mujoco.mjr_text(font, f'L PD: {left:.2f}', context,
                    .69, .82, 1., .5, .06)


def interactive_replay(model, data, frames, case, speed, once, guides, *,
                       case_name, preset, camera_path, screenshot_dir,
                       start_time=0., paused=False, controls=True):
    """Minimal MuJoCo viewer using the original font at 200% scale."""
    import glfw

    if not glfw.init():
        raise mujoco.FatalError('could not initialize GLFW')
    window = glfw.create_window(int(preset.width_px), int(preset.height_px),
                                'G1 arm-stability replay', None, None)
    if window is None:
        glfw.terminate()
        raise mujoco.FatalError('could not create GLFW window')
    glfw.set_window_size_limits(window, 240, 240, 7680, 7680)
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    camera = mujoco.MjvCamera()
    option = mujoco.MjvOption()
    perturb = mujoco.MjvPerturb()
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_200)
    camera_at(camera, data, reset=True)
    default_fovy = float(model.vis.global_.fovy)
    state = dict(paused=paused, restart=False, guides=guides,
                 last_x=0., last_y=0., seek=None, screenshot=False,
                 case_request=None, panel_request=False,
                 speed=speed, lock_aspect=preset.lock_aspect, clock_reset=False,
                 offset=preset.apply(camera, model, data))
    duration = replay_duration(case, frames)
    cache = {case_name: (case, frames)}
    panel = None

    def get_view():
        return CameraPreset.capture(camera, model, data, glfw.get_window_size(window),
                                    lock_aspect=state['lock_aspect'])

    def apply_window_size(view):
        # Remove the old constraint before applying a different window ratio.
        glfw.set_window_aspect_ratio(window, glfw.DONT_CARE, glfw.DONT_CARE)
        glfw.set_window_size(window, int(view.width_px), int(view.height_px))
        terms = view.aspect_terms()
        if terms is not None:
            glfw.set_window_aspect_ratio(window, *terms)
        state['lock_aspect'] = view.lock_aspect

    def apply_view(view):
        state['offset'] = view.apply(camera, model, data)
        apply_window_size(view)

    apply_window_size(preset)

    def save_view():
        save_camera(camera_path, get_view())
        message = f'Camera saved: {camera_path}'
        print(message, flush=True)
        return message

    def load_view():
        apply_view(load_camera(camera_path))
        return f'Camera loaded: {camera_path}'

    def seek(t):
        if not np.isfinite(t) or not 0 <= t <= duration:
            raise ValueError(f'Time must be between 0 and {duration:g} s')
        state['seek'], state['paused'] = t, True
        state['clock_reset'] = True

    def step(direction):
        # Several repeat events can arrive before the next frame is rendered.
        current_time = state['seek'] if state['seek'] is not None else data.time
        index = np.searchsorted(frames['time'], current_time)
        index = int(np.clip(index+direction, 0, len(frames['time'])-1))
        seek(float(frames['time'][index]))

    def screenshot():
        state['screenshot'], state['paused'] = True, True

    def select_case(name):
        if name not in ('D01', 'D02', 'D03'):
            raise ValueError('Unknown replay case')
        state['case_request'], state['paused'] = name, True

    def toggle_pause():
        state['paused'] = not state['paused']
        state['clock_reset'] = True

    def set_speed(value):
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError('Playback speed must be finite and greater than zero')
        state['speed'] = value
        state['clock_reset'] = True
        return f'Playback speed: {value:g}x'

    def reset_view():
        apply_view(CameraPreset(fovy=default_fovy))
        return 'Default camera restored.'

    def open_panel():
        try:
            return ControlPanel(get_view, dict(
                apply=apply_view, save=save_view, load=load_view,
                reset=reset_view, seek=seek, step=step, screenshot=screenshot,
                case=select_case, pause=toggle_pause, speed=set_speed),
                duration, case_name, screenshot_dir, get_speed=lambda: state['speed'])
        except Exception as exc:
            print(f'Camera panel unavailable: {exc}. C retries; CLI presets still work.',
                  flush=True)
            return None

    def key_callback(_window, key, _scancode, action, _mods):
        if key in (glfw.KEY_LEFT, glfw.KEY_RIGHT):
            if action in (glfw.PRESS, glfw.REPEAT):
                step(-1 if key == glfw.KEY_LEFT else 1)
            return
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_SPACE:
            toggle_pause()
        elif key == glfw.KEY_R:
            state['restart'] = True
        elif key == glfw.KEY_G:
            state['guides'] = not state['guides']
        elif key == glfw.KEY_S:
            screenshot()
        elif key in (glfw.KEY_1, glfw.KEY_2, glfw.KEY_3):
            select_case({glfw.KEY_1: 'D01', glfw.KEY_2: 'D02', glfw.KEY_3: 'D03'}[key])
        elif key == glfw.KEY_C:
            state['panel_request'] = True
        elif key in (glfw.KEY_F5, glfw.KEY_F9):
            try:
                message = save_view() if key == glfw.KEY_F5 else load_view()
                if panel is not None and not panel.closed:
                    panel.refresh()
                    panel.message.set(message)
            except (ValueError, OSError, TypeError) as exc:
                print(f'Camera: {exc}', flush=True)
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
        camera.distance = float(np.clip(camera.distance, .02, 100.))
        view = get_view()
        state['offset'] = (view.offset_x, view.offset_y, view.height)

    def scroll_callback(_window, _xoffset, yoffset):
        mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                              0., -.05*yoffset, scene, camera)
        camera.distance = float(np.clip(camera.distance, .02, 100.))

    glfw.set_key_callback(window, key_callback)
    glfw.set_cursor_pos_callback(window, cursor_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    t = start_time
    previous = time.monotonic()
    if controls:
        panel = open_panel()
    try:
        while not glfw.window_should_close(window):
            if panel is not None and not panel.closed:
                panel.pump(case_name, float(data.time), state['paused'])
            if state['panel_request']:
                if panel is None or panel.closed:
                    panel = open_panel()
                else:
                    panel.root.deiconify()
                    panel.root.lift()
                state['panel_request'] = False
            request = state['case_request']
            if request is not None:
                if request not in cache:
                    new_case, new_frames, new_model = load_case(request)
                    if new_case['xml_path'] != case['xml_path']:
                        raise ValueError('In-window switching requires the same robot model')
                    cache[request] = (new_case, new_frames)
                    del new_model
                case, frames = cache[request]
                case_name = request
                duration = replay_duration(case, frames)
                state['case_request'] = None
                t = min(float(data.time), duration)
                print(f'Switched to {case_name} at t={t:.3f} s; camera unchanged.', flush=True)
            started = time.monotonic()
            elapsed, previous = started-previous, started
            if state['clock_reset']:
                elapsed, state['clock_reset'] = 0., False
            if state['seek'] is not None:
                t, state['seek'] = state['seek'], None
            elif state['restart']:
                t, state['restart'] = 0., False
            elif not state['paused']:
                t += elapsed*state['speed']
            if t > duration:
                if once:
                    set_frame(model, data, frames, duration)
                    break
                t %= duration
            set_frame(model, data, frames, t)
            camera_at(camera, data, offset=state['offset'])
            glfw.make_context_current(window)
            mujoco.mjv_updateScene(model, data, option, perturb, camera,
                                   mujoco.mjtCatBit.mjCAT_ALL, scene)
            annotate(scene, model, data, enabled=state['guides'])
            width, height = glfw.get_framebuffer_size(window)
            if width == 0 or height == 0:
                glfw.wait_events_timeout(.05)
                continue
            mujoco.mjr_render(mujoco.MjrRect(0, 0, width, height), scene, context)
            controller_label = case.get('controller_label', 'PID').upper()
            if 'PID' in controller_label:
                controller_label = 'PID'
            draw_text(context, model, data, controller_label)
            if state['screenshot']:
                stamp = datetime.now(ZoneInfo('Europe/Berlin')).strftime('%Y%m%d_%H%M%S_%f')
                path = screenshot_dir/f'{case_name}_t{data.time:07.3f}_{stamp}.png'
                pixels = np.empty((height, width, 3), dtype=np.uint8)
                mujoco.mjr_finish()
                mujoco.mjr_readPixels(pixels, None, mujoco.MjrRect(0, 0, width, height), context)
                save_screenshot(path, pixels, case_name, case, data.time, get_view())
                state['screenshot'] = False
                print(f'Screenshot: {path}', flush=True)
                if panel is not None and not panel.closed:
                    panel.message.set(f'Saved {path.name}')
            glfw.swap_buffers(window)
            glfw.poll_events()
            time.sleep(max(0., 1/60-(time.monotonic()-started)))
    finally:
        if panel is not None and not panel.closed:
            panel.close()
        context.free()
        glfw.destroy_window(window)
        glfw.terminate()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    available_cases = tuple(json.loads(
        (EVIDENCE / 'manifest.json').read_text())['cases'])
    parser.add_argument('case', choices=available_cases)
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--once', action='store_true', help='Exit after the selected recording ends')
    parser.add_argument('--no-guides', action='store_true')
    parser.add_argument('--check', action='store_true', help='Validate files without a GUI')
    parser.add_argument('--render-check', type=Path, help='Save an offscreen frame, then exit')
    parser.add_argument('--camera', type=Path, default=PRESENTATION/'camera.json',
                        help='Shared camera preset; auto-load when this file exists')
    parser.add_argument('--time', type=float, help='Start or render at an exact task time')
    parser.add_argument('--paused', action='store_true', help='Start paused')
    parser.add_argument('--width', type=int, help='Override the preset window width')
    parser.add_argument('--height', type=int, help='Override the preset window height')
    parser.add_argument('--no-controls', action='store_true', help='Hide the camera panel; C opens it')
    parser.add_argument('--screenshot-dir', type=Path, default=PRESENTATION/'screenshots')
    args = parser.parse_args(argv)
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error('--speed must be finite and positive')
    case, frames, model = load_case(args.case)
    duration = replay_duration(case, frames)
    stop_time = float(case.get('stop_time_s', 6.4))
    data = mujoco.MjData(model)
    if args.time is not None and (not np.isfinite(args.time) or not 0 <= args.time <= duration):
        parser.error(f'--time must be between 0 and {duration:g}')
    try:
        preset = (load_camera(args.camera) if args.camera.exists()
                  else CameraPreset(fovy=float(model.vis.global_.fovy)))
        if args.width is not None:
            preset.width_px = args.width
        if args.height is not None:
            preset.height_px = args.height
        preset.validate()
    except (ValueError, OSError, TypeError) as exc:
        parser.error(f'Camera preset: {exc}')
    timing = '6 ms position PID, no preview' if args.case.startswith('D01') else '6 ms MPC, 9 intervals'
    print(f"{args.case}: {case['title']} | {timing} | {args.speed:g}x")
    print('Robot LEFT stays fixed PD. Gray: world up; orange/cyan: actual LEFT/RIGHT axes.')
    print(f'Recorded 0-{duration:g} s, including stop request at {stop_time:g} s. '
          'No physical angle amplification.', flush=True)
    if args.check:
        for t in (0., min(stop_time, duration)/2, min(stop_time, duration), duration):
            set_frame(model, data, frames, t)
            assert np.isfinite(data.site_xmat).all()
        print('PASS: hash, model, finite saved trajectory and forward kinematics')
        return
    if args.render_check:
        set_frame(model, data, frames, args.time if args.time is not None
                  else min(stop_time, duration)/2)
        camera = mujoco.MjvCamera()
        camera_at(camera, data, reset=True)
        preset.apply(camera, model, data)
        width, height = int(preset.width_px), int(preset.height_px)
        model.vis.global_.offwidth = max(width, model.vis.global_.offwidth)
        model.vis.global_.offheight = max(height, model.vis.global_.offheight)
        with mujoco.Renderer(
                model, height=height, width=width,
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
            draw_text(renderer._mjr_context, model, data, controller_label)
            mujoco.mjr_finish()
            pixels = np.empty((height, width, 3), dtype=np.uint8)
            mujoco.mjr_readPixels(pixels, None, renderer._rect,
                                  renderer._mjr_context)
            save_screenshot(args.render_check, pixels, args.case, case, data.time, preset)
        print(f'Saved actual replay frame: {args.render_check}')
        return

    mode = 'Single replay' if args.once else 'Looping replay'
    print(f'{mode}. SPACE: pause/resume; R: restart; G: guides; close window: exit.')
    print('Mouse: left rotate, right pan, middle/scroll zoom. Tracking follows translation.')
    print(f'Camera: {args.camera}; F5 saves, F9 loads. C opens the numeric controls.')
    print(f'Screenshots: {args.screenshot_dir}; S or Take screenshot. 1/2/3 switch methods; arrows step frames.')
    start_time = 0. if args.time is None else args.time
    set_frame(model, data, frames, start_time)
    try:
        interactive_replay(model, data, frames, case, args.speed, args.once,
                           not args.no_guides, case_name=args.case, preset=preset,
                           camera_path=args.camera, screenshot_dir=args.screenshot_dir,
                           start_time=start_time, paused=args.paused,
                           controls=not args.no_controls)
        if args.once:
            print(f'Replay closed at t = {data.time:.3f} / {duration:g} s', flush=True)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
