"""Camera presets and an optional Tk control panel for saved-state playback."""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math


@dataclass
class CameraPreset:
    azimuth: float = 135.
    elevation: float = -10.
    distance: float = 1.65
    fovy: float = 45.
    offset_x: float = 0.
    offset_y: float = 0.
    height: float = 1.02
    width_px: int = 1000
    height_px: int = 1000
    lock_aspect: bool = True

    def validate(self):
        for name, value in asdict(self).items():
            if not math.isfinite(float(value)):
                raise ValueError(f'{name} must be finite')
        if not -90. <= self.elevation <= 90.:
            raise ValueError('Elevation must be between -90 and 90 degrees')
        if not .02 <= self.distance <= 100.:
            raise ValueError('Distance must be between 0.02 and 100 metres')
        if not 1. <= self.fovy <= 150.:
            raise ValueError('Field of view must be between 1 and 150 degrees')
        if any(int(v) != v or not 240 <= v <= 7680
               for v in (self.width_px, self.height_px)):
            raise ValueError('Window dimensions must be integers from 240 to 7680')
        if not isinstance(self.lock_aspect, bool):
            raise ValueError('Aspect lock must be true or false')
        return self

    @classmethod
    def capture(cls, camera, model, data, window_size, lock_aspect=True):
        return cls(float(camera.azimuth), float(camera.elevation),
                   float(camera.distance), float(model.vis.global_.fovy),
                   float(camera.lookat[0]-data.qpos[0]),
                   float(camera.lookat[1]-data.qpos[1]),
                   float(camera.lookat[2]), *map(int, window_size),
                   lock_aspect).validate()

    def aspect_terms(self):
        self.validate()
        if not self.lock_aspect:
            return None
        width, height = int(self.width_px), int(self.height_px)
        divisor = math.gcd(width, height)
        return width//divisor, height//divisor

    def apply(self, camera, model, data):
        self.validate()
        camera.azimuth = self.azimuth
        camera.elevation = self.elevation
        camera.distance = self.distance
        camera.lookat[:] = (data.qpos[0]+self.offset_x,
                            data.qpos[1]+self.offset_y, self.height)
        model.vis.global_.fovy = self.fovy
        return (self.offset_x, self.offset_y, self.height)


def save_camera(path, preset):
    path = Path(path)
    preset.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'version': 1, 'camera': asdict(preset),
               'target': 'XY offsets from recorded root; Z is world height'}
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def load_camera(path):
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict) or payload.get('version') != 1:
        raise ValueError('Unsupported camera preset version')
    try:
        return CameraPreset(**payload['camera']).validate()
    except (KeyError, TypeError) as exc:
        raise ValueError(f'Invalid camera preset: {exc}') from exc


class ControlPanel:
    """Pump from the GLFW main thread; do not start another GUI event loop."""
    labels = (
        ('azimuth', 'Azimuth (deg)'), ('elevation', 'Elevation (deg)'),
        ('distance', 'Distance / zoom (m)'), ('fovy', 'Field of view (deg)'),
        ('offset_x', 'Target X offset (m)'), ('offset_y', 'Target Y offset (m)'),
        ('height', 'Target height (m)'),
        ('width_px', 'Window width'), ('height_px', 'Window height'),
    )

    def __init__(self, get_view, actions, duration, initial_case, screenshot_dir,
                 *, get_speed=lambda: 1.):
        import tkinter as tk
        from tkinter import font as tkfont, ttk
        self.tk = tk
        self.root = tk.Tk()
        self.root.title('G1 replay - camera and screenshots')
        self.root.geometry(f'{min(800, self.root.winfo_screenwidth()-80)}x'
                           f'{min(1280, self.root.winfo_screenheight()-100)}')
        self.root.minsize(600, 500)
        self.style = ttk.Style(self.root)
        if 'clam' in self.style.theme_names():
            self.style.theme_use('clam')
        self.font_px = 24
        self.ui_fonts = [tkfont.nametofont(name, root=self.root) for name in (
            'TkDefaultFont', 'TkTextFont', 'TkMenuFont', 'TkHeadingFont',
            'TkCaptionFont', 'TkSmallCaptionFont', 'TkIconFont', 'TkTooltipFont')]
        self.heading_font = tkfont.Font(root=self.root, family='DejaVu Sans',
                                        size=-26, weight='bold')
        self.font_size_text = tk.StringVar()
        self.set_text_size(24)
        self.style.configure('TButton', font='TkDefaultFont', padding=(12, 8))
        self.style.configure('TEntry', font='TkTextFont', padding=(7, 5))
        self.style.configure('TRadiobutton', font='TkDefaultFont', padding=(0, 5))
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.get_view, self.actions = get_view, actions
        self.get_speed = get_speed
        self.closed, self.dirty = False, False
        self.fields = {}
        self.message = tk.StringVar(value='Drag in the replay window, or enter numbers here.')
        outer = ttk.Frame(self.root)
        outer.pack(fill='both', expand=True)
        canvas = tk.Canvas(outer, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient='vertical', command=canvas.yview)
        scrollbar.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        canvas.configure(yscrollcommand=scrollbar.set)
        body = ttk.Frame(canvas, padding=18)
        body_window = canvas.create_window((0, 0), window=body, anchor='nw')
        body.bind('<Configure>', lambda event: canvas.configure(scrollregion=canvas.bbox('all')))
        self.wrapped_labels = []

        def resize_body(event):
            canvas.itemconfigure(body_window, width=event.width)
            for label in self.wrapped_labels:
                label.configure(wraplength=max(250, event.width-45))

        canvas.bind('<Configure>', resize_body)
        self.root.bind_all('<MouseWheel>', lambda event: canvas.yview_scroll(
            -1 if event.delta > 0 else 1, 'units'))
        self.root.bind_all('<Button-4>', lambda event: canvas.yview_scroll(-1, 'units'))
        self.root.bind_all('<Button-5>', lambda event: canvas.yview_scroll(1, 'units'))
        ttk.Label(body, text='Shared camera for all three methods',
                  font=self.heading_font).pack(anchor='w')
        font_row = ttk.Frame(body)
        font_row.pack(fill='x', pady=(6, 12))
        ttk.Label(font_row, text='Text size:').pack(side='left', padx=(0, 12))
        ttk.Button(font_row, text='A -', command=lambda: self.set_text_size(
            self.font_px-2)).pack(side='left', padx=3)
        ttk.Button(font_row, text='A +', command=lambda: self.set_text_size(
            self.font_px+2)).pack(side='left', padx=3)
        ttk.Label(font_row, textvariable=self.font_size_text).pack(side='left', padx=12)
        form = ttk.Frame(body)
        form.pack(fill='x', pady=8)
        for row, (name, label) in enumerate(self.labels):
            ttk.Label(form, text=label).grid(row=row, column=0, sticky='w', pady=5)
            var = tk.StringVar()
            self.fields[name] = var
            entry = ttk.Entry(form, textvariable=var, width=16, font='TkTextFont')
            entry.grid(row=row, column=1, sticky='ew', padx=8)
            entry.bind('<KeyRelease>', self.mark_dirty)
            entry.bind('<Return>', lambda event: self.apply())
        form.columnconfigure(1, weight=1)
        size_row = ttk.Frame(body)
        size_row.pack(fill='x', pady=(2, 6))
        ttk.Label(size_row, text='Window ratio:').pack(side='left', padx=(0, 8))
        self.aspect_text = tk.StringVar()
        aspect = ttk.Combobox(size_row, textvariable=self.aspect_text,
                              values=('1:1', '4:3', '16:9', 'Custom'),
                              state='readonly', width=8, font='TkTextFont')
        aspect.pack(side='left')
        aspect.bind('<<ComboboxSelected>>', lambda event: self.choose_aspect())
        self.lock_aspect = tk.BooleanVar(value=True)
        ttk.Checkbutton(size_row, text='Lock ratio', variable=self.lock_aspect,
                        command=self.apply).pack(side='left', padx=12)
        ttk.Button(body, text='Recommended size: 1000 x 1000 (1:1)',
                   command=self.recommended_size).pack(fill='x', pady=4)
        for buttons in (
            (('Apply numbers', self.apply), ('Use current mouse view', self.refresh)),
            (('Save camera (F5)', self.save),
             ('Load camera (F9)', lambda: self.call('load'))),
            (('Reset camera', lambda: self.call('reset')),),
        ):
            row = ttk.Frame(body)
            row.pack(fill='x', pady=2)
            for label, command in buttons:
                ttk.Button(row, text=label, command=command).pack(
                    side='left', fill='x', expand=True, padx=2)
        ttk.Separator(body).pack(fill='x', pady=10)
        ttk.Label(body, text='Switch method at the SAME time and camera:').pack(anchor='w')
        self.case = tk.StringVar(value=initial_case)
        for case, label in (('D01', '1  Position PID'), ('D02', '2  No-preview MPC'),
                            ('D03', '3  Template-preview MPC')):
            ttk.Radiobutton(body, text=label, variable=self.case, value=case,
                            command=lambda: self.call('case', self.case.get())).pack(anchor='w')
        ttk.Separator(body).pack(fill='x', pady=8)
        self.speed_text = tk.StringVar(value=f'{self.get_speed():g}')
        self.speed_status = tk.StringVar()
        speed_row = ttk.Frame(body)
        speed_row.pack(fill='x', pady=(0, 4))
        ttk.Label(speed_row, text='Playback speed:').pack(side='left', padx=(0, 8))
        speed_entry = ttk.Entry(speed_row, textvariable=self.speed_text,
                                width=7, font='TkTextFont')
        speed_entry.pack(side='left')
        speed_entry.bind('<Return>', lambda event: self.apply_speed())
        ttk.Button(speed_row, text='Apply speed', command=self.apply_speed).pack(
            side='left', padx=8)
        speed_buttons = ttk.Frame(body)
        speed_buttons.pack(fill='x')
        for speed in (.05, .1, .25, .5, 1.):
            ttk.Button(speed_buttons, text=f'{speed:g}x', width=6,
                       command=lambda value=speed: self.choose_speed(value)).pack(
                           side='left', expand=True, fill='x', padx=2)
        ttk.Label(body, textvariable=self.speed_status).pack(anchor='w', pady=4)
        row = ttk.Frame(body)
        row.pack(fill='x', pady=8)
        self.time_text = tk.StringVar(value='0')
        ttk.Label(row, text=f'Time (0-{duration:g} s)').pack(side='left')
        entry = ttk.Entry(row, textvariable=self.time_text, width=10, font='TkTextFont')
        self.time_entry = entry
        self.time_dirty = False
        entry.bind('<KeyRelease>', self.mark_time_dirty)
        entry.pack(side='left', padx=8)
        entry.bind('<Return>', lambda event: self.seek())
        ttk.Button(row, text='Go / pause', command=self.seek).pack(side='left')
        row = ttk.Frame(body)
        row.pack(fill='x')
        for label, callback in (
            ('Play / pause', lambda: self.call('pause')),
            ('Previous frame', lambda: self.call('step', -1)),
            ('Next frame', lambda: self.call('step', 1)),
        ):
            ttk.Button(row, text=label, command=callback).pack(side='left', expand=True)
        ttk.Button(body, text='Take screenshot (S)',
                   command=lambda: self.call('screenshot')).pack(fill='x', pady=10)
        self.clock = tk.StringVar()
        ttk.Label(body, textvariable=self.clock).pack(anchor='w')
        message_label = ttk.Label(body, textvariable=self.message, wraplength=730)
        message_label.pack(anchor='w', pady=6)
        path_label = ttk.Label(body, text='PNG + camera/time metadata:\n'+str(screenshot_dir),
                               wraplength=730)
        path_label.pack(anchor='w')
        self.wrapped_labels.extend((message_label, path_label))
        self.refresh()

    def set_text_size(self, pixels):
        self.font_px = int(max(16, min(38, pixels)))
        for font in self.ui_fonts:
            font.configure(family='DejaVu Sans', size=-self.font_px)
        self.heading_font.configure(size=-(self.font_px+2))
        self.font_size_text.set(f'{self.font_px} px')

    def mark_dirty(self, event=None):
        if event is not None and event.keysym in ('Return', 'Tab', 'Shift_L', 'Shift_R'):
            return
        self.dirty = True

    def mark_time_dirty(self, event=None):
        if event is not None and event.keysym not in ('Return', 'Tab'):
            self.time_dirty = True

    def call(self, action, *args):
        try:
            result = self.actions[action](*args)
            if action in ('load', 'reset'):
                self.refresh()
            if result:
                self.message.set(str(result))
        except (ValueError, OSError, TypeError) as exc:
            self.message.set(str(exc))

    def apply(self):
        try:
            values = {name: float(var.get()) for name, var in self.fields.items()}
            view = CameraPreset(**values, lock_aspect=self.lock_aspect.get()).validate()
            self.actions['apply'](view)
            self.dirty = False
            self.update_aspect(view)
            self.message.set('Camera and window size applied.')
            return True
        except (ValueError, OSError) as exc:
            self.message.set(str(exc))
            return False

    def update_aspect(self, view):
        terms = view.aspect_terms()
        self.lock_aspect.set(view.lock_aspect)
        name = f'{terms[0]}:{terms[1]}' if terms is not None else 'Custom'
        self.aspect_text.set(name if name in ('4:3', '16:9', '1:1') else 'Custom')

    def choose_aspect(self):
        name = self.aspect_text.get()
        if name == 'Custom':
            return
        try:
            numer, denom = map(int, name.split(':'))
            width = float(self.fields['width_px'].get())
            if not math.isfinite(width):
                raise ValueError('Window width must be finite')
            units = max(math.ceil(240/min(numer, denom)), round(width/numer))
            self.fields['width_px'].set(str(units*numer))
            self.fields['height_px'].set(str(units*denom))
            self.lock_aspect.set(True)
            self.apply()
        except ValueError as exc:
            self.message.set(str(exc))

    def recommended_size(self):
        self.fields['width_px'].set('1000')
        self.fields['height_px'].set('1000')
        self.lock_aspect.set(True)
        self.apply()

    def apply_speed(self):
        self.call('speed', self.speed_text.get())

    def choose_speed(self, speed):
        self.speed_text.set(f'{speed:g}')
        self.apply_speed()

    def save(self):
        if not self.dirty or self.apply():
            self.call('save')

    def seek(self):
        try:
            self.actions['seek'](float(self.time_text.get()))
            self.time_dirty = False
        except ValueError as exc:
            self.message.set(str(exc))

    def refresh(self):
        self.dirty = False
        view = self.get_view()
        for name, var in self.fields.items():
            var.set(f'{getattr(view, name):.8g}')
        self.update_aspect(view)

    def pump(self, case, task_time, paused):
        if self.closed:
            return
        self.root.update_idletasks()
        self.root.update()
        if self.closed:
            return
        self.case.set(case)
        self.speed_status.set(f'Current speed: {self.get_speed():g}x')
        self.clock.set(f'{task_time:.3f} s  |  '+('PAUSED' if paused else 'PLAYING'))
        if not self.time_dirty and self.root.focus_get() != self.time_entry:
            self.time_text.set(f'{task_time:.3f}')
        if not self.dirty:
            self.refresh()

    def close(self):
        if not self.closed:
            self.closed = True
            self.root.destroy()
