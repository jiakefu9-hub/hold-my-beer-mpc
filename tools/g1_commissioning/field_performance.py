"""Normalize field host performance before DDS, without handling passwords.

Only PID/MPC executable entrypoints call prepare(); importing or --preflight
does nothing. Uses fixed system executables, never runs Python as root, never
installs sudoers rules. A user can authorize sudo in their own launching
terminal with `sudo -v`; otherwise it fails before creating a publisher.
"""
import os
import resource
import subprocess

from mpc_host import _read, select_cpu
from realtime_environment import parse_cpu_list

PLATFORM = '/sys/firmware/acpi/platform_profile'


def field_cpus(cpu, compute_process=False):
    selected = select_cpu(cpu)
    siblings = parse_cpu_list(_read(
        f'/sys/devices/system/cpu/cpu{selected}/topology/thread_siblings_list'))
    targets = siblings | {selected}
    if compute_process:
        # Match ControlThreadScope.prepare_workers and ProcessMpcRuntime.
        housekeeping = set(os.sched_getaffinity(0)) - targets
        if not housekeeping:
            raise ValueError('need housekeeping CPUs for MPC transport')
        transport = 2 if 2 in housekeeping else min(housekeeping)
        targets |= {transport} | parse_cpu_list(_read(
            f'/sys/devices/system/cpu/cpu{transport}/topology/thread_siblings_list'))
    return sorted(targets)


def snapshot(cpus):
    return {
        'governors': {str(cpu): _read(
            f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor') for cpu in cpus},
        'platform_profile': _read(PLATFORM),
        'rt_priority_limits': list(resource.getrlimit(resource.RLIMIT_RTPRIO)),
    }


def _sudo(arguments, text=None):
    # -n never prompts or reads a password from this process. Interactive
    # authentication belongs to sudo -v in the user's terminal, not chat.
    try:
        subprocess.run(['/usr/bin/sudo', '-n', *arguments], input=text,
                       text=True, capture_output=True, check=True, timeout=15)
    except (subprocess.SubprocessError, OSError) as exc:
        detail = getattr(exc, 'stderr', '') or str(exc)
        raise RuntimeError('Host preparation failed before DDS. In the SAME launching '
                           'terminal run `sudo -v`, then retry. Never send passwords to '
                           'Codex or run the robot program with sudo. '
                           f'Settings already applied are not rolled back. Detail: {detail}') from exc


def prepare(cpu, *, compute_process=False, rt_priority=0):
    priority = int(rt_priority)
    if not 0 <= priority <= 40:
        raise ValueError('RT priority must be 0..40')
    cpus = field_cpus(cpu, compute_process)
    before = snapshot(cpus)
    if any(value is None for value in before['governors'].values()):
        raise RuntimeError('CPU governor unavailable; review this host before field execution')
    if before['platform_profile'] is None:
        raise RuntimeError('ACPI platform profile unavailable; review this host before field execution')
    actions = []
    # Fail unsupported profile before changing any setting.
    choices = (_read(PLATFORM + '_choices') or '').split()
    if before['platform_profile'] != 'performance' and 'performance' not in choices:
        raise RuntimeError('This host does not advertise a performance platform profile')
    if any(value != 'performance' for value in before['governors'].values()):
        _sudo(['/usr/bin/cpupower', '-c', ','.join(map(str, cpus)),
               'frequency-set', '-g', 'performance'])
        actions.append('cpu_governors_performance')
    if before['platform_profile'] != 'performance':
        _sudo(['/usr/bin/tee', PLATFORM], 'performance\n')
        actions.append('platform_profile_performance')
    soft, hard = resource.getrlimit(resource.RLIMIT_RTPRIO)
    if priority and soft != resource.RLIM_INFINITY and soft < priority:
        if hard == resource.RLIM_INFINITY or hard >= priority:
            resource.setrlimit(resource.RLIMIT_RTPRIO, (priority, hard))
        else:
            _sudo(['/usr/bin/prlimit', '--pid', str(os.getpid()), '--rtprio=40:40'])
        actions.append('current_process_rt_permission')
    after = snapshot(cpus)
    if (any(value != 'performance' for value in after['governors'].values())
            or after['platform_profile'] != 'performance'
            or (priority and after['rt_priority_limits'][0] != resource.RLIM_INFINITY
                and after['rt_priority_limits'][0] < priority)):
        raise RuntimeError('Host settings verification failed before DDS; no robot output')
    print(f'Field host checked: CPUs {cpus}, governor/platform=performance, FIFO request={priority}')
    return {'before': before, 'after': after, 'actions': actions, 'cpus': cpus,
            'password_handled': False, 'robot_contacted': False,
            'limitations': 'fixed policy is not guaranteed frequency, no thermal or deadline guarantee'}
