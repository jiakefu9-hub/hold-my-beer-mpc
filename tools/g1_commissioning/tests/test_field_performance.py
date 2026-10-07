"""Host-policy tests: all privileged actions mocked, no hardware or sysfs writes."""
from contextlib import ExitStack, redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import field_performance as perf


class PerformanceTest(unittest.TestCase):
    def state(self, governor='performance', platform='performance', limit=40):
        return dict(governors={'2':governor,'7':governor},
                    platform_profile=platform,rt_priority_limits=[limit,limit])

    def setup_mocks(self, before, after=None):
        stack=ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(perf,'field_cpus',return_value=[2,7]))
        stack.enter_context(patch.object(perf,'snapshot',side_effect=[before,after or before]))
        stack.enter_context(patch.object(perf,'_read',return_value='balanced performance'))
        stack.enter_context(patch.object(perf.resource,'getrlimit',return_value=(40,40)))
        stack.enter_context(redirect_stdout(io.StringIO()))
        return stack.enter_context(patch.object(perf,'_sudo'))

    def test_already_ready_does_not_request_sudo(self):
        sudo=self.setup_mocks(self.state())
        result=perf.prepare(7,compute_process=True,rt_priority=20)
        sudo.assert_not_called()
        self.assertEqual(result['actions'],[])

    def test_applies_and_verifies_governors_platform_and_permissions(self):
        sudo=self.setup_mocks(self.state('powersave','low-power',0),self.state())
        with patch.object(perf.resource,'getrlimit',return_value=(0,0)):
            result=perf.prepare(7,compute_process=True,rt_priority=20)
        self.assertEqual(sudo.call_args_list[0].args[0],
                         ['/usr/bin/cpupower','-c','2,7','frequency-set','-g','performance'])
        self.assertEqual(sudo.call_args_list[1].args,(['/usr/bin/tee',perf.PLATFORM],'performance\n'))
        self.assertEqual(sudo.call_args_list[2].args[0][-1],'--rtprio=40:40')
        self.assertEqual(len(result['actions']),3)

    def test_checks_setting_not_just_command_return_code(self):
        self.setup_mocks(self.state('powersave'))
        with self.assertRaisesRegex(RuntimeError,'verification failed'):
            perf.prepare(7)

    def test_no_password_capture_no_sudoers_install(self):
        with patch.object(perf.subprocess,'run',side_effect=subprocess.CalledProcessError(
                1,['sudo'],stderr='password required')) as run:
            with self.assertRaisesRegex(RuntimeError,'sudo -v'):
                perf._sudo(['/usr/bin/cpupower','-c','7','frequency-set','-g','performance'])
        self.assertEqual(run.call_args.args[0][:2],['/usr/bin/sudo','-n'])
        self.assertIsNone(run.call_args.kwargs['input'])

    def test_unsupported_platform_fails_before_changes(self):
        sudo=self.setup_mocks(self.state(platform=None))
        with self.assertRaisesRegex(RuntimeError,'unavailable'):
            perf.prepare(7)
        sudo.assert_not_called()

    def test_cpu_selection_matches_transport_and_includes_smt(self):
        def read(path):
            return '6-7' if '/cpu7/' in path else '1-2'
        with patch.object(perf.os,'sched_getaffinity',return_value={0,1,2,6,7}), \
             patch.object(perf,'_read',side_effect=read):
            self.assertEqual(perf.field_cpus(7,True),[1,2,6,7])

    def test_mpc_offline_preflight_never_changes_host(self):
        import g1_walk_mpc as mpc
        with patch.object(perf,'prepare') as prepare, \
             patch.object(mpc,'select_cpu'), patch.object(mpc,'preflight',return_value={}), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(mpc.main(['--preflight']),0)
        prepare.assert_not_called()


if __name__=='__main__':
    unittest.main()
