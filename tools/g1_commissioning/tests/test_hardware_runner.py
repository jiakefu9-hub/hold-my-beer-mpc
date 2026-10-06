"""Execute the real field runner against in-memory transports, never DDS.

The controller is a stationary stub; actual MPC math has separate replay tests.
Real worker threads use an accelerated monotonic clock (2x). Every SDK import
is replaced and socket creation is forbidden, even if the laptop is plugged in.
This validates software sequencing, not firmware or network behavior.
"""
import contextlib
import gc
import io
import json
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import g1_walk_pid as runner
from hardware_pid_control import ARM_MOTOR_INDICES, FixedH0Heading, HardwarePidPlan


class Clock:
    factor = 2.

    def __init__(self):
        self.start = time.monotonic_ns()
        self.local = threading.local()

    def monotonic_ns(self):
        value = self.start + round((time.monotonic_ns()-self.start)*self.factor)
        self.local.last_ns = value
        return value

    def monotonic(self):
        return self.monotonic_ns()*1e-9

    def sleep(self, seconds):
        time.sleep(seconds/self.factor)


class Harness:
    def __init__(self, failure=None, stationary=False, torque=False):
        self.clock, self.failure, self.stationary = Clock(), failure, stationary
        self.torque = torque
        self.rows, self.writes, self.velocities, self.closed, self.registrations = [], [], [], [], []
        self.factory_calls = self.publisher_count = 0
        self.epoch = None
        self.streams = None
        self.injected_at = None
        self.events = []
        self.handlers = {signal.SIGINT: signal.SIG_DFL, signal.SIGTERM: signal.SIG_DFL}

    def event(self):
        item = threading.Event()
        wait = item.wait
        item.wait = lambda timeout=None: wait(None if timeout is None else timeout/self.clock.factor)
        self.events.append(item)
        return item

    def install_signal(self, kind, handler):
        old = self.handlers[kind]
        self.handlers[kind] = handler
        return old

    def task_s(self):
        return 0. if self.epoch is None else (self.clock.monotonic_ns()-self.epoch)*1e-9

    def journal(self):
        harness = self

        class MemoryJournal:
            failed = threading.Event()
            dropped = 0

            def record(self, row):
                harness.rows.append(runner.json_safe(row))

        return MemoryJournal()

    def runtime(self):
        harness = self

        class Controller:
            control_dt = .006
            last_diagnostics = {}

            def reset(self): pass
            def set_measured_dq(self, dq): pass

            def step(self, slots, quaternion, yaw0, dt):
                return runner.EXPECTED_TARGET_Q[5:10].copy(), np.zeros(5), {"mpc_active": True}

        class Runtime:
            stationary = harness.stationary
            controller = Controller()
            actuation = 'measured_torque_preview' if harness.torque else 'reference_servo'
            field_trial = harness.torque

            def __init__(self):
                from hardware_mpc_field import TorqueHandback
                self.handback = TorqueHandback()
                self.predictor = None
                if harness.failure == 'startup_delay':
                    from hardware_mpc_predictor import HardwareMpcPredictor
                    self.predictor = HardwareMpcPredictor('hold_current')

            def prepare_startup(self):
                if self.predictor is not None:
                    harness.clock.sleep(.120)  # slow pre-task GC
                    assert harness.epoch is None

            def enter_control_thread(self, cpu):
                if self.predictor is not None:
                    harness.clock.sleep(.120)  # affinity / host evidence
                    harness.clock.monotonic_ns()  # refresh fake subscriber clock
                return {'test_fake_affinity': True}

            def validate_field_entry(self):
                assert harness.torque

            def release_frame(self, elapsed_s):
                return self.handback.normal_release(elapsed_s)

            def accept_packet(self, frame, packet):
                if harness.torque:
                    self.handback.accept(packet)

            def make_message(self, frame, low, constructor, crc):
                if not harness.torque:
                    return runner.make_arm_message(frame,low,constructor,crc)
                if ('stage' not in frame or frame['stage'] in
                        {'normal_stop_wait','operator_stop_wait','operator_arm_release'}):
                    self.handback.apply(frame)
                elif not frame.get('diagnostics',{}).get('torque_handover'):
                    frame['diagnostics'].update(controller_kind=self.actuation,
                        tau_ff_candidate_nm=[-1.]*5 if frame['weight']>0 else [0.]*5)
                motors=[SimpleNamespace(q=0.,dq=0.,kp=0.,kd=0.,tau=0.) for _ in range(35)]
                for slot,i in enumerate(ARM_MOTOR_INDICES):
                    for attr,key in (('q','q_rad'),('dq','dq_rad_s'),('kp','kp'),('kd','kd')):
                        setattr(motors[i],attr,float(frame[key][slot]))
                for j,i in enumerate(range(22,27)):
                    motors[i].tau=frame['diagnostics']['tau_ff_candidate_nm'][j]
                motors[29].q=frame['weight']
                return SimpleNamespace(frame=frame,motor_cmd=motors,mode_pr=0,mode_machine=4,crc=0)

            def create_plan(self, initial, profile):
                return HardwarePidPlan(initial, profile["target_q_array"],
                                       profile["kp_array"], profile["kd_array"], self.controller)

            def set_epoch(self, epoch):
                harness.epoch = epoch
                if self.predictor is not None:
                    self.predictor.set_grid_origin(epoch)
                    for t in range(epoch-500_000_000,epoch+1,2_000_000):
                        self.predictor.observe_low(t,np.zeros(35),np.zeros(35))
                        self.predictor.observe_imu(t,[1,0,0,0],[0,0,0],[0,0,9.81])
                    self.predictor.query(epoch,0.,use_learned=False)

            def prepare(self, now, low, imu, yaw0, task_s, **kwargs):
                if self.predictor is not None:
                    self.predictor.observe_low(low.received_ns,low.q,low.dq)
                    self.predictor.observe_imu(imu.received_ns,imu.quaternion,
                                              imu.gyro,imu.accelerometer)
                    self.predictor.query(min(low.received_ns,imu.received_ns),0.,use_learned=False)
                    if task_s >= .1 and harness.injected_at is None:
                        harness.injected_at = now
                        harness.handlers[signal.SIGINT](None,None)
                if task_s >= 6 and harness.injected_at is None:
                    harness.injected_at = now
                    if harness.failure == "compute":
                        raise RuntimeError("injected MPC solve failure")
                    if harness.failure == "remote":
                        keys = (1 << 5) | (1 << 9)
                        harness.streams.interlock.observe_remote([0, 0, keys & 255, keys >> 8])
                    if harness.failure == "fsm":
                        harness.streams.interlock.observe_fsm(0, 1, now, now)
                    if harness.failure == "operator":
                        harness.handlers[signal.SIGINT](None, None)

        return Runtime()

    def stream_class(self):
        harness = self

        class Streams:
            def __init__(self, journal, interlock, crc, observer=None):
                self.interlock, self.heading, self.sequence = interlock, FixedH0Heading(), 0
                self.lock = threading.Lock()
                harness.streams = self

            def low_callback(self, message): pass
            def imu_callback(self, message): pass
            def set_epoch(self, ns): harness.epoch = ns
            def heading_current(self):
                with self.lock:
                    return self.heading.current()

            def freeze_heading(self):
                with self.lock:
                    return self.heading.freeze(), self.heading.current()

            def latest(self):
                # A real subscriber snapshot already exists when health()
                # reads it. Do not manufacture a "future" snapshot after the
                # check's captured now_ns, as a lazy fake otherwise would.
                now = harness.clock.local.last_ns
                t = harness.task_s()
                yaw = .3 if t < 5 else .3 + .01*(t-5)
                with self.lock:
                    self.heading.observe(now, t, yaw, 0.)
                    self.sequence += 1
                q = np.zeros(35)
                q[list(ARM_MOTOR_INDICES)] = runner.EXPECTED_TARGET_Q
                stamp = now-200_000_000 if harness.failure == "stale" and t >= 6 else now
                return (runner.LowSnapshot(stamp, self.sequence, self.sequence, 0, 4, q, np.zeros(35), True),
                        runner.ImuSnapshot(now, self.sequence, np.array([1.,0,0,0]),
                                           np.zeros(3), np.zeros(3), np.array([0.,0,9.81])))

        return Streams

    def modules(self, root):
        harness = self

        class Subscriber:
            def __init__(self, topic, typ): self.topic = topic

            def Init(self, callback, queue_length):
                if harness.failure == "subscriber_init" and self.topic == "rt/secondary_imu":
                    raise RuntimeError("injected IMU subscription init failure")

        class Publisher:
            def __init__(self, topic, typ):
                assert topic == "rt/arm_sdk"
                self.topic = topic
                harness.publisher_count += 1

            def Init(self): pass

            def Write(self, frame):
                now = harness.clock.monotonic_ns()
                harness.writes.append(dict(stamp=now, **runner.json_safe(frame)))
                if harness.failure in {"write_false", "write_raise"} and harness.task_s() >= 6:
                    harness.injected_at = now
                    if harness.failure == "write_raise":
                        raise RuntimeError("injected DDS write exception")
                    return False
                return True

        class Client:
            def __init__(self, service, lease): pass
            def SetTimeout(self, timeout): pass
            def _SetApiVerson(self, version): pass

            def _RegistApi(self, api, priority):
                harness.registrations.append(api)
                if api == 7105 and harness.failure == 'startup_delay':
                    harness.clock.sleep(.120)  # slow RPC construction
                if api == 7105 and harness.failure == "velocity_init":
                    raise RuntimeError("injected velocity client init failure")

            def _Call(self, api, payload):
                if api == 7001:
                    return 0, '{"data":500}'
                assert api == 7105
                harness.velocities.append((harness.task_s(), json.loads(payload)["velocity"]))
                return 0, '{}'

        def factory(domain, nic):
            assert nic == "in-memory-only"
            harness.factory_calls += 1

        base = "unitree_sdk2py"
        values = {base: {"__file__": str(root / "__init__.py")},
            base+".core.channel": dict(ChannelFactoryInitialize=factory, ChannelPublisher=Publisher,
                                        ChannelSubscriber=Subscriber),
            base+".g1.loco.g1_loco_api": dict(LOCO_API_VERSION="1.0.0.0", LOCO_SERVICE_NAME="sport",
                ROBOT_API_ID_LOCO_GET_FSM_ID=7001, ROBOT_API_ID_LOCO_SET_VELOCITY=7105),
            base+".idl.default": dict(unitree_hg_msg_dds__LowCmd_=object),
            base+".idl.unitree_hg.msg.dds_": dict(IMUState_=object, LowCmd_=object, LowState_=object),
            base+".rpc.client": dict(Client=Client), base+".utils.crc": dict(CRC=object)}
        result = {}
        for name, attributes in values.items():
            module = ModuleType(name)
            module.__dict__.update(attributes)
            result[name] = module
        return result

    def run(self):
        profile = dict(robot_id="FAKE", startup_valid_samples_int=5,
                       target_q_array=runner.EXPECTED_TARGET_Q.copy(),
                       kp_array=np.r_[np.full(11,20.),0,0], kd_array=np.r_[np.ones(11),0,0])
        args = SimpleNamespace(nic="in-memory-only", cpu=2)
        clock = self.clock

        class FastPeriodicClock(runner.PeriodicClock):
            def wait(self):
                clock.sleep(max(0., (self.scheduled_ns-clock.monotonic_ns())*1e-9))

        def packet(frame, state, constructor, crc):
            # Runner expects attributes for audit; transport stores a dict.
            return SimpleNamespace(frame=frame, mode_pr=0, mode_machine=4, crc=0)

        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            # Match MpcRuntime's bounded active-run GC policy. Do not amplify
            # a test-fixture collection pause into a fabricated sensor outage.
            if gc.isenabled():
                gc.collect()
                gc.disable()
                stack.callback(gc.enable)
            root = Path(directory)
            for name in ("core/channel.py", "rpc/client.py", "rpc/client_base.py", "utils/crc.py",
                         "g1/loco/g1_loco_api.py", "idl/unitree_hg/msg/dds_/_LowCmd_.py",
                         "idl/unitree_hg/msg/dds_/_LowState_.py"):
                p = root/name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("# Fake SDK provenance fixture, not an executable SDK\n")
            modules = self.modules(root)
            publisher = modules["unitree_sdk2py.core.channel"].ChannelPublisher
            write = publisher.Write
            publisher.Write = lambda obj, msg: write(obj, msg.frame)
            for patch in (
                mock.patch.dict(sys.modules, modules),
                mock.patch("socket.socket", side_effect=AssertionError("network forbidden in runner test")),
                mock.patch.object(runner, "time", clock),
                mock.patch.object(runner, "threading", SimpleNamespace(Event=self.event, Thread=threading.Thread, Lock=threading.Lock)),
                mock.patch.object(runner, "signal", SimpleNamespace(SIGINT=signal.SIGINT, SIGTERM=signal.SIGTERM, signal=self.install_signal)),
                mock.patch.object(runner, "Streams", self.stream_class()),
                mock.patch.object(runner, "PeriodicClock", FastPeriodicClock),
                mock.patch.object(runner, "make_arm_message", packet),
                mock.patch.object(runner, "execution_evidence", return_value={}),
                mock.patch.object(runner, "pin_control_thread", return_value={"test_fake_affinity": True}),
                mock.patch.object(runner.os, "sched_setaffinity"),
                mock.patch.object(runner, "close_sdk_endpoint", side_effect=lambda e,k: self.closed.append(e.topic) or "already_closed"),
                mock.patch("builtins.input", return_value="NO" if self.failure == "confirmation" else "EXECUTE FAKE"),
            ):
                stack.enter_context(patch)
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            result = runner.run_device(args, profile, None, {}, self.journal(), self.runtime())
        self.assert_no_mode_api()
        return result

    def assert_no_mode_api(self):
        assert set(self.registrations) <= {7001,7105}
        assert self.handlers == {signal.SIGINT: signal.SIG_DFL, signal.SIGTERM: signal.SIG_DFL}


class FieldRunnerTest(unittest.TestCase):
    def test_slow_startup_uses_fresh_predictor_epoch_before_first_write(self):
        h=Harness('startup_delay',stationary=True,torque=True)
        self.assertEqual(h.run(),130)
        self.assertTrue(h.writes)
        first_write=next(r for r in h.rows if r.get('event')=='dds_write')
        # The three injected 120 ms pre-task stalls must not enter task time.
        # Exact first-loop scheduling is intentionally not a wall-clock test:
        # assert ramp semantics and a broad fresh-epoch bound instead.
        self.assertLess(first_write['task_elapsed_s'],.1)
        self.assertAlmostEqual(h.writes[0]['weight'],
                               first_write['task_elapsed_s']/3.,places=10)
        self.assertEqual(h.writes[-1]['weight'],0.)
        self.assertFalse(any(r.get('event')=='session_fault' for r in h.rows))
        self.assertTrue(all(v==[0.,0.,0.] for _,v in h.velocities))
        events=[r.get('event') for r in h.rows]
        self.assertLess(events.index('control_runtime'),events.index('task_epoch'))

    def assert_gradual_release(self, h):
        frames = [r for r in h.rows if r.get("event") == "dds_write" and (
            "release" in r.get("stage", "") or r.get("stage") in {"arm_ramp_out", "complete"})]
        self.assertGreater(len(frames), 400)
        self.assertEqual(frames[-1]["weight"], 0.)
        self.assertGreaterEqual(frames[-1]["write_begin_monotonic_ns"]-frames[0]["write_begin_monotonic_ns"], 2_990_000_000)
        weights = np.array([r["weight"] for r in frames])
        self.assertLessEqual(np.max(-np.diff(weights)), .002+1e-12)
        self.assertLessEqual(np.max(np.diff(weights)), 1e-12)
        for r in frames:
            np.testing.assert_allclose(r["dq_command_rad_s"], 0.)

    def test_stationary_and_walk_normal_release(self):
        for stationary in (True, False):
            with self.subTest(stationary=stationary):
                h = Harness(stationary=stationary)
                self.assertEqual(h.run(), 0)
                self.assert_gradual_release(h)
                self.assertEqual(set(h.closed), {"rt/arm_sdk","rt/lowstate","rt/secondary_imu"})
                if stationary:
                    self.assertTrue(all(v == [0.,0.,0.] for _,v in h.velocities))
                    self.assertFalse(any(r.get("stage") == "forward_walk" for r in h.rows))
                else:
                    self.assertTrue(any(5 <= t < 15 and v[0] == .5 for t,v in h.velocities))
                    self.assertTrue(all(v[0] == 0 for t,v in h.velocities if t >= 15))
                self.assertEqual(h.velocities[-1][1], [0.,0.,0.])

    def test_nonzero_torque_normal_fault_operator_and_remote_handover(self):
        for failure, code in ((None,0),('compute',3),('operator',130),('remote',3)):
            with self.subTest(failure=failure):
                h=Harness(failure,stationary=True,torque=True)
                self.assertEqual(h.run(),code)
                if failure=='remote':
                    self.assertLess(h.writes[-1]['stamp'],h.injected_at)
                    continue
                self.assert_gradual_release(h)
                release=[r for r in h.writes if r.get('diagnostics',{}).get('torque_handover')]
                self.assertGreater(len(release),400)
                for r in release:
                    expected=-1. if r['weight']>0 else 0.
                    np.testing.assert_allclose(r['diagnostics']['tau_ff_candidate_nm'],expected)

    def test_mpc_compute_failure_and_operator_stop_release(self):
        for failure, code in (("compute",3),("operator",130)):
            with self.subTest(failure=failure):
                h = Harness(failure)
                self.assertEqual(h.run(), code)
                self.assert_gradual_release(h)
                self.assertEqual(h.velocities[-1][1], [0.,0.,0.])

    def test_dds_failure_never_retries_arm_write(self):
        for failure in ("write_false","write_raise"):
            with self.subTest(failure=failure):
                h = Harness(failure)
                self.assertEqual(h.run(), 3)
                self.assertEqual(h.writes[-1]["stamp"], h.injected_at)
                fault = next(r for r in h.rows if r.get("event") == "session_fault")
                self.assertFalse(fault["fault_release"]["attempted"])
                self.assertIn("further arm output disabled", fault["fault_release"]["reason"])
                self.assertEqual(h.velocities[-1][1], [0.,0.,0.])

    def test_remote_fsm_and_stale_state_stop_arm_output(self):
        for failure in ("remote","fsm","stale"):
            with self.subTest(failure=failure):
                h = Harness(failure)
                self.assertEqual(h.run(), 3)
                if h.injected_at is not None:
                    self.assertLess(h.writes[-1]["stamp"], h.injected_at)
                fault = next(r for r in h.rows if r.get("event") == "session_fault")
                self.assertFalse(fault["fault_release"]["completed"])

    def test_partial_initialization_confirmation_and_rpc_constructor_failures(self):
        for failure in ("subscriber_init","confirmation","velocity_init"):
            with self.subTest(failure=failure):
                h = Harness(failure)
                self.assertNotEqual(h.run(), 0)
                self.assertFalse(any(v[0] != 0 for _,v in h.velocities))
                self.assertIn("rt/lowstate", h.closed)
                self.assertIn("rt/secondary_imu", h.closed)
                if failure != "velocity_init":
                    self.assertEqual(h.publisher_count, 0)
                    self.assertEqual(h.writes, [])
                else:
                    self.assertTrue(any(r.get("event") == "velocity_exception" for r in h.rows))


if __name__ == "__main__":
    unittest.main()
