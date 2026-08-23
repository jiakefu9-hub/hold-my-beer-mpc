"""Right-arm runtime boundary with lazy platform-specific exports.

Importing a lightweight shared contract must not initialize the MuJoCo
simulation adapter.  Existing package-level names remain source-compatible and
are loaded only when a caller actually requests them.
"""

from importlib import import_module

__all__ = (
    "AdapterMode",
    "AdapterStatusSnapshot",
    "CommandMode",
    "CommandWriteReceipt",
    "CppDdqMapperResult",
    "CppDdqTorqueMapper",
    "CppNoSafeTorqueError",
    "CppExecutorResult",
    "CppRightArmExecutor",
    "RightArmSimProcess",
    "RobotStateSnapshot",
    "SimProcessResult",
    "SimProcessShadowValidator",
    "SimRuntimeError",
    "UnitreeArmSharedMemoryClient",
)


_EXPORT_MODULE = {
    "CppExecutorResult": ".cpp_executor",
    "CppRightArmExecutor": ".cpp_executor",
    "CppDdqMapperResult": ".cpp_ddq_mapper",
    "CppDdqTorqueMapper": ".cpp_ddq_mapper",
    "CppNoSafeTorqueError": ".cpp_ddq_mapper",
    "RightArmSimProcess": ".sim_process",
    "SimProcessResult": ".sim_process",
    "SimProcessShadowValidator": ".sim_process",
    "SimRuntimeError": ".sim_process",
    "AdapterMode": ".unitree_shm",
    "AdapterStatusSnapshot": ".unitree_shm",
    "CommandMode": ".unitree_shm",
    "CommandWriteReceipt": ".unitree_shm",
    "RobotStateSnapshot": ".unitree_shm",
    "UnitreeArmSharedMemoryClient": ".unitree_shm",
}


def __getattr__(name: str):
    module_name = _EXPORT_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
