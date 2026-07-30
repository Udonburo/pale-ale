from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import gate12c2_resource_qualification as resource


TEST_ATTEMPT_ID = resource.sha256_bytes(b"gate12c2-v07-test-attempt")


def raw_handle(value: object) -> int:
    if type(value) is int:
        return value
    inner = getattr(value, "value", None)
    if type(inner) is int:
        return inner
    return 0


class KernelFunction:
    def __init__(self, callback):
        self.callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.callback(*args)


class FakeKernel32:
    REQUIRED_FLAGS = (
        resource.WindowsJobApi.JOB_OBJECT_LIMIT_JOB_MEMORY
        | resource.WindowsJobApi.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    )

    def __init__(
        self,
        *,
        physical_ram_bytes: int = 16 * 1024**3 + 123,
        page_size_bytes: int = 4096,
        outside_job: bool = True,
        first_initialize_returns_true: bool = False,
        first_initialize_error: int = 122,
        first_attribute_bytes: int = 128,
        second_attribute_bytes: int | None = None,
        second_initialize_succeeds: bool = True,
        update_succeeds: bool = True,
        delete_raises: bool = False,
        create_process_succeeds: bool = True,
        residual_process_information_on_failure: bool = False,
        child_in_job: bool = True,
        process_id_matches: bool = True,
        thread_id_matches: bool = True,
        initial_suspend_count: int = 1,
        accounting: tuple[int, int, int] = (1, 1, 0),
        limit_flags: int | None = None,
        reported_limit: int | None = None,
        fail_close_handles: set[int] | None = None,
        noop_close_handles: set[int] | None = None,
    ) -> None:
        self.physical_ram_bytes = physical_ram_bytes
        self.page_size_bytes = page_size_bytes
        self.outside_job = outside_job
        self.first_initialize_returns_true = first_initialize_returns_true
        self.first_initialize_error = first_initialize_error
        self.first_attribute_bytes = first_attribute_bytes
        self.second_attribute_bytes = (
            first_attribute_bytes
            if second_attribute_bytes is None
            else second_attribute_bytes
        )
        self.second_initialize_succeeds = second_initialize_succeeds
        self.update_succeeds = update_succeeds
        self.delete_raises = delete_raises
        self.create_process_succeeds = create_process_succeeds
        self.residual_process_information_on_failure = (
            residual_process_information_on_failure
        )
        self.child_in_job = child_in_job
        self.process_id_matches = process_id_matches
        self.thread_id_matches = thread_id_matches
        self.suspend_count = initial_suspend_count
        self.accounting = accounting
        self.limit_flags = (
            self.REQUIRED_FLAGS if limit_flags is None else limit_flags
        )
        self.reported_limit = reported_limit
        self.configured_limit = 0
        self.configured_flags = 0
        self.fail_close_handles = set(fail_close_handles or ())
        self.noop_close_handles = set(noop_close_handles or ())

        self.job_handle = 100
        self.process_handle = 200
        self.thread_handle = 300
        self.current_process_handle = 999
        self.process_id = 1234
        self.thread_id = 5678
        self.creation_ticks = (
            resource.WINDOWS_TO_UNIX_EPOCH_100NS + 123_456_789
        )
        self.open_handles: set[int] = set()
        self.inheritable_handles: set[int] = set()
        self.calls: list[str] = []
        self.close_calls: list[int] = []
        self.initialize_calls: list[tuple[bool, int, int]] = []
        self.update_calls: list[dict[str, int]] = []
        self.delete_calls = 0
        self.create_process_calls: list[dict[str, Any]] = []

        self.GlobalMemoryStatusEx = KernelFunction(self._memory_status)
        self.GetNativeSystemInfo = KernelFunction(self._native_system)
        self.GetCurrentProcess = KernelFunction(
            lambda: self.current_process_handle
        )
        self.IsProcessInJob = KernelFunction(self._is_process_in_job)
        self.CreateJobObjectW = KernelFunction(self._create_job)
        self.SetInformationJobObject = KernelFunction(self._set_job)
        self.QueryInformationJobObject = KernelFunction(self._query_job)
        self.GetHandleInformation = KernelFunction(self._handle_information)
        self.CloseHandle = KernelFunction(self._close_handle)
        self.InitializeProcThreadAttributeList = KernelFunction(
            self._initialize_attributes
        )
        self.UpdateProcThreadAttribute = KernelFunction(
            self._update_attribute
        )
        self.DeleteProcThreadAttributeList = KernelFunction(
            self._delete_attributes
        )
        self.CreateProcessW = KernelFunction(self._create_process)
        self.GetProcessId = KernelFunction(self._get_process_id)
        self.GetThreadId = KernelFunction(self._get_thread_id)
        self.GetProcessTimes = KernelFunction(self._get_process_times)
        self.SuspendThread = KernelFunction(self._suspend_thread)
        self.ResumeThread = KernelFunction(self._resume_thread)

    def _memory_status(self, pointer):
        self.calls.append("GlobalMemoryStatusEx")
        pointer._obj.ullTotalPhys = self.physical_ram_bytes
        pointer._obj.ullAvailPhys = self.physical_ram_bytes // 2
        return True

    def _native_system(self, pointer):
        self.calls.append("GetNativeSystemInfo")
        pointer._obj.dwPageSize = self.page_size_bytes

    def _is_process_in_job(self, process_handle, job_handle, output):
        process = raw_handle(process_handle)
        job = raw_handle(job_handle)
        self.calls.append("IsProcessInJob")
        if process == self.current_process_handle and job == 0:
            output._obj.value = 0 if self.outside_job else 1
            return True
        if process == self.process_handle and job == self.job_handle:
            output._obj.value = 1 if self.child_in_job else 0
            return True
        output._obj.value = 0
        return True

    def _create_job(self, security, name):
        self.calls.append("CreateJobObjectW")
        if security is not None or name is not None:
            return 0
        self.open_handles.add(self.job_handle)
        return self.job_handle

    def _set_job(self, handle, info_class, info, size):
        self.calls.append("SetInformationJobObject")
        if raw_handle(handle) != self.job_handle:
            return False
        if info_class != resource.WindowsJobApi.JOB_OBJECT_EXTENDED_LIMIT_INFORMATION:
            return False
        if size != ctypes.sizeof(resource._JOBOBJECT_EXTENDED_LIMIT_INFORMATION):
            return False
        self.configured_flags = int(
            info._obj.BasicLimitInformation.LimitFlags
        )
        self.configured_limit = int(info._obj.JobMemoryLimit)
        return True

    def _query_job(self, handle, info_class, info, size, returned):
        self.calls.append("QueryInformationJobObject")
        if raw_handle(handle) not in self.open_handles:
            ctypes.set_last_error(6)
            return False
        if (
            info_class
            == resource.WindowsJobApi.JOB_OBJECT_EXTENDED_LIMIT_INFORMATION
        ):
            info._obj.BasicLimitInformation.LimitFlags = self.limit_flags
            info._obj.JobMemoryLimit = (
                self.configured_limit
                if self.reported_limit is None
                else self.reported_limit
            )
        elif (
            info_class
            == resource.WindowsJobApi.JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION
        ):
            active, total, terminated = self.accounting
            info._obj.ActiveProcesses = active
            info._obj.TotalProcesses = total
            info._obj.TotalTerminatedProcesses = terminated
        else:
            return False
        returned._obj.value = size
        return True

    def _handle_information(self, handle, flags):
        raw = raw_handle(handle)
        if raw not in self.open_handles:
            ctypes.set_last_error(6)
            return False
        flags._obj.value = (
            resource.WindowsJobApi.HANDLE_FLAG_INHERIT
            if raw in self.inheritable_handles
            else 0
        )
        return True

    def _close_handle(self, handle):
        raw = raw_handle(handle)
        self.close_calls.append(raw)
        self.calls.append(f"CloseHandle:{raw}")
        if raw in self.fail_close_handles:
            ctypes.set_last_error(5)
            return False
        if raw in self.noop_close_handles:
            return True
        if raw not in self.open_handles:
            ctypes.set_last_error(6)
            return False
        self.open_handles.remove(raw)
        return True

    def _initialize_attributes(self, list_pointer, count, flags, size):
        first = not bool(list_pointer)
        self.initialize_calls.append((first, int(count), int(flags)))
        self.calls.append(
            "InitializeProcThreadAttributeList:first"
            if first
            else "InitializeProcThreadAttributeList:second"
        )
        if first:
            size._obj.value = self.first_attribute_bytes
            ctypes.set_last_error(self.first_initialize_error)
            return self.first_initialize_returns_true
        size._obj.value = self.second_attribute_bytes
        return self.second_initialize_succeeds

    def _update_attribute(
        self,
        list_pointer,
        flags,
        attribute,
        value,
        value_bytes,
        previous,
        returned,
    ):
        self.calls.append("UpdateProcThreadAttribute")
        job_value = raw_handle(
            ctypes.cast(
                value, ctypes.POINTER(ctypes.wintypes.HANDLE)
            ).contents
        )
        self.update_calls.append(
            {
                "list_pointer": raw_handle(list_pointer),
                "flags": int(flags),
                "attribute": int(attribute),
                "job_value": job_value,
                "value_bytes": int(value_bytes),
                "previous_is_null": int(previous is None),
                "returned_is_null": int(returned is None),
            }
        )
        return self.update_succeeds

    def _delete_attributes(self, list_pointer):
        self.calls.append("DeleteProcThreadAttributeList")
        self.delete_calls += 1
        if not bool(list_pointer):
            raise OSError("invalid attribute pointer")
        if self.delete_raises:
            raise OSError("delete failed")

    def _create_process(
        self,
        application_name,
        command_line,
        process_security,
        thread_security,
        inherit_handles,
        flags,
        environment,
        current_directory,
        startup,
        process_info,
    ):
        self.calls.append("CreateProcessW")
        self.create_process_calls.append(
            {
                "application_name": application_name,
                "command_line": ctypes.wstring_at(command_line),
                "process_security_is_null": process_security is None,
                "thread_security_is_null": thread_security is None,
                "inherit_handles": bool(inherit_handles),
                "flags": int(flags),
                "environment_is_null": environment is None,
                "current_directory": current_directory,
                "startup_cb": int(startup._obj.StartupInfo.cb),
                "attribute_pointer": raw_handle(
                    startup._obj.lpAttributeList
                ),
            }
        )
        if self.create_process_succeeds or self.residual_process_information_on_failure:
            process_info._obj.hProcess = self.process_handle
            process_info._obj.hThread = self.thread_handle
            process_info._obj.dwProcessId = self.process_id
            process_info._obj.dwThreadId = self.thread_id
            self.open_handles.update(
                {self.process_handle, self.thread_handle}
            )
        return self.create_process_succeeds

    def _get_process_id(self, handle):
        if raw_handle(handle) != self.process_handle:
            return 0
        return self.process_id if self.process_id_matches else self.process_id + 1

    def _get_thread_id(self, handle):
        if raw_handle(handle) != self.thread_handle:
            return 0
        return self.thread_id if self.thread_id_matches else self.thread_id + 1

    def _get_process_times(
        self, handle, creation, exit_time, kernel, user
    ):
        if raw_handle(handle) != self.process_handle:
            return False
        creation._obj.dwLowDateTime = self.creation_ticks & 0xFFFFFFFF
        creation._obj.dwHighDateTime = self.creation_ticks >> 32
        exit_time._obj.dwLowDateTime = 0
        exit_time._obj.dwHighDateTime = 0
        kernel._obj.dwLowDateTime = 0
        kernel._obj.dwHighDateTime = 0
        user._obj.dwLowDateTime = 0
        user._obj.dwHighDateTime = 0
        return True

    def _suspend_thread(self, handle):
        if raw_handle(handle) != self.thread_handle:
            return resource.WindowsJobApi.INVALID_SUSPEND_COUNT
        previous = self.suspend_count
        self.suspend_count += 1
        return previous

    def _resume_thread(self, handle):
        if raw_handle(handle) != self.thread_handle or self.suspend_count <= 0:
            return resource.WindowsJobApi.INVALID_SUSPEND_COUNT
        previous = self.suspend_count
        self.suspend_count -= 1
        return previous


def job_api(kernel: FakeKernel32) -> resource.WindowsJobApi:
    api = object.__new__(resource.WindowsJobApi)
    api.kernel32 = kernel
    return api


def geometry(kernel: FakeKernel32) -> resource.ResourceMemoryGeometry:
    return resource.derive_resource_memory_geometry(
        kernel.physical_ram_bytes, kernel.page_size_bytes
    )


def make_launch(
    kernel: FakeKernel32 | None = None,
) -> tuple[
    resource.WatchdogLocalWindowsJobLaunch,
    resource.WindowsJobApi,
    FakeKernel32,
]:
    selected = kernel or FakeKernel32()
    api = job_api(selected)
    launch = api.launch_scientific_child_suspended(
        preflight_geometry=geometry(selected),
        application_name="python.exe",
        command_line="python.exe -B frozen_scientific_entry.py",
        current_directory=Path.cwd(),
    )
    return launch, api, selected


class Clock:
    def __init__(self, *values: int) -> None:
        self.values = list(values)
        self.last = values[-1] if values else 0

    def __call__(self) -> int:
        if self.values:
            self.last = self.values.pop(0)
        return self.last


def telemetry_metrics() -> dict[str, int]:
    return {
        "watchdog_pid": 101,
        "watchdog_creation_time_ns": 1_000,
        "guardian_pid": 102,
        "guardian_creation_time_ns": 1_001,
        "coordinator_pid": 103,
        "coordinator_creation_time_ns": 1_002,
        "replay_root_pid": 104,
        "replay_root_creation_time_ns": 1_003,
        "job_active_process_count": 4,
        "job_total_process_count": 4,
        "job_terminated_process_count": 0,
        "job_current_memory_bytes": 1_000_000,
        "job_peak_memory_bytes": 2_000_000,
        "sampled_replay_job_rss_bytes": 1_500_000,
        "sampled_control_plane_rss_bytes": 500_000,
        "sampled_combined_rss_bytes": 2_000_000,
        "available_physical_memory_bytes": 8_000_000_000,
        "total_physical_memory_bytes": 16_000_000_000,
        "qualification_volume_free_bytes": 20_000_000_000,
        "scheduled_output_file_count": 1,
        "scheduled_output_bytes": 1_000,
        "partial_or_temp_count": 0,
    }


def utc(index: int) -> str:
    return f"2026-07-30T00:00:{index:02d}.000000Z"


def telemetry_paths(directory: Path) -> tuple[Path, Path]:
    return (
        directory / "telemetry.jsonl.pending",
        directory / "telemetry.jsonl",
    )


def write_success_telemetry(
    directory: Path,
    *,
    periodic: bool = False,
) -> tuple[bytes, dict[str, Any]]:
    pending, final = telemetry_paths(directory)
    writer = resource.AppendOnlyTelemetryWriter(
        pending,
        final,
        attempt_identity_sha256=TEST_ATTEMPT_ID,
    )
    monotonic = 10
    for index, event in enumerate(resource.SUCCESS_MILESTONES):
        writer.append(
            event_code=event,
            utc_time=utc(index),
            monotonic_ns=monotonic,
            metrics=telemetry_metrics(),
        )
        monotonic += 1
        if periodic and event == "CHILD_RESUMED":
            writer.append(
                event_code="PERIODIC_SAMPLE",
                utc_time=utc(index),
                monotonic_ns=monotonic,
                metrics=telemetry_metrics(),
            )
            monotonic += 1
    receipt = writer.close()
    return final.read_bytes(), receipt
