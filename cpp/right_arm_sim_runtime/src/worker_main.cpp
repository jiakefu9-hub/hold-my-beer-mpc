#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>

#include "right_arm_sim_runtime/protocol.hpp"
#include "right_arm_sim_runtime/runtime.hpp"
#include "right_arm_sim_runtime/shared_memory.hpp"

namespace rsr = right_arm_sim_runtime;

namespace {

struct Options {
    std::string scene_path;
    std::string shared_memory_name{"/g1_right_arm_sim_runtime"};
    int request_fd{-1};
    int response_fd{-1};
    bool print_layout{false};
    bool reset_shared_memory{false};
    bool unlink_on_exit{false};
};

int ParseFd(const std::string& text, const char* name) {
    std::size_t parsed = 0;
    const long value = std::stol(text, &parsed);
    if (parsed != text.size() || value < 0 || value > 1'000'000L) {
        throw std::invalid_argument(std::string(name) + " is not a valid fd");
    }
    return static_cast<int>(value);
}

void PrintUsage(const char* executable) {
    std::cout
        << "用法: " << executable << " --scene FILE --request-fd FD "
        << "--response-fd FD [选项]\n"
        << "  --shm-name NAME    POSIX共享内存名\n"
        << "  --reset-shm        启动前删除同名共享内存\n"
        << "  --unlink-on-exit   正常退出后删除共享内存名字\n"
        << "  --print-layout     只输出固定ABI布局\n"
        << "\nexternal-step模式：request-fd每收到1字节，处理一个request；\n"
        << "response写入后在response-fd写1字节。该程序不访问DDS。\n";
}

Options ParseOptions(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        const auto value = [&](const char* name) {
            if (++index >= argc) {
                throw std::invalid_argument(std::string(name) + " needs a value");
            }
            return std::string(argv[index]);
        };
        if (argument == "--scene") {
            options.scene_path = value("--scene");
        } else if (argument == "--shm-name") {
            options.shared_memory_name = value("--shm-name");
        } else if (argument == "--request-fd") {
            options.request_fd = ParseFd(value("--request-fd"), "--request-fd");
        } else if (argument == "--response-fd") {
            options.response_fd = ParseFd(value("--response-fd"), "--response-fd");
        } else if (argument == "--print-layout") {
            options.print_layout = true;
        } else if (argument == "--reset-shm") {
            options.reset_shared_memory = true;
        } else if (argument == "--unlink-on-exit") {
            options.unlink_on_exit = true;
        } else if (argument == "--help" || argument == "-h") {
            PrintUsage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (!options.print_layout &&
        (options.scene_path.empty() || options.request_fd < 0 ||
         options.response_fd < 0)) {
        throw std::invalid_argument(
            "--scene, --request-fd and --response-fd are required");
    }
    return options;
}

void PrintLayout() {
    std::cout
        << "protocol_version=" << rsr::kProtocolVersion << '\n'
        << "layout_size=" << sizeof(rsr::SharedMemoryLayout) << '\n'
        << "request_offset="
        << offsetof(rsr::SharedMemoryLayout, request) << '\n'
        << "request_payload_size="
        << sizeof(rsr::SimulationRequestPayload) << '\n'
        << "response_offset="
        << offsetof(rsr::SharedMemoryLayout, response) << '\n'
        << "response_payload_size="
        << sizeof(rsr::SimulationResponsePayload) << '\n'
        << "request.qpos_offset="
        << offsetof(rsr::SimulationRequestPayload, qpos) << '\n'
        << "request.qvel_offset="
        << offsetof(rsr::SimulationRequestPayload, qvel) << '\n'
        << "request.reference_qacc_offset="
        << offsetof(rsr::SimulationRequestPayload, reference_qacc) << '\n'
        << "request.fixed_ctrl_offset="
        << offsetof(rsr::SimulationRequestPayload, fixed_ctrl) << '\n'
        << "request.xfrc_applied_offset="
        << offsetof(rsr::SimulationRequestPayload, xfrc_applied) << '\n'
        << "request.executor_config_offset="
        << offsetof(rsr::SimulationRequestPayload, executor_config) << '\n'
        << "response.rnea_output_offset="
        << offsetof(rsr::SimulationResponsePayload, rnea_output) << '\n'
        << "response.mapper_output_offset="
        << offsetof(rsr::SimulationResponsePayload, mapper_output) << '\n'
        << "response.executor_output_offset="
        << offsetof(rsr::SimulationResponsePayload, executor_output) << '\n'
        << "response.final_tau_offset="
        << offsetof(rsr::SimulationResponsePayload, final_tau) << '\n';
}

bool ReadNotification(int descriptor) {
    unsigned char byte = 0;
    while (true) {
        const ssize_t result = ::read(descriptor, &byte, 1);
        if (result == 1) {
            return true;
        }
        if (result == 0) {
            return false;
        }
        if (errno != EINTR) {
            throw std::runtime_error(
                std::string("request pipe read: ") + std::strerror(errno));
        }
    }
}

void WriteNotification(int descriptor) {
    const unsigned char byte = 1;
    while (true) {
        const ssize_t result = ::write(descriptor, &byte, 1);
        if (result == 1) {
            return;
        }
        if (result < 0 && errno == EINTR) {
            continue;
        }
        throw std::runtime_error(
            std::string("response pipe write: ") + std::strerror(errno));
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = ParseOptions(argc, argv);
        if (options.print_layout) {
            PrintLayout();
            return 0;
        }
        if (options.reset_shared_memory) {
            rsr::SharedMemoryRegion::Unlink(options.shared_memory_name);
        }
        auto region = rsr::SharedMemoryRegion::Open(
            options.shared_memory_name, true);
        rsr::SimulationRuntime runtime(options.scene_path);
        const auto& dimensions = runtime.dimensions();
        std::cout << "right_arm_sim_runtime external-step ready: nq="
                  << dimensions.nq << ", nv=" << dimensions.nv
                  << ", nu=" << dimensions.nu
                  << ", nbody=" << dimensions.nbody << "\n";
        std::cout.flush();

        bool shutdown = false;
        while (!shutdown && ReadNotification(options.request_fd)) {
            rsr::SimulationRequestPayload request;
            rsr::SimulationResponsePayload response;
            if (!rsr::ReadSeqlock(region.get()->request, request)) {
                response.worker_start_monotonic_ns = rsr::MonotonicNowNs();
                response.worker_finish_monotonic_ns =
                    response.worker_start_monotonic_ns;
                response.status = static_cast<std::uint32_t>(
                    rsr::RuntimeStatus::kInvalidRequest);
                std::strncpy(
                    response.error,
                    "unable to read a stable request snapshot",
                    rsr::kErrorCapacity - 1U);
            } else {
                const bool processed = runtime.Process(request, response);
                (void)processed;
                shutdown = response.status == static_cast<std::uint32_t>(
                    rsr::RuntimeStatus::kShutdown);
            }
            rsr::WriteSeqlock(region.get()->response, response);
            WriteNotification(options.response_fd);
        }
        if (options.unlink_on_exit) {
            rsr::SharedMemoryRegion::Unlink(options.shared_memory_name);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "right_arm_sim_runtime_worker: " << error.what() << '\n';
        return 1;
    }
}
