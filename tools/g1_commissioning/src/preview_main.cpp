#include "g1_commissioning/core.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace gc = g1_commissioning;

namespace {

struct Options {
    std::string profile_path;
    std::string state_path;
    std::string output_path;
};

void Usage(const char* executable) {
    std::cout
        << "Usage: " << executable
        << " --profile FILE --state FILE --output JSONL\n\n"
        << "Offline only: no Unitree SDK, DDS, RPC, or publisher is linked.\n";
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
        if (argument == "--profile") {
            options.profile_path = value("--profile");
        } else if (argument == "--state") {
            options.state_path = value("--state");
        } else if (argument == "--output") {
            options.output_path = value("--output");
        } else if (argument == "--help" || argument == "-h") {
            Usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (options.profile_path.empty() || options.state_path.empty() ||
        options.output_path.empty()) {
        throw std::invalid_argument("--profile, --state and --output are required");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = ParseOptions(argc, argv);
        const gc::SiteProfile profile = gc::LoadSiteProfile(options.profile_path);
        const gc::OfflineSnapshot snapshot =
            gc::LoadOfflineSnapshot(options.state_path);
        const auto preview_validation = gc::ValidateProfile(
            profile, gc::ValidationUse::kPreview);
        if (!preview_validation.ok()) {
            for (const auto& error : preview_validation.errors) {
                std::cerr << "profile: " << error << '\n';
            }
            return 2;
        }
        const auto state_validation = gc::ValidateState(
            snapshot.state, profile, snapshot.validation_now_monotonic_ns,
            true, false);
        if (!state_validation.ok()) {
            for (const auto& error : state_validation.errors) {
                std::cerr << "state: " << error << '\n';
            }
            return 2;
        }
        if (std::filesystem::exists(options.output_path)) {
            throw std::runtime_error("refusing to overwrite output: " +
                                     options.output_path);
        }
        std::ofstream output(options.output_path);
        if (!output) {
            throw std::runtime_error("cannot create output: " + options.output_path);
        }

        const auto real_validation = gc::ValidateProfile(
            profile, gc::ValidationUse::kRealOutput);
        output << gc::ProfileSummaryJson(profile, real_validation) << '\n';
        gc::TrajectoryPlanner planner(profile, snapshot.state);
        const double period_s = profile.control_period_ms / 1000.0;
        std::uint64_t sequence = 1;
        for (double elapsed = 0.0;
             elapsed < planner.total_duration_s(); elapsed += period_s) {
            auto frame = planner.Sample(elapsed);
            frame.sequence = sequence++;
            output << gc::FrameJson(
                frame, &snapshot.state, "offline_would_write", "synthetic_preview")
                   << '\n';
        }
        auto terminal = planner.Sample(planner.total_duration_s());
        terminal.sequence = sequence;
        output << gc::FrameJson(
            terminal, &snapshot.state, "offline_would_write", "synthetic_preview")
               << '\n';
        std::cout << "offline_preview_completed=true\n"
                  << "real_output_profile_gate_passed="
                  << (real_validation.ok() ? "true" : "false") << '\n'
                  << "planned_duration_s=" << planner.total_duration_s() << '\n'
                  << "records=" << sequence << '\n'
                  << "output=" << options.output_path << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "preview failed: " << error.what() << '\n';
        return 1;
    }
}
