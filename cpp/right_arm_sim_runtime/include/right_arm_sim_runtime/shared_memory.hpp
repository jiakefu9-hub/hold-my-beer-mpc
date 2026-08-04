#pragma once

#include <string>
#include <utility>

#include "right_arm_sim_runtime/protocol.hpp"

namespace right_arm_sim_runtime {

class SharedMemoryRegion {
public:
    SharedMemoryRegion() = default;
    ~SharedMemoryRegion();

    SharedMemoryRegion(const SharedMemoryRegion&) = delete;
    SharedMemoryRegion& operator=(const SharedMemoryRegion&) = delete;
    SharedMemoryRegion(SharedMemoryRegion&& other) noexcept;
    SharedMemoryRegion& operator=(SharedMemoryRegion&& other) noexcept;

    static SharedMemoryRegion Open(
        const std::string& name, bool create_if_missing);
    static void Unlink(const std::string& name);

    [[nodiscard]] SharedMemoryLayout* get() noexcept { return layout_; }
    [[nodiscard]] const SharedMemoryLayout* get() const noexcept {
        return layout_;
    }

private:
    SharedMemoryRegion(
        std::string name, int descriptor, SharedMemoryLayout* layout)
        : name_(std::move(name)), descriptor_(descriptor), layout_(layout) {}
    void Close() noexcept;

    std::string name_;
    int descriptor_{-1};
    SharedMemoryLayout* layout_{nullptr};
};

}  // namespace right_arm_sim_runtime
