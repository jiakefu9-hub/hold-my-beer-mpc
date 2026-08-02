#pragma once

#include <string>
#include <utility>

#include "unitree_arm_adapter/protocol.hpp"

namespace unitree_arm_adapter {

class SharedMemoryRegion {
public:
    SharedMemoryRegion() = default;
    ~SharedMemoryRegion();

    SharedMemoryRegion(const SharedMemoryRegion&) = delete;
    SharedMemoryRegion& operator=(const SharedMemoryRegion&) = delete;
    SharedMemoryRegion(SharedMemoryRegion&& other) noexcept;
    SharedMemoryRegion& operator=(SharedMemoryRegion&& other) noexcept;

    // create_if_missing只负责首次创建；已有但版本不匹配时直接失败，绝不覆盖。
    static SharedMemoryRegion Open(
        const std::string& name, bool create_if_missing);
    static void Unlink(const std::string& name);

    [[nodiscard]] SharedMemoryLayout* get() noexcept { return layout_; }
    [[nodiscard]] const SharedMemoryLayout* get() const noexcept {
        return layout_;
    }
    [[nodiscard]] const std::string& name() const noexcept { return name_; }

private:
    SharedMemoryRegion(
        std::string name, int file_descriptor, SharedMemoryLayout* layout)
        : name_(std::move(name)),
          file_descriptor_(file_descriptor),
          layout_(layout) {}

    void Close() noexcept;

    std::string name_;
    int file_descriptor_{-1};
    SharedMemoryLayout* layout_{nullptr};
};

}  // namespace unitree_arm_adapter
