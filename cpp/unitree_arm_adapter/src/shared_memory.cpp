#include "unitree_arm_adapter/shared_memory.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <new>
#include <stdexcept>
#include <string>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace unitree_arm_adapter {
namespace {

std::runtime_error SystemError(const std::string& operation) {
    return std::runtime_error(
        operation + " failed: " + std::strerror(errno));
}

void ValidateName(const std::string& name) {
    if (name.size() < 2U || name.front() != '/' ||
        name.find('/', 1U) != std::string::npos) {
        throw std::invalid_argument(
            "POSIX shared-memory name must look like /g1_arm_mpc");
    }
}

}  // namespace

SharedMemoryRegion::~SharedMemoryRegion() { Close(); }

SharedMemoryRegion::SharedMemoryRegion(SharedMemoryRegion&& other) noexcept
    : name_(std::move(other.name_)),
      file_descriptor_(std::exchange(other.file_descriptor_, -1)),
      layout_(std::exchange(other.layout_, nullptr)) {}

SharedMemoryRegion& SharedMemoryRegion::operator=(
    SharedMemoryRegion&& other) noexcept {
    if (this != &other) {
        Close();
        name_ = std::move(other.name_);
        file_descriptor_ = std::exchange(other.file_descriptor_, -1);
        layout_ = std::exchange(other.layout_, nullptr);
    }
    return *this;
}

SharedMemoryRegion SharedMemoryRegion::Open(
    const std::string& name, bool create_if_missing) {
    ValidateName(name);
    const int flags = O_RDWR | (create_if_missing ? O_CREAT : 0);
    const int descriptor = ::shm_open(name.c_str(), flags, 0600);
    if (descriptor < 0) {
        throw SystemError("shm_open(" + name + ")");
    }

    bool locked = false;
    try {
        if (::flock(descriptor, LOCK_EX) != 0) {
            throw SystemError("flock");
        }
        locked = true;

        struct stat info {};
        if (::fstat(descriptor, &info) != 0) {
            throw SystemError("fstat");
        }
        const bool newly_created = info.st_size == 0;
        if (newly_created) {
            if (!create_if_missing) {
                throw std::runtime_error("shared-memory object is empty");
            }
            if (::ftruncate(
                    descriptor,
                    static_cast<off_t>(sizeof(SharedMemoryLayout))) != 0) {
                throw SystemError("ftruncate");
            }
        } else if (
            info.st_size != static_cast<off_t>(sizeof(SharedMemoryLayout))) {
            throw std::runtime_error(
                "shared-memory layout size mismatch; unlink it explicitly "
                "only after stopping all attached processes");
        }

        void* mapping = ::mmap(
            nullptr,
            sizeof(SharedMemoryLayout),
            PROT_READ | PROT_WRITE,
            MAP_SHARED,
            descriptor,
            0);
        if (mapping == MAP_FAILED) {
            throw SystemError("mmap");
        }
        auto* layout = static_cast<SharedMemoryLayout*>(mapping);
        if (newly_created) {
            std::memset(mapping, 0, sizeof(SharedMemoryLayout));
            new (layout) SharedMemoryLayout();
            layout->layout_size = sizeof(SharedMemoryLayout);
        } else if (
            layout->magic != kSharedMemoryMagic ||
            layout->version != kProtocolVersion ||
            layout->layout_size != sizeof(SharedMemoryLayout)) {
            ::munmap(mapping, sizeof(SharedMemoryLayout));
            throw std::runtime_error(
                "shared-memory magic/version mismatch; refusing to overwrite");
        }

        if (::flock(descriptor, LOCK_UN) != 0) {
            ::munmap(mapping, sizeof(SharedMemoryLayout));
            throw SystemError("flock unlock");
        }
        locked = false;
        return SharedMemoryRegion(name, descriptor, layout);
    } catch (...) {
        if (locked) {
            ::flock(descriptor, LOCK_UN);
        }
        ::close(descriptor);
        throw;
    }
}

void SharedMemoryRegion::Unlink(const std::string& name) {
    ValidateName(name);
    if (::shm_unlink(name.c_str()) != 0 && errno != ENOENT) {
        throw SystemError("shm_unlink(" + name + ")");
    }
}

void SharedMemoryRegion::Close() noexcept {
    if (layout_ != nullptr) {
        ::munmap(layout_, sizeof(SharedMemoryLayout));
        layout_ = nullptr;
    }
    if (file_descriptor_ >= 0) {
        ::close(file_descriptor_);
        file_descriptor_ = -1;
    }
}

}  // namespace unitree_arm_adapter
