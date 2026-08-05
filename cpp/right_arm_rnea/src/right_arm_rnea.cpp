#include "right_arm_rnea/right_arm_rnea_c.h"

#include <mujoco/mujoco.h>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/parsers/mjcf.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Nanoseconds = std::chrono::nanoseconds;

constexpr uint32_t kAbiVersion = 3;
constexpr std::array<const char*, RIGHT_ARM_RNEA_JOINT_COUNT> kRightArmJointNames = {
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
};

struct MjModelDeleter {
    void operator()(mjModel* model) const noexcept {
        if (model != nullptr) {
            mj_deleteModel(model);
        }
    }
};

using UniqueMjModel = std::unique_ptr<mjModel, MjModelDeleter>;

struct ScalarVelocityMapping {
    Eigen::Index pin_index{0};
    int mujoco_index{0};
};

void WriteError(char* destination, size_t capacity, const std::string& message) noexcept {
    if (destination == nullptr || capacity == 0) {
        return;
    }
    const size_t count = std::min(capacity - 1, message.size());
    std::memcpy(destination, message.data(), count);
    destination[count] = '\0';
}

bool IsFiniteArray(const double* values, size_t count) noexcept {
    if (values == nullptr) {
        return false;
    }
    for (size_t index = 0; index < count; ++index) {
        if (!std::isfinite(values[index])) {
            return false;
        }
    }
    return true;
}

double Sign(double value) noexcept {
    return static_cast<double>((value > 0.0) - (value < 0.0));
}

std::filesystem::path ResolveRobotMjcf(const std::filesystem::path& scene_path) {
    std::ifstream stream(scene_path);
    if (!stream) {
        throw std::runtime_error("无法读取 MJCF: " + scene_path.string());
    }
    const std::string xml(
        (std::istreambuf_iterator<char>(stream)),
        std::istreambuf_iterator<char>());
    const std::regex include_pattern(
        R"(<include\s+[^>]*file\s*=\s*[\"']([^\"']+)[\"'][^>]*/?>)");
    std::sregex_iterator begin(xml.begin(), xml.end(), include_pattern);
    const std::sregex_iterator end;
    if (begin == end) {
        return scene_path;
    }
    const std::filesystem::path included =
        scene_path.parent_path() / (*begin)[1].str();
    ++begin;
    if (begin != end) {
        throw std::runtime_error("scene MJCF 含多个 include，无法唯一确定机器人模型");
    }
    if (!std::filesystem::is_regular_file(included)) {
        throw std::runtime_error("MJCF include 不存在: " + included.string());
    }
    return std::filesystem::canonical(included);
}

Eigen::Matrix3d MuJoCoQuaternionToRotation(const double* qpos, int free_q_index) {
    // MuJoCo free joint: xyz + wxyz；这里保持与 Python 参考实现逐项相同，
    // 不在实时路径重新归一化有效四元数。
    const double w = qpos[free_q_index + 3];
    const double x = qpos[free_q_index + 4];
    const double y = qpos[free_q_index + 5];
    const double z = qpos[free_q_index + 6];
    Eigen::Matrix3d rotation;
    rotation <<
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - z * w),
        2.0 * (x * z + y * w),
        2.0 * (x * y + z * w),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - x * w),
        2.0 * (x * z - y * w),
        2.0 * (y * z + x * w),
        1.0 - 2.0 * (x * x + y * y);
    return rotation;
}

class RightArmRneaCore {
public:
    explicit RightArmRneaCore(const std::filesystem::path& scene_path) {
        const std::filesystem::path canonical_scene =
            std::filesystem::canonical(scene_path);
        char load_error[1024] = {};
        mujoco_model_.reset(mj_loadXML(
            canonical_scene.c_str(), nullptr, load_error, sizeof(load_error)));
        if (!mujoco_model_) {
            throw std::runtime_error(
                std::string("MuJoCo 加载 scene 失败: ") + load_error);
        }

        const std::filesystem::path robot_mjcf = ResolveRobotMjcf(canonical_scene);
        // 【核心初始化】Pinocchio 与 Python 后端读取同一个机器人 MJCF。
        pinocchio::mjcf::buildModel(robot_mjcf.string(), pinocchio_model_, false);
        if (pinocchio_model_.nq != mujoco_model_->nq
            || pinocchio_model_.nv != mujoco_model_->nv) {
            throw std::runtime_error(
                "Pinocchio/MuJoCo nq/nv 不一致: "
                + std::to_string(pinocchio_model_.nq) + "/"
                + std::to_string(pinocchio_model_.nv) + " vs "
                + std::to_string(mujoco_model_->nq) + "/"
                + std::to_string(mujoco_model_->nv));
        }
        pinocchio_data_ = std::make_unique<pinocchio::Data>(pinocchio_model_);

        q_pin_.resize(pinocchio_model_.nq);
        v_pin_.setZero(pinocchio_model_.nv);
        a_pin_.setZero(pinocchio_model_.nv);
        jacobian_.setZero(6, pinocchio_model_.nv);
        jacobian_dot_.setZero(6, pinocchio_model_.nv);
        mujoco_q_for_pin_q_.assign(
            static_cast<size_t>(pinocchio_model_.nq), -1);
        BuildMappings();
        ee_frame_id_ = ResolveFrame("right_grasp_site");
        imu_frame_id_ = ResolveFrame("imu_in_torso");
    }

    size_t nq() const noexcept {
        return static_cast<size_t>(mujoco_model_->nq);
    }

    size_t nv() const noexcept {
        return static_cast<size_t>(mujoco_model_->nv);
    }

    RightArmRneaOutput Compute(
        const double* qpos,
        const double* qvel,
        const double* reference_qacc,
        const double* desired_ddq,
        const double* passive,
        const double* friction_loss,
        double timestep,
        double breakaway_steps) {
        RightArmRneaOutput output{};
        const auto core_start = Clock::now();

        // 【核心映射】先按预计算索引把 MuJoCo xyz+wxyz 转成 Pin xyz+xyzw。
        for (Eigen::Index pin_q = 0; pin_q < pinocchio_model_.nq; ++pin_q) {
            q_pin_[pin_q] = qpos[mujoco_q_for_pin_q_[static_cast<size_t>(pin_q)]];
        }
        v_pin_.setZero();
        for (const ScalarVelocityMapping& mapping : scalar_velocity_mappings_) {
            v_pin_[mapping.pin_index] = qvel[mapping.mujoco_index];
        }

        const Eigen::Matrix3d rotation_world_base =
            MuJoCoQuaternionToRotation(qpos, mujoco_free_q_index_);
        // MuJoCo free-joint 平动速度是世界系；Pinocchio free-flyer 是 body 系。
        v_pin_.segment<3>(pinocchio_free_v_index_) =
            rotation_world_base.transpose()
            * Eigen::Map<const Eigen::Vector3d>(qvel + mujoco_free_v_index_);
        // 两者的 free-joint 角速度均按 body 系表达，直接复制。
        v_pin_.segment<3>(pinocchio_free_v_index_ + 3) =
            Eigen::Map<const Eigen::Vector3d>(qvel + mujoco_free_v_index_ + 3);

        a_pin_.setZero();
        for (const ScalarVelocityMapping& mapping : scalar_velocity_mappings_) {
            a_pin_[mapping.pin_index] = reference_qacc[mapping.mujoco_index];
        }
        const Eigen::Vector3d base_linear_velocity =
            v_pin_.segment<3>(pinocchio_free_v_index_);
        const Eigen::Vector3d base_angular_velocity =
            v_pin_.segment<3>(pinocchio_free_v_index_ + 3);
        // MuJoCo free-joint 平动加速度是世界系导数；Pinocchio 使用 body
        // 空间加速度，因此需要旋转并减去 omega x v 的坐标导数项。
        a_pin_.segment<3>(pinocchio_free_v_index_) =
            rotation_world_base.transpose()
                * Eigen::Map<const Eigen::Vector3d>(
                    reference_qacc + mujoco_free_v_index_)
            - base_angular_velocity.cross(base_linear_velocity);
        a_pin_.segment<3>(pinocchio_free_v_index_ + 3) =
            Eigen::Map<const Eigen::Vector3d>(
                reference_qacc + mujoco_free_v_index_ + 3);
        for (size_t joint = 0; joint < RIGHT_ARM_RNEA_JOINT_COUNT; ++joint) {
            a_pin_[pinocchio_arm_v_indices_[joint]] = desired_ddq[joint];
        }

        const auto rnea_start = Clock::now();
        const Eigen::VectorXd& full_tau = pinocchio::rnea(
            pinocchio_model_, *pinocchio_data_, q_pin_, v_pin_, a_pin_);
        const auto rnea_end = Clock::now();
        for (size_t joint = 0; joint < RIGHT_ARM_RNEA_JOINT_COUNT; ++joint) {
            output.tau_rnea[joint] = full_tau[pinocchio_arm_v_indices_[joint]];
        }
        const auto core_end = Clock::now();
        output.rnea_elapsed_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<Nanoseconds>(rnea_end - rnea_start).count());
        output.core_elapsed_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<Nanoseconds>(core_end - core_start).count());

        // 【半核心】与 Python 名义前馈完全相同的 passive/friction 后处理。
        for (size_t joint = 0; joint < RIGHT_ARM_RNEA_JOINT_COUNT; ++joint) {
            const double breakaway_velocity =
                breakaway_steps * timestep * std::abs(desired_ddq[joint]);
            const double arm_velocity = qvel[mujoco_arm_v_indices_[joint]];
            const double direction =
                std::abs(arm_velocity) < breakaway_velocity
                    ? Sign(desired_ddq[joint])
                    : Sign(arm_velocity);
            output.tau_constraint_friction[joint] =
                -friction_loss[joint] * direction;
            output.tau_ff[joint] =
                output.tau_rnea[joint]
                - passive[joint]
                - output.tau_constraint_friction[joint];
        }
        return output;
    }

    RightArmKinematicsBatchOutput ComputeKinematicsBatch(
        const double* qpos_reference,
        const double* q_arm,
        const double* dq_arm,
        const uint8_t* acceleration_required,
        size_t state_count) {
        RightArmKinematicsBatchOutput output{};
        output.state_count = static_cast<int32_t>(state_count);
        const auto core_start = Clock::now();

        // 【核心批处理】冻结的整机 qpos 只映射一次；随后每个节点只覆盖
        // 5 个右臂 q，并在同一个 Pinocchio Data 中顺序更新运动学。
        for (Eigen::Index pin_q = 0; pin_q < pinocchio_model_.nq; ++pin_q) {
            q_pin_[pin_q] =
                qpos_reference[mujoco_q_for_pin_q_[static_cast<size_t>(pin_q)]];
        }
        for (size_t state = 0; state < state_count; ++state) {
            for (size_t joint = 0; joint < RIGHT_ARM_RNEA_JOINT_COUNT; ++joint) {
                q_pin_[pinocchio_arm_q_indices_[joint]] =
                    q_arm[state * RIGHT_ARM_RNEA_JOINT_COUNT + joint];
            }
            v_pin_.setZero();
            for (size_t joint = 0; joint < RIGHT_ARM_RNEA_JOINT_COUNT; ++joint) {
                v_pin_[pinocchio_arm_v_indices_[joint]] =
                    dq_arm[state * RIGHT_ARM_RNEA_JOINT_COUNT + joint];
            }

            const bool need_acceleration = acceleration_required[state] != 0;
            if (need_acceleration) {
                pinocchio::computeJointJacobiansTimeVariation(
                    pinocchio_model_, *pinocchio_data_, q_pin_, v_pin_);
            } else {
                pinocchio::computeJointJacobians(
                    pinocchio_model_, *pinocchio_data_, q_pin_);
            }
            pinocchio::updateFramePlacements(
                pinocchio_model_, *pinocchio_data_);
            jacobian_.setZero();
            pinocchio::getFrameJacobian(
                pinocchio_model_,
                *pinocchio_data_,
                ee_frame_id_,
                pinocchio::LOCAL_WORLD_ALIGNED,
                jacobian_);
            jacobian_dot_.setZero();
            if (need_acceleration) {
                pinocchio::getFrameJacobianTimeVariation(
                    pinocchio_model_,
                    *pinocchio_data_,
                    ee_frame_id_,
                    pinocchio::LOCAL_WORLD_ALIGNED,
                    jacobian_dot_);
            }

            const auto& ee = pinocchio_data_->oMf[ee_frame_id_];
            const auto& imu = pinocchio_data_->oMf[imu_frame_id_];
            StoreVector3(ee.translation(), output.ee_position_world, state);
            StoreRotation(ee.rotation(), output.ee_rotation_world, state);
            StoreVector3(imu.translation(), output.imu_position_world, state);
            StoreRotation(imu.rotation(), output.imu_rotation_world, state);
            StoreArmJacobian(
                jacobian_, 0, output.J_v_world, state);
            StoreArmJacobian(
                jacobian_, 3, output.J_w_world, state);
            StoreArmJacobian(
                jacobian_dot_, 0, output.dJ_v_world, state);
            StoreArmJacobian(
                jacobian_dot_, 3, output.dJ_w_world, state);
        }
        output.core_elapsed_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<Nanoseconds>(Clock::now() - core_start)
                .count());
        return output;
    }

private:
    pinocchio::FrameIndex ResolveFrame(const char* name) const {
        const pinocchio::FrameIndex frame_id = pinocchio_model_.getFrameId(name);
        if (frame_id >= pinocchio_model_.frames.size()
            || pinocchio_model_.frames[frame_id].name != name) {
            throw std::runtime_error(std::string("Pinocchio 模型缺少 frame: ") + name);
        }
        return frame_id;
    }

    static void StoreVector3(
        const Eigen::Vector3d& value, double* output, size_t state) {
        for (size_t row = 0; row < 3; ++row) {
            output[state * 3 + row] = value[static_cast<Eigen::Index>(row)];
        }
    }

    static void StoreRotation(
        const Eigen::Matrix3d& value, double* output, size_t state) {
        for (size_t row = 0; row < 3; ++row) {
            for (size_t col = 0; col < 3; ++col) {
                output[state * 9 + row * 3 + col] = value(
                    static_cast<Eigen::Index>(row),
                    static_cast<Eigen::Index>(col));
            }
        }
    }

    void StoreArmJacobian(
        const pinocchio::Data::Matrix6x& value,
        Eigen::Index row_offset,
        double* output,
        size_t state) const {
        for (size_t row = 0; row < 3; ++row) {
            for (size_t joint = 0; joint < RIGHT_ARM_RNEA_JOINT_COUNT; ++joint) {
                output[
                    state * 3 * RIGHT_ARM_RNEA_JOINT_COUNT
                    + row * RIGHT_ARM_RNEA_JOINT_COUNT + joint] = value(
                        row_offset + static_cast<Eigen::Index>(row),
                        pinocchio_arm_v_indices_[joint]);
            }
        }
    }

    void BuildMappings() {
        bool free_joint_found = false;
        for (pinocchio::JointIndex pin_id = 1;
             pin_id < static_cast<pinocchio::JointIndex>(pinocchio_model_.njoints);
             ++pin_id) {
            const std::string& name = pinocchio_model_.names[pin_id];
            const int mujoco_id = mj_name2id(
                mujoco_model_.get(), mjOBJ_JOINT, name.c_str());
            if (mujoco_id < 0) {
                throw std::runtime_error(
                    "MuJoCo 模型缺少 Pinocchio 关节: " + name);
            }
            const auto& pin_joint = pinocchio_model_.joints[pin_id];
            const int pin_nq = pin_joint.nq();
            const int pin_nv = pin_joint.nv();
            const Eigen::Index pin_q = pin_joint.idx_q();
            const Eigen::Index pin_v = pin_joint.idx_v();
            const int mujoco_q = mujoco_model_->jnt_qposadr[mujoco_id];
            const int mujoco_v = mujoco_model_->jnt_dofadr[mujoco_id];
            const int mujoco_type = mujoco_model_->jnt_type[mujoco_id];

            if (pin_nq == 1 && pin_nv == 1) {
                mujoco_q_for_pin_q_[static_cast<size_t>(pin_q)] = mujoco_q;
                scalar_velocity_mappings_.push_back({pin_v, mujoco_v});
            } else if (
                pin_nq == 7 && pin_nv == 6 && mujoco_type == mjJNT_FREE) {
                if (free_joint_found) {
                    throw std::runtime_error("当前后端只支持一个 floating base");
                }
                free_joint_found = true;
                // Pin q: xyz+xyzw；MuJoCo q: xyz+wxyz。
                const std::array<int, 7> mapping = {
                    mujoco_q,
                    mujoco_q + 1,
                    mujoco_q + 2,
                    mujoco_q + 4,
                    mujoco_q + 5,
                    mujoco_q + 6,
                    mujoco_q + 3,
                };
                for (size_t offset = 0; offset < mapping.size(); ++offset) {
                    mujoco_q_for_pin_q_[static_cast<size_t>(pin_q) + offset] =
                        mapping[offset];
                }
                pinocchio_free_v_index_ = pin_v;
                mujoco_free_q_index_ = mujoco_q;
                mujoco_free_v_index_ = mujoco_v;
            } else {
                throw std::runtime_error(
                    "暂不支持关节 " + name + " 的 nq/nv="
                    + std::to_string(pin_nq) + "/" + std::to_string(pin_nv));
            }
        }
        if (!free_joint_found) {
            throw std::runtime_error("模型缺少 floating base");
        }
        for (const int index : mujoco_q_for_pin_q_) {
            if (index < 0) {
                throw std::runtime_error("Pinocchio/MuJoCo qpos 映射不完整");
            }
        }

        for (size_t joint = 0; joint < kRightArmJointNames.size(); ++joint) {
            const int mujoco_id = mj_name2id(
                mujoco_model_.get(), mjOBJ_JOINT, kRightArmJointNames[joint]);
            const pinocchio::JointIndex pin_id =
                pinocchio_model_.getJointId(kRightArmJointNames[joint]);
            if (mujoco_id < 0 || pin_id == 0
                || pinocchio_model_.names[pin_id] != kRightArmJointNames[joint]) {
                throw std::runtime_error(
                    std::string("右臂关节映射失败: ") + kRightArmJointNames[joint]);
            }
            const auto& pin_joint = pinocchio_model_.joints[pin_id];
            if (pin_joint.nq() != 1 || pin_joint.nv() != 1) {
                throw std::runtime_error(
                    std::string("右臂关节不是单自由度: ")
                    + kRightArmJointNames[joint]);
            }
            pinocchio_arm_q_indices_[joint] = pin_joint.idx_q();
            pinocchio_arm_v_indices_[joint] = pin_joint.idx_v();
            mujoco_arm_v_indices_[joint] = mujoco_model_->jnt_dofadr[mujoco_id];
        }
    }

    UniqueMjModel mujoco_model_;
    pinocchio::Model pinocchio_model_;
    std::unique_ptr<pinocchio::Data> pinocchio_data_;
    Eigen::VectorXd q_pin_;
    Eigen::VectorXd v_pin_;
    Eigen::VectorXd a_pin_;
    pinocchio::Data::Matrix6x jacobian_;
    pinocchio::Data::Matrix6x jacobian_dot_;
    std::vector<int> mujoco_q_for_pin_q_;
    std::vector<ScalarVelocityMapping> scalar_velocity_mappings_;
    Eigen::Index pinocchio_free_v_index_{0};
    int mujoco_free_q_index_{0};
    int mujoco_free_v_index_{0};
    std::array<Eigen::Index, RIGHT_ARM_RNEA_JOINT_COUNT>
        pinocchio_arm_q_indices_{};
    std::array<Eigen::Index, RIGHT_ARM_RNEA_JOINT_COUNT>
        pinocchio_arm_v_indices_{};
    std::array<int, RIGHT_ARM_RNEA_JOINT_COUNT> mujoco_arm_v_indices_{};
    pinocchio::FrameIndex ee_frame_id_{0};
    pinocchio::FrameIndex imu_frame_id_{0};
};

}  // namespace

struct RightArmRneaHandle {
    explicit RightArmRneaHandle(const std::filesystem::path& path) : core(path) {}
    RightArmRneaCore core;
};

extern "C" {

RightArmRneaHandle* right_arm_rnea_create(
    const char* scene_mjcf_path,
    char* error_message,
    size_t error_capacity) {
    WriteError(error_message, error_capacity, "");
    if (scene_mjcf_path == nullptr || scene_mjcf_path[0] == '\0') {
        WriteError(error_message, error_capacity, "scene_mjcf_path 不能为空");
        return nullptr;
    }
    try {
        return new RightArmRneaHandle(std::filesystem::path(scene_mjcf_path));
    } catch (const std::exception& error) {
        WriteError(error_message, error_capacity, error.what());
        return nullptr;
    } catch (...) {
        WriteError(error_message, error_capacity, "创建 RNEA 后端时发生未知异常");
        return nullptr;
    }
}

void right_arm_rnea_destroy(RightArmRneaHandle* handle) {
    delete handle;
}

size_t right_arm_rnea_mujoco_nq(const RightArmRneaHandle* handle) {
    return handle == nullptr ? 0 : handle->core.nq();
}

size_t right_arm_rnea_mujoco_nv(const RightArmRneaHandle* handle) {
    return handle == nullptr ? 0 : handle->core.nv();
}

RightArmRneaStatus right_arm_rnea_compute(
    RightArmRneaHandle* handle,
    const double* mujoco_qpos,
    size_t qpos_count,
    const double* mujoco_qvel,
    size_t qvel_count,
    const double* mujoco_reference_qacc,
    size_t reference_qacc_count,
    const double* desired_right_arm_ddq,
    size_t ddq_count,
    const double* tau_passive,
    size_t passive_count,
    const double* friction_loss,
    size_t friction_count,
    double mujoco_timestep,
    double friction_breakaway_steps,
    RightArmRneaOutput* output,
    char* error_message,
    size_t error_capacity) {
    WriteError(error_message, error_capacity, "");
    if (handle == nullptr || output == nullptr || mujoco_qpos == nullptr
        || mujoco_qvel == nullptr || mujoco_reference_qacc == nullptr
        || desired_right_arm_ddq == nullptr
        || tau_passive == nullptr || friction_loss == nullptr) {
        WriteError(error_message, error_capacity, "C ABI 收到空指针");
        return RIGHT_ARM_RNEA_INVALID_ARGUMENT;
    }
    *output = RightArmRneaOutput{};
    if (qpos_count != handle->core.nq() || qvel_count != handle->core.nv()
        || reference_qacc_count != handle->core.nv()
        || ddq_count != RIGHT_ARM_RNEA_JOINT_COUNT
        || passive_count != RIGHT_ARM_RNEA_JOINT_COUNT
        || friction_count != RIGHT_ARM_RNEA_JOINT_COUNT) {
        WriteError(error_message, error_capacity, "RNEA 输入维度与模型不一致");
        return RIGHT_ARM_RNEA_DIMENSION_MISMATCH;
    }
    if (!IsFiniteArray(mujoco_qpos, qpos_count)
        || !IsFiniteArray(mujoco_qvel, qvel_count)
        || !IsFiniteArray(mujoco_reference_qacc, reference_qacc_count)
        || !IsFiniteArray(desired_right_arm_ddq, ddq_count)
        || !IsFiniteArray(tau_passive, passive_count)
        || !IsFiniteArray(friction_loss, friction_count)
        || !std::isfinite(mujoco_timestep) || mujoco_timestep <= 0.0
        || !std::isfinite(friction_breakaway_steps)
        || friction_breakaway_steps < 0.0) {
        WriteError(error_message, error_capacity, "RNEA 输入含非有限值或非法参数");
        return RIGHT_ARM_RNEA_NONFINITE_INPUT;
    }
    for (size_t joint = 0; joint < friction_count; ++joint) {
        if (friction_loss[joint] < 0.0) {
            WriteError(error_message, error_capacity, "friction_loss 不能为负数");
            return RIGHT_ARM_RNEA_INVALID_ARGUMENT;
        }
    }
    try {
        *output = handle->core.Compute(
            mujoco_qpos,
            mujoco_qvel,
            mujoco_reference_qacc,
            desired_right_arm_ddq,
            tau_passive,
            friction_loss,
            mujoco_timestep,
            friction_breakaway_steps);
        return RIGHT_ARM_RNEA_OK;
    } catch (const std::exception& error) {
        WriteError(error_message, error_capacity, error.what());
        return RIGHT_ARM_RNEA_INTERNAL_ERROR;
    } catch (...) {
        WriteError(error_message, error_capacity, "RNEA 计算发生未知异常");
        return RIGHT_ARM_RNEA_INTERNAL_ERROR;
    }
}

RightArmRneaStatus right_arm_kinematics_batch_compute(
    RightArmRneaHandle* handle,
    const double* mujoco_qpos_reference,
    size_t qpos_count,
    const double* q_arm,
    const double* dq_arm,
    const uint8_t* acceleration_required,
    size_t state_count,
    RightArmKinematicsBatchOutput* output,
    char* error_message,
    size_t error_capacity) {
    WriteError(error_message, error_capacity, "");
    if (handle == nullptr || mujoco_qpos_reference == nullptr
        || q_arm == nullptr || dq_arm == nullptr
        || acceleration_required == nullptr || output == nullptr) {
        WriteError(error_message, error_capacity, "批运动学 C ABI 收到空指针");
        return RIGHT_ARM_RNEA_INVALID_ARGUMENT;
    }
    *output = RightArmKinematicsBatchOutput{};
    if (qpos_count != handle->core.nq()
        || state_count == 0
        || state_count > RIGHT_ARM_KINEMATICS_MAX_STATES) {
        WriteError(
            error_message,
            error_capacity,
            "批运动学 qpos 维度或 state_count 与接口上限不一致");
        return RIGHT_ARM_RNEA_DIMENSION_MISMATCH;
    }

    const size_t arm_value_count =
        state_count * static_cast<size_t>(RIGHT_ARM_RNEA_JOINT_COUNT);
    if (!IsFiniteArray(mujoco_qpos_reference, qpos_count)
        || !IsFiniteArray(q_arm, arm_value_count)
        || !IsFiniteArray(dq_arm, arm_value_count)) {
        WriteError(error_message, error_capacity, "批运动学输入含 NaN 或 Inf");
        return RIGHT_ARM_RNEA_NONFINITE_INPUT;
    }
    for (size_t state = 0; state < state_count; ++state) {
        if (acceleration_required[state] > 1) {
            WriteError(
                error_message,
                error_capacity,
                "acceleration_required 只能包含 0 或 1");
            return RIGHT_ARM_RNEA_INVALID_ARGUMENT;
        }
    }

    try {
        *output = handle->core.ComputeKinematicsBatch(
            mujoco_qpos_reference,
            q_arm,
            dq_arm,
            acceleration_required,
            state_count);
        return RIGHT_ARM_RNEA_OK;
    } catch (const std::exception& error) {
        WriteError(error_message, error_capacity, error.what());
        return RIGHT_ARM_RNEA_INTERNAL_ERROR;
    } catch (...) {
        WriteError(error_message, error_capacity, "批运动学计算发生未知异常");
        return RIGHT_ARM_RNEA_INTERNAL_ERROR;
    }
}

const char* right_arm_rnea_status_string(RightArmRneaStatus status) {
    switch (status) {
        case RIGHT_ARM_RNEA_OK:
            return "ok";
        case RIGHT_ARM_RNEA_INVALID_ARGUMENT:
            return "invalid_argument";
        case RIGHT_ARM_RNEA_DIMENSION_MISMATCH:
            return "dimension_mismatch";
        case RIGHT_ARM_RNEA_NONFINITE_INPUT:
            return "nonfinite_input";
        case RIGHT_ARM_RNEA_MODEL_ERROR:
            return "model_error";
        case RIGHT_ARM_RNEA_INTERNAL_ERROR:
            return "internal_error";
    }
    return "unknown";
}

uint32_t right_arm_rnea_abi_version(void) {
    return kAbiVersion;
}

}  // extern "C"
