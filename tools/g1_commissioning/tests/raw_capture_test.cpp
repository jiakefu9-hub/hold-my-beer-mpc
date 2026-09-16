#include "g1_commissioning/raw_capture.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <unitree/dds_wrapper/common/crc.h>

namespace gc = g1_commissioning;
int failures = 0;
#define CHECK(x) do { if (!(x)) { std::cerr << __LINE__ << ": " << #x << '\n'; ++failures; } } while(false)
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    // No ChannelFactory, subscribers, RPC clients or publishers instantiated.
    gc::RawImuRecord imu;
    imu.received_ns = 123;
    imu.sequence = 7;
    imu.message.quaternion({1,0,0,0});
    imu.message.accelerometer({0,0,9.81F});
    imu.message.gyroscope({1.25F,-0.75F,0.125F});
    const auto text = gc::RawImuJson(imu);
    CHECK(text.find("\"gyroscope_rad_s\":[1.25,-0.75,0.125]") != std::string::npos);
    CHECK(text.find("\"received_monotonic_ns\":123") != std::string::npos);
    CHECK(text.find("angular_acceleration") == std::string::npos);
    imu.message.gyroscope()[0] = std::numeric_limits<float>::quiet_NaN();
    CHECK(gc::RawImuJson(imu).find("\"NaN\"") != std::string::npos);
    gc::RawLowStateRecord low;
    low.message.tick(567);
    low.message.motor_state()[0].q(0.125F);
    low.message.motor_state()[26].dq(0.25F);
    low.message.crc(crc32_core(reinterpret_cast<std::uint32_t*>(&low.message),
        static_cast<std::uint32_t>(sizeof(low.message) / 4 - 1)));
    auto data = gc::RawLowStateJson(low);
    CHECK(data.find("\"crc_valid\":true") != std::string::npos);
    CHECK(data.find("\"q_rad\":0.125") != std::string::npos);
    CHECK(data.find("\"index\":34") != std::string::npos);
    low.message.tick(568);
    CHECK(gc::RawLowStateJson(low).find("\"crc_valid\":false") != std::string::npos);
    const auto directory = std::filesystem::path(argv[1]) / ("raw_test_" + std::to_string(gc::MonotonicNowNs()));
    gc::RawJournal journal(directory.string());
    for (int i = 0; i < 100; ++i) { imu.sequence = static_cast<std::uint64_t>(i); journal.Push(imu); journal.Push(low); }
    journal.Text(gc::CaptureEvent("test_end"));
    journal.Finish();
    CHECK(journal.Healthy());
    CHECK(journal.written() == 201 && journal.dropped() == 0);
    std::ifstream log(directory / "raw.jsonl");
    std::string line;
    int lines = 0;
    while (std::getline(log, line)) { CHECK(!line.empty() && line.front() == '{' && line.back() == '}'); ++lines; }
    CHECK(lines == 201);
    bool rejected = false;
    try { gc::RawJournal overwrite(directory.string()); } catch (...) { rejected = true; }
    CHECK(rejected);
    journal.Text("{}");
    CHECK(!journal.Healthy() && journal.dropped() == 1);
    std::cout << "raw preservation, complete drain and no-overwrite: " << failures << " failures\n";
    return failures ? 1 : 0;
}
