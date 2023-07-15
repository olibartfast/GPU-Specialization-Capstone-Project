#pragma once
#include <fstream>
#include <sstream>
#include <cpuid.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;


std::string getGPUModel() {
    std::string gpuModel;

    // Iterate over the directories in /proc/driver/nvidia/gpus
    fs::path gpuDirectory("/proc/driver/nvidia/gpus");
    for (const auto& directory : fs::directory_iterator(gpuDirectory)) {
        std::string gpuDirectoryPath = directory.path();
        std::ifstream gpuInfoStream(gpuDirectoryPath + "/information");

        std::string gpuInfoLine;
        while (std::getline(gpuInfoStream, gpuInfoLine)) {
            if (gpuInfoLine.find("Model:") != std::string::npos) {
                size_t colonPos = gpuInfoLine.find(":");
                if (colonPos != std::string::npos) {
                    gpuModel = gpuInfoLine.substr(colonPos + 1);
                    break;
                }
            }
        }

        if (!gpuModel.empty()) {
            break;
        }
    }

    return gpuModel;
}


std::string getCPUInfo() {
    std::string cpuInfo;

    unsigned int cpuInfoRegs[4];
    __cpuid(0x80000000, cpuInfoRegs[0], cpuInfoRegs[1], cpuInfoRegs[2], cpuInfoRegs[3]);
    unsigned int extMaxId = cpuInfoRegs[0];

    char brand[48];
    if (extMaxId >= 0x80000004) {
        __cpuid(0x80000002, cpuInfoRegs[0], cpuInfoRegs[1], cpuInfoRegs[2], cpuInfoRegs[3]);
        memcpy(brand, cpuInfoRegs, sizeof(cpuInfoRegs));
        __cpuid(0x80000003, cpuInfoRegs[0], cpuInfoRegs[1], cpuInfoRegs[2], cpuInfoRegs[3]);
        memcpy(brand + 16, cpuInfoRegs, sizeof(cpuInfoRegs));
        __cpuid(0x80000004, cpuInfoRegs[0], cpuInfoRegs[1], cpuInfoRegs[2], cpuInfoRegs[3]);
        memcpy(brand + 32, cpuInfoRegs, sizeof(cpuInfoRegs));
        cpuInfo = brand;
    }

    return cpuInfo;
}
