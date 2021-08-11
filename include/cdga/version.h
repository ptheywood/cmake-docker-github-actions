#ifndef INCLUDE_CDGA_VERSION_H_
#define INCLUDE_CDGA_VERSION_H_

// #defined version, MAJOR, 3 digit MINOR, 3 digit PATCH
#define CDGA_VERSION 0000001

namespace cdga {

// Namesapced VERSION 
static constexpr unsigned int VERSION = CDGA_VERSION;
// Major
static constexpr unsigned int VERSION_MAJOR = cdga::VERSION / 1000000;
// Minor
static constexpr unsigned int VERSION_MINOR = cdga::VERSION / 1000 % 1000;
// Patch
static constexpr unsigned int VERSION_PATCH = cdga::VERSION % 1000;

// Build meta data, implemented in version.cpp
extern const char VERSION_BUILDMETADATA[];

}  // namespace cdga

#endif //  INCLUDE_CDGA_VERSION_H_
