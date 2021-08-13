#include <stdio.h>
#include <cstdlib>

#ifdef NDEBUG
#undef NDEBUG
#endif  // NDEBUG
#include <assert.h>

#include "cdga/cdga.h"

// Terrible pretend tests.
void test_version() { 
    #if defined(CDGA_VERSION) && CDGA_VERSION > 0
        assert(true);
    #else
        assert(false);
    #endif
    assert(CDGA_VERSION == cdga::VERSION);
    printf("cdga::VERSION %d\n", cdga::VERSION);
    printf("cdga::VERSION_MAJOR %d\n", cdga::VERSION_MAJOR);
    printf("cdga::VERSION_MINOR %d\n", cdga::VERSION_MINOR);
    printf("cdga::VERSION_PATCH %d\n", cdga::VERSION_PATCH);
    printf("cdga::VERSION_BUILDMETADATA %s\n", cdga::VERSION_BUILDMETADATA);
}

void test_Demo() {
    auto d = cdga::Demo();
    constexpr unsigned int N = 4;
    for(unsigned int i = 0; i < N; i++) {
        d.demo();
    }
}

int main(int argc, char * argv[]) {
    bool success = true;
    test_version();
    test_Demo();
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
