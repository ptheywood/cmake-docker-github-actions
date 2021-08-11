#include "cdga/Demo.h"

#include <stdio.h>

namespace cdga {

Demo::Demo() : count(0) { }

Demo::~Demo() { }

void Demo::demo() {
    this->count++;
    printf("Demo::demo() has been called %u times\n", this->count);
}

}  // namespace cdga

