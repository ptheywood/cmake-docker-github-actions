#! /usr/bin/env python3
import pycdga

# Dummy test file, not actually pytest?
def main():
    print("tests.py")
    print(f"(pycdga.VERSION {pycdga.VERSION}")
    print(f"(pycdga.VERSION_MAJOR {pycdga.VERSION_MAJOR}")
    print(f"(pycdga.VERSION_MINOR {pycdga.VERSION_MINOR}")
    print(f"(pycdga.VERSION_PATCH {pycdga.VERSION_PATCH}")
    print(f"(pycdga.VERSION_BUILDMETADATA {pycdga.VERSION_BUILDMETADATA}")
    d = pycdga.Demo()
    for i in range(4):
        d.demo()

if __name__ == "__main__":
    main()