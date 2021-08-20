#ifndef INCLUDE_CDGA_DETAIL_DSOSTUFF_H_
#define INCLUDE_CDGA_DETAIL_DSOSTUFF_H_

// @todo - exceptions instead of exit()?
#include <cstdlib>  // for exit / EXIT_FAILURE
#include <stdio.h>  // fpritnf

// If Using dlopen, include the appropriate header. 
// This is only currently implemented for linux.
#if defined(USE_DLOPEN_CUDA)
    #if !defined(_WIN32)
        #include <dlfcn.h>
    #else 
        // @todo else use <windows.h>, for GetProcAddress
        #error "@todo - use GetProcAddress on windows"
    #endif  // _MSC_VER
#endif  // USE_DLOPEN

// C++17 namespace statements are great.
namespace cdga::detail {

// @todo - this is a terrible name
// Class for loading symbols from dynamic shared objects. 
// Only implemented for linux (i.e. not msvc at this time, and if an appropriate preprocessor symbol is passed.
// 
class DSOStuff {
public:
#if defined(USE_DLOPEN_CUDA) && !defined(_WIN32)
    // Static inlined methods for common operations between dlopen and windows equivalents
    static inline const char * GetError() {
        return dlerror();
    }
    /** 
     * Open a library, returning the handle to it. 
     */
    static inline void * OpenLibraryHandle(const char * lib) {
        void * handle = dlopen(lib, RTLD_LAZY);
        if (handle == nullptr) {
            // For now, die miserably. Exception would be better, or allow the calling location to deal with this?
            fprintf(stderr, "Failed to open DSO %s: %s\n", lib, dlerror());
            exit(EXIT_FAILURE);
        }
        // Clear any errors?
        dlerror();
        return handle;
    }

    // Using a non null handle, try to load a symbol
    static inline void * SymbolFromLibrary(void* libhandle, const char * symbol) {
        if (libhandle == nullptr) {
            fprintf(stderr, "Invalid DSO handle %p loading symbol %s\n", libhandle, symbol);
            exit(EXIT_FAILURE);
        }
        void * rtn = dlsym(libhandle, symbol);
        // Check for errors
        auto err = dlerror();
        if (err != NULL) {
            fprintf(stderr, "Error loading symbol %s: %s\n", symbol, err);
            exit(EXIT_FAILURE);
        }
        return rtn;
    }

    // close a handle, potentially changing the value to be nullptr
    // Can't be reference because of the void type
    // dlopen etc use void * for the handle :(
    // 0 indicates success.
    static inline int CloseLibraryHandle(void** libhandle) { 
        int rtn = 0; // 0 indicates success
        if (*libhandle != nullptr) {
            rtn = dlclose(*libhandle);
            *libhandle = nullptr;
            // Clear any errors?
            dlerror();
        }
        return rtn;
    }


    // Static method to load everything we want from a given library? 

#else 
    // @todo windows support
#endif
};

}  // namespace cdga::detail

#endif //  INCLUDE_CDGA_DETAIL_DSOSTUFF_H_
