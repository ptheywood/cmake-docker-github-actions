#include <cuda_runtime.h>


// Provide a stub containing the bits of libcuda.so required
// @todo - macros to generate this, or include per cuda version specific stuff? Not sure how we could check which version of the driver api is currently available at runtime? Might just have to assume based on teh version built with and let cuda emit the driver mismatch error?  

// @todo - use the macro to control if this is used? 

#if defined(USE_DLOPEN_CUDA)
#include <cdga/detail/DSOStuff.h>
#define SSTRING(s) #s
#define STRING(s) SSTRING(s)


/* // unnamed namespace to provide file scoped methods used to load the DSO?
namespace {

    // @todo this could be a lot better?
    void * getDSOHandleCUDADriverAPI() {
        // @todo - this might have other names? 
        const char * dsoname = "libcuda.so." STRING(__CUDACC_VER_MAJOR__) "." STRING(__CUDACC_VER_MINOR__);
        static void * handle = cdga::detail::DSOStuff::OpenLibraryHandle(dsoname);
        // @todo - how to close this handle... 
        // detail::DSOStuff::CloseLibraryHandle(&handle);

        return handle;
    }

    template <typename T>
    T symbolFromDSO(const char * name) {
      void * sym = nullptr;
      // Get the handle that should be opened once?
      void * handle = getDSOHandleCUDADriverAPI();
      if (handle != nullptr) {
        sym = cdga::detail::DSOStuff::SymbolFromLibrary(handle, name);
      }
      return reinterpret_cast<T>(sym);
    }

}  // namespace

#undef SSTRING
#undef STRING


// Not using a namespace, as these need to be extern c and not in a namespace

// @todo include different versions here from separate files? Or use macros to do magical things.

// @todo - Not convinced by this type of stub, given it adds overhead to each call. Would rather just pull the symbols in? 

extern "C" {

// cudaFree
// __cudaPopCallConfiguration
// __cudaPushCallConfiguration
// __cudaRegisterFatBinary
// __cudaRegisterFatBinaryEnd
// __cudaRegisterFunction
// __cudaUnregisterFatBinary
// cudaDeviceSynchronize
// cudaFree
// cudaGetErrorString
// cudaGetLastError
// cudaLaunchKernel


}  // extern "C"  */



#endif