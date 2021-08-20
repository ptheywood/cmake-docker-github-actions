// Doesn't have a matching header, because the matching header is <cuda.h>?
#include <cuda.h>


// Provide a stub containing the bits of libcuda.so required
// @todo - macros to generate this, or include per cuda version specific stuff? Not sure how we could check which version of the driver api is currently available at runtime? Might just have to assume based on teh version built with and let cuda emit the driver mismatch error?  

// @todo - use the macro to control if this is used? 

#if defined(USE_DLOPEN_CUDA)
#include <cdga/detail/DSOStuff.h>

// unnamed namespace to provide file scoped methods used to load the DSO?
namespace {

    // @todo this could be a lot better?
    void * getDSOHandleCUDADriverAPI() {
        // @todo - this might have other names? 
        static void * handle = cdga::detail::DSOStuff::OpenLibraryHandle("libcuda.so.1");
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


// Not using a namespace, as these need to be extern c and not in a namespace

// @todo include different versions here from separate files? Or use macros to do magical things.

// @todo - Not convinced by this type of stub, given it adds overhead to each call. Would rather just pull the symbols in? 

extern "C" {

CUresult CUDAAPI cuCtxGetCurrent( CUcontext* ctx ) {
    // @todo std::function? or a using statemnt? 
    static auto fptr = symbolFromDSO<CUresult(CUDAAPI *)(CUcontext*)>("cuCtxGetCurrent");
    if (!fptr) {
      // Return something indicating a generic error in that api. 
      return CUDA_ERROR_UNKNOWN; // This might not be the best choice? 
    }
    return fptr(ctx);
}


CUresult CUDAAPI cuGetErrorString(CUresult error, const char **pStr) {
    // @todo std::function? or a using statemnt? 
    static auto fptr = symbolFromDSO<CUresult(CUDAAPI *)(CUresult,const char**)>("cuGetErrorString");
    if (!fptr) {
      // Return something indicating a generic error in that api. 
      return CUDA_ERROR_UNKNOWN; // This might not be the best choice? 
    }
    return fptr(error, pStr);
}

// alternative? incomplete / not sure it will work outside of a function.
// CUresult (*cuGetErrorString)( CUresult, const char** );

// cuGetErrorString = (CUresult (*)( CUresult, const char** )) cdga::detail::DSOStuff::SymbolFromLibrary(getDSOHandleCUDADriverAPI(), "cuGetErrorString");

}  // extern "C" 



#endif