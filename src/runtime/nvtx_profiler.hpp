#pragma once
#include <string>

#ifdef USE_NVTX
    #include <nvToolsExt.h>
    #define NVTX_RANGE_PUSH(name) nvtxRangePushA(name)
    #define NVTX_RANGE_POP()      nvtxRangePop()
#else
    // Graceful fallback for non-NVTX environments
    #define NVTX_RANGE_PUSH(name) ((void)0)
    #define NVTX_RANGE_POP()      ((void)0)
#endif

/**
 * @brief Scoped RAII NVTX Range Tracer
 * 
 * Automatically creates timeline ranges for NVIDIA Nsight Systems (nsys)
 * and Nsight Compute (ncu) profiling.
 * 
 * Usage:
 *   {
 *       ScopedNvtxRange range("FusedGeLU_Execution");
 *       launch_fused_bias_gelu(...);
 *   } // Automatically pops range on scope exit
 */
class ScopedNvtxRange {
public:
    explicit ScopedNvtxRange(const char* name) {
        NVTX_RANGE_PUSH(name);
    }
    ~ScopedNvtxRange() {
        NVTX_RANGE_POP();
    }
    ScopedNvtxRange(const ScopedNvtxRange&) = delete;
    ScopedNvtxRange& operator=(const ScopedNvtxRange&) = delete;
};
