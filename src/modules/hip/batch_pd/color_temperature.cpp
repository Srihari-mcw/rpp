#include <hip/hip_runtime.h>
#include "rpp_hip_host_decls.hpp"

#define saturate_8u(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)))

__device__ unsigned char temperature(unsigned char input, unsigned char value, int RGB)
{
    if(RGB == 0)
    {
        return saturate_8u((short)input + (short)value);
    }
    else if(RGB == 1)
    {
        return (input);
    }
    else
    {
        return saturate_8u((short)input - (short)value);
    }
}

extern "C" __global__ void temperature_planar(unsigned char *input,
                                              unsigned char *output,
                                              const unsigned int height,
                                              const unsigned int width,
                                              const unsigned int channel,
                                              const int modificationValue)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if (id_x >= width || id_y >= height)
    {
        return;
    }

    int pixIdx = id_x + id_y * width;
    int c = width * height;
    int res = input[pixIdx] + modificationValue;
    output[pixIdx] = saturate_8u(res);

    if (channel > 1)
    {
        output[pixIdx + c] = input[pixIdx + c];
        res = input[pixIdx + c * 2] - modificationValue;
        output[pixIdx + c * 2] = saturate_8u(res);
    }
}

extern "C" __global__ void temperature_packed(unsigned char *input,
                                              unsigned char *output,
                                              const unsigned int height,
                                              const unsigned int width,
                                              const unsigned int channel,
                                              const int modificationValue)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if (id_x >= width || id_y >= height)
    {
        return;
    }

    int pixIdx = id_y * width * channel + id_x * channel;
    int res = input[pixIdx] + modificationValue;
    output[pixIdx] = saturate_8u(res);
    output[pixIdx + 1] = input[pixIdx + 1];
    res = input[pixIdx+2] - modificationValue;
    output[pixIdx+2] = saturate_8u(res);
}



extern "C" __global__ void color_temperature_batch(unsigned char *input,
                                                   unsigned char *output,
                                                   int *modificationValue,
                                                   unsigned int *xroi_begin,
                                                   unsigned int *xroi_end,
                                                   unsigned int *yroi_begin,
                                                   unsigned int *yroi_end,
                                                   unsigned int *height,
                                                   unsigned int *width,
                                                   unsigned int *max_width,
                                                   unsigned long long *batch_index,
                                                   const unsigned int channel,
                                                   unsigned int *inc, // use width * height for pln and 1 for pkd
                                                   const int plnpkdindex) // use 1 pln 3 for pkd
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    unsigned char modificationValuetmp = modificationValue[id_z];
    long pixIdx = 0;
    pixIdx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex;

    if((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {
        for(int indextmp = channel - 1; indextmp >= 0; indextmp--)
        {
            output[pixIdx] = temperature(input[pixIdx], modificationValuetmp, indextmp);
            pixIdx += inc[id_z];
        }
    }
    else if((id_x < width[id_z] ) && (id_y < height[id_z]))
    {
            for(int indextmp = 0; indextmp < channel; indextmp++)
            {
                output[pixIdx] = input[pixIdx];
                pixIdx += inc[id_z];
            }
        }
}

RppStatus hip_exec_color_temperature_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(color_temperature_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle.GetInitHandle()->mem.mgpu.intArr[0].intmem,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                       handle.GetInitHandle()->mem.mgpu.srcSize.height,
                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
                       handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                       handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                       channel,
                       handle.GetInitHandle()->mem.mgpu.inc,
                       plnpkdind);

    return RPP_SUCCESS;
}