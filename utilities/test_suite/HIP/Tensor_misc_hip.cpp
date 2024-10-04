/*
MIT License

Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "../rpp_test_suite_misc.h"

void CreateDesc(RpptGenericDescPtr descriptorPtr3D, Rpp32u batch_size, Rpp32u height, Rpp32u width,
                           Rpp32u channels) {
    RpptGenericDesc result;

    descriptorPtr3D->layout = NHWC;
    descriptorPtr3D->dataType = F32;
    descriptorPtr3D->numDims = 4;
    descriptorPtr3D->offsetInBytes = 0;

    descriptorPtr3D->dims[0] = batch_size;
    descriptorPtr3D->dims[1] = height;
    descriptorPtr3D->dims[2] = width;
    descriptorPtr3D->dims[3] = channels;

    descriptorPtr3D->strides[3] = sizeof(Rpp32f);
    descriptorPtr3D->strides[2] = descriptorPtr3D->strides[3] * channels;
    descriptorPtr3D->strides[1] = descriptorPtr3D->strides[2] * width;
    descriptorPtr3D->strides[0] = descriptorPtr3D->strides[1] * height;

}

void fill_roi_values_sample(Rpp32u *roi_tensor, Rpp32u height, Rpp32u width,
                           Rpp32u channels)
{
    std::array<Rpp32u, 6> roi = {0, 0, 0, width, height, channels};
    std::copy(roi.begin(), roi.end(), &roi_tensor[0]);
}

int main(int argc, char **argv)
{
    // Handle inputs
    const int MIN_ARG_COUNT = 9;
    if (argc < MIN_ARG_COUNT)
    {
        cout << "\nImproper Usage! Needs all arguments!\n";
        cout << "\nUsage: ./Tensor_misc_hip <case number = 0:2> <test type 0/1> <toggle 0/1> <number of dimensions> <batch size> <num runs> <additional param> <dst path> <script path>\n";
        return -1;
    }
    Rpp32u testCase, testType, nDim, batchSize, numRuns, toggle;
    bool qaMode;

    testCase = atoi(argv[1]);
    testType = atoi(argv[2]);
    toggle = atoi(argv[3]);
    nDim = atoi(argv[4]);
    batchSize = atoi(argv[5]);
    numRuns = atoi(argv[6]);
    string dst = argv[8];
    string scriptPath = argv[9];
    qaMode = (testType == 0);
    bool axisMaskCase = (testCase == 1);
    bool permOrderCase = (testCase == 0);
    int additionalParam = (axisMaskCase || permOrderCase) ? atoi(argv[7]) : 1;
    int axisMask = additionalParam, permOrder = additionalParam;

    if (qaMode && batchSize != 3)
    {
        cout<<"QA mode can only run with batchsize 3"<<std::endl;
        return -1;
    }

    string funcName = augmentationMiscMap[testCase];
    if (funcName.empty())
    {
        cout << "\ncase " << testCase << " is not supported\n";
        return -1;
    }

    string func = funcName;
    if (axisMaskCase)
    {
        char additionalParam_char[2];
        std::sprintf(additionalParam_char, "%d", axisMask);
        func += "_" + std::to_string(nDim) + "d" + "_axisMask";
        func += additionalParam_char;
    }
    if (permOrderCase)
    {
        char additionalParam_char[2];
        std::sprintf(additionalParam_char, "%d", permOrder);
        func += "_" + std::to_string(nDim) + "d" + "_permOrder";
        func += additionalParam_char;
    }

    // fill roi based on mode and number of dimensions
    Rpp32u *roiTensor;
    CHECK_RETURN_STATUS(hipHostMalloc(&roiTensor, nDim * 2 * batchSize, sizeof(Rpp32u)));
    fill_roi_values(nDim, batchSize, roiTensor, qaMode);

    Rpp32u batch_size = 1;
    Rpp32u height = 480;
    Rpp32u width = 720;
    Rpp32u channels = 3;

    Rpp32u *roi_tensor;
    CHECK_RETURN_STATUS(hipHostMalloc(&roi_tensor, 6, sizeof(Rpp32u)));
    fill_roi_values_sample(roi_tensor, width, height, channels);

    // set src/dst generic tensor descriptors
    RpptGenericDescPtr srcDescriptorPtrND, dstDescriptorPtrND;
    CHECK_RETURN_STATUS(hipHostMalloc(&srcDescriptorPtrND, sizeof(RpptGenericDesc)));
    CHECK_RETURN_STATUS(hipHostMalloc(&dstDescriptorPtrND, sizeof(RpptGenericDesc)));

    RpptGenericDescPtr src_desc, dst_desc;
    CHECK_RETURN_STATUS(hipHostMalloc(&src_desc, sizeof(RpptGenericDesc)));
    CHECK_RETURN_STATUS(hipHostMalloc(&dst_desc, sizeof(RpptGenericDesc)));

    CreateDesc(src_desc, batch_size, height, width, channels);
    CreateDesc(dst_desc, batch_size, height, width, channels);

    // set dims and compute strides
    int bitDepth = 2, offSetInBytes = 0;
    set_generic_descriptor(srcDescriptorPtrND, nDim, offSetInBytes, bitDepth, batchSize, roiTensor);
    set_generic_descriptor(dstDescriptorPtrND, nDim, offSetInBytes, bitDepth, batchSize, roiTensor);
    set_generic_descriptor_layout(srcDescriptorPtrND, dstDescriptorPtrND, nDim, toggle, qaMode);

    Rpp32u bufferSize = 1;
    for(int i = 0; i <= nDim; i++)
        bufferSize *= srcDescriptorPtrND->dims[i];

    // allocate memory for input / output
    Rpp32f *inputF32 = NULL, *outputF32 = NULL;
    inputF32 = static_cast<Rpp32f *>(calloc(bufferSize, sizeof(Rpp32f)));
    outputF32 = static_cast<Rpp32f *>(calloc(bufferSize, sizeof(Rpp32f)));

    void *d_inputF32, *d_outputF32;
    CHECK_RETURN_STATUS(hipMalloc(&d_inputF32, bufferSize * sizeof(Rpp32f)));
    CHECK_RETURN_STATUS(hipMalloc(&d_outputF32, bufferSize * sizeof(Rpp32f)));

    Rpp32u total_size = batch_size * height * width * channels;

    RppPtr_t input = (RppPtr_t)(calloc(total_size, sizeof(Rpp32f)));

    // Populate input tensor with data which results in non-zero
    // mean/std_dev.
    for (int i = 0; i < total_size; i++) {
        static_cast<Rpp32f *>(input)[i] = static_cast<Rpp32f>(i);
    }

    RppPtr_t output = (RppPtr_t)(calloc(total_size, sizeof(Rpp32f)));

    void *d_input, *d_output;
    CHECK_RETURN_STATUS(hipMalloc(&d_input, total_size * sizeof(Rpp32f)));
    CHECK_RETURN_STATUS(hipMalloc(&d_output, total_size * sizeof(Rpp32f)));
    CHECK_RETURN_STATUS(hipMemcpy(d_input, input, total_size * sizeof(Rpp32f), hipMemcpyHostToDevice));
    CHECK_RETURN_STATUS(hipDeviceSynchronize());

    // read input data
    if(qaMode)
        read_data(inputF32, nDim, 0, scriptPath, funcName);
    else
    {
        std::srand(0);
        for(int i = 0; i < bufferSize; i++)
            inputF32[i] = static_cast<float>((std::rand() % 255));
    }

    // copy data from HOST to HIP
    CHECK_RETURN_STATUS(hipMemcpy(d_inputF32, inputF32, bufferSize * sizeof(Rpp32f), hipMemcpyHostToDevice));
    CHECK_RETURN_STATUS(hipDeviceSynchronize());

    Rpp32u *permTensor = nullptr;
    if (testCase == 0)
        CHECK_RETURN_STATUS(hipHostMalloc(&permTensor, nDim * sizeof(Rpp32u)));

    /*rppHandle_t handle;
    hipStream_t stream;
    CHECK_RETURN_STATUS(hipStreamCreate(&stream));
    rppCreateWithStreamAndBatchSize(&handle, stream, batchSize);*/

    rppHandle_t rpp_handle;
    hipStream_t stream;
    CHECK_RETURN_STATUS(hipStreamCreate(&stream));
    rppCreateWithBatchSize(&rpp_handle, 1);

    Rpp32f *meanTensor = nullptr, *stdDevTensor = nullptr;
    Rpp32f *meanTensorCPU = nullptr, *stdDevTensorCPU = nullptr;
    Rpp32f *meanArr = nullptr, *stdDevArr = nullptr;
    Rpp32f *mean = nullptr, *std_dev = nullptr;
    bool externalMeanStd = true;

    double startWallTime, endWallTime;
    double maxWallTime = 0, minWallTime = 500, avgWallTime = 0, wallTime = 0;
    string testCaseName;

    // case-wise RPP API and measure time script for Unit and Performance test
    cout << "\nRunning " << func << " " << numRuns << " times (each time with a batch size of " << batchSize << ") and computing mean statistics...";
    for(int perfCount = 0; perfCount < numRuns; perfCount++)
    {
        switch(testCase)
        {
            case 0:
            {
                testCaseName  = "transpose";
                fill_perm_values(nDim, permTensor, qaMode, permOrder);

                for(int i = 1; i <= nDim; i++)
                    dstDescriptorPtrND->dims[i] = roiTensor[nDim + permTensor[i - 1]];
                compute_strides(dstDescriptorPtrND);

                startWallTime = omp_get_wtime();
                rppt_transpose_gpu(d_inputF32, srcDescriptorPtrND, d_outputF32, dstDescriptorPtrND, permTensor, roiTensor, rpp_handle);

                break;
            }
            case 1:
            {
                testCaseName  = "normalize";
                float scale = 1.0;
                float shift = 0.0;

                // computeMeanStddev set to 3 means both mean and stddev should be computed internally.
                // Wherein 0th bit used to represent computeMean and 1st bit for computeStddev.
                Rpp8u computeMeanStddev = 3;
                externalMeanStd = !computeMeanStddev; // when mean and stddev is passed from user

                Rpp32u size = 1; // length of mean and stddev tensors differ based on axisMask and nDim
                Rpp32u maxSize = 1;
                for(int batch = 0; batch < batchSize; batch++)
                {
                    size = 1;
                    for(int i = 0; i < nDim; i++)
                        size *= ((axisMask & (int)(pow(2,i))) >= 1) ? 1 : roiTensor[(nDim * 2 * batch) + nDim + i];
                    maxSize = max(maxSize, size);
                }

                // allocate memory if no memory is allocated
                if(meanTensor == nullptr)
                    CHECK_RETURN_STATUS(hipMalloc(&meanTensor, maxSize * batchSize * sizeof(Rpp32f)));

                if(stdDevTensor == nullptr)
                    CHECK_RETURN_STATUS(hipMalloc(&stdDevTensor, maxSize * batchSize * sizeof(Rpp32f)));

                if(mean == nullptr)
                    CHECK_RETURN_STATUS(hipMalloc(&mean, 3 * sizeof(Rpp32f)));

                if(std_dev == nullptr)
                    CHECK_RETURN_STATUS(hipMalloc(&std_dev, 3 * sizeof(Rpp32f)));

                if(!computeMeanStddev)
                {
                    if(meanTensorCPU == nullptr)
                        meanTensorCPU = static_cast<Rpp32f *>(malloc(maxSize * sizeof(Rpp32f)));
                    if(stdDevTensorCPU == nullptr)
                        stdDevTensorCPU = static_cast<Rpp32f *>(malloc(maxSize * sizeof(Rpp32f)));
                    fill_mean_stddev_values(nDim, maxSize, meanTensorCPU, stdDevTensorCPU, qaMode, axisMask, scriptPath);
                    CHECK_RETURN_STATUS(hipMemcpy(meanTensor, meanTensorCPU, maxSize * sizeof(Rpp32f), hipMemcpyHostToDevice));
                    CHECK_RETURN_STATUS(hipMemcpy(stdDevTensor, stdDevTensorCPU, maxSize * sizeof(Rpp32f), hipMemcpyHostToDevice));
                    CHECK_RETURN_STATUS(hipDeviceSynchronize());
                }

                startWallTime = omp_get_wtime();

                Rpp32u axis_mask = 3;
                Rpp8u compute_mean_stddev = 3;

                // Note: Shift is 128 to center at 128 instead of 0, since centering around
                // 0 will cause underflow on uint8_t.

                //rppt_normalize_gpu(d_inputF32, srcDescriptorPtrND, d_outputF32, dstDescriptorPtrND, axisMask, meanTensor, stdDevTensor, computeMeanStddev, scale, shift, roiTensor, handle);
                rppt_normalize_gpu(d_input, src_desc, d_output, dst_desc, axis_mask, mean,
                        std_dev, compute_mean_stddev, 1.0f, 128.0f, roi_tensor,
                        rpp_handle);
                printf("Mean: [%f, %f, %f]\nStandard Deviation: [%f, %f, %f]\n", mean[0],
                        mean[1], mean[2], std_dev[0], std_dev[1], std_dev[2]);
                printf("First 100 values of output:\n");
                for (int i = 0; i < 100; i++) {
                    printf("%f, ", static_cast<Rpp32f *>(output)[i]);
                }
                printf("\n");
                break;
            }
            case 2:
            {
                testCaseName  = "log";

                startWallTime = omp_get_wtime();
                rppt_log_gpu(d_inputF32, srcDescriptorPtrND, d_outputF32, dstDescriptorPtrND, roiTensor, rpp_handle);

                break;
            }
            default:
            {
                cout << "functionality is not supported" <<std::endl;
                exit(0);
            }
        }
        CHECK_RETURN_STATUS(hipDeviceSynchronize());
        endWallTime = omp_get_wtime();

        wallTime = endWallTime - startWallTime;
        maxWallTime = std::max(maxWallTime, wallTime);
        minWallTime = std::min(minWallTime, wallTime);
        avgWallTime += wallTime;
    }
    rppDestroyGPU(rpp_handle);

    printf("Error before comparison\n");
    // compare outputs if qaMode is true
    if(qaMode)
    {
        CHECK_RETURN_STATUS(hipDeviceSynchronize());
        CHECK_RETURN_STATUS(hipMemcpy(outputF32, d_outputF32, bufferSize * sizeof(Rpp32f), hipMemcpyDeviceToHost));
        CHECK_RETURN_STATUS(hipDeviceSynchronize());
        compare_output(outputF32, nDim, batchSize, bufferSize, dst, func, testCaseName, additionalParam, scriptPath, externalMeanStd);
    }
    else
    {
        maxWallTime *= 1000;
        minWallTime *= 1000;
        avgWallTime *= 1000;
        avgWallTime /= numRuns;
        cout << fixed << "\nmax,min,avg wall times in ms/batch = " << maxWallTime << "," << minWallTime << "," << avgWallTime;
    }


    free(inputF32);
    free(outputF32);
    free(input);
    free(output);
    CHECK_RETURN_STATUS(hipHostFree(srcDescriptorPtrND));
    CHECK_RETURN_STATUS(hipHostFree(dstDescriptorPtrND));
    CHECK_RETURN_STATUS(hipHostFree(src_desc));
    CHECK_RETURN_STATUS(hipHostFree(dst_desc));
    CHECK_RETURN_STATUS(hipHostFree(roiTensor));
    CHECK_RETURN_STATUS(hipHostFree(roi_tensor));
    CHECK_RETURN_STATUS(hipFree(d_input));
    CHECK_RETURN_STATUS(hipFree(d_output));
    CHECK_RETURN_STATUS(hipFree(d_inputF32));
    CHECK_RETURN_STATUS(hipFree(d_outputF32));
    if(meanTensor != nullptr)
        CHECK_RETURN_STATUS(hipFree(meanTensor));
    if(stdDevTensor != nullptr)
        CHECK_RETURN_STATUS(hipFree(stdDevTensor));
    if(meanTensor != nullptr)
        CHECK_RETURN_STATUS(hipFree(mean));
    if(stdDevTensor != nullptr)
        CHECK_RETURN_STATUS(hipFree(std_dev));
    if (permTensor != nullptr)
        CHECK_RETURN_STATUS(hipHostFree(permTensor));
    if(meanTensorCPU != nullptr)
        free(meanTensorCPU);
    if(stdDevTensorCPU != nullptr)
        free(stdDevTensorCPU);

    return 0;
}
