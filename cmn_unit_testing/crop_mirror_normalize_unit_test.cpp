#include <stdio.h>
#include <dirent.h>
#include <string.h> 
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "filesystem.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>
#include <fstream>
#include <turbojpeg.h>
#include <random>
#include <map>
#include "rppdefs.h"
#include "rpp.h"
#include <hip/hip_runtime_api.h>
#include <hip/hip_fp16.h>

#include "rppt_tensor_geometric_augmentations.h"
using namespace std;
using namespace cv;

typedef half Rpp16f;

#define MAX_IMAGE_DUMP 20

inline int set_input_channels(int layoutType)
{
    if(layoutType == 0 || layoutType == 1)
        return 3;
    else
        return 1;
}

std::map<int, string> augmentationMap =
{
    {0, "brightness"},
    {1, "gamma_correction"},
    {2, "blend"},
    {4, "contrast"},
    {5, "pixelate"},
    {6, "jitter"},
    {8, "noise"},
    {13, "exposure"},
    {20, "flip"},
    {21, "resize"},
    {23, "rotate"},
    {24, "warp_afffine"},
    {26, "lens_correction"},
    {29, "water"},
    {30, "non_linear_blend"},
    {31, "color_cast"},
    {32, "erase"},
    {33, "crop_and_patch"},
    {34, "lut"},
    {35, "glitch"},
    {36, "color_twist"},
    {37, "crop"},
    {38, "crop_mirror_normalize"},
    {39, "resize_crop_mirror"},
    {45, "color_temperature"},
    {46, "vignette"},
    {49, "box_filter"},
    {54, "gaussian_filter"},
    {61, "magnitude"},
    {63, "phase"},
    {65, "bitwise_and"},
    {68, "bitwise_or"},
    {70, "copy"},
    {79, "remap"},
    {80, "resize_mirror_normalize"},
    {81, "color_jitter"},
    {82, "ricap"},
    {83, "gridmask"},
    {84, "spatter"},
    {85, "swap_channels"},
    {86, "color_to_greyscale"},
    {87, "tensor_sum"},
    {88, "tensor_min"},
    {89, "tensor_max"},
    {90, "tensor_mean"},
    {91, "tensor_stddev"},
    {92, "slice"}
};

// converts image data from PLN3 to PKD3
inline void convert_pln3_to_pkd3(Rpp8u *output, RpptDescPtr descPtr)
{
    unsigned long long bufferSize = ((unsigned long long)descPtr->h * (unsigned long long)descPtr->w * (unsigned long long)descPtr->c * (unsigned long long)descPtr->n) + descPtr->offsetInBytes;
    Rpp8u *outputCopy = (Rpp8u *)calloc(bufferSize, sizeof(Rpp8u));
    memcpy(outputCopy, output, bufferSize * sizeof(Rpp8u));

    Rpp8u *outputCopyTemp;
    outputCopyTemp = outputCopy + descPtr->offsetInBytes;

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(descPtr->n)
    for (int count = 0; count < descPtr->n; count++)
    {
        Rpp8u *outputCopyTempR, *outputCopyTempG, *outputCopyTempB;
        outputCopyTempR = outputCopyTemp + count * descPtr->strides.nStride;
        outputCopyTempG = outputCopyTempR + descPtr->strides.cStride;
        outputCopyTempB = outputCopyTempG + descPtr->strides.cStride;
        Rpp8u *outputTemp = output + descPtr->offsetInBytes + count * descPtr->strides.nStride;

        for (int i = 0; i < descPtr->h; i++)
        {
            for (int j = 0; j < descPtr->w; j++)
            {
                *outputTemp = *outputCopyTempR;
                outputTemp++;
                outputCopyTempR++;
                *outputTemp = *outputCopyTempG;
                outputTemp++;
                outputCopyTempG++;
                *outputTemp = *outputCopyTempB;
                outputTemp++;
                outputCopyTempB++;
            }
        }
    }

    free(outputCopy);
}

template <typename T>
inline T validate_pixel_range(T pixel)
{
    pixel = (pixel < static_cast<Rpp32f>(0)) ? (static_cast<Rpp32f>(0)) : ((pixel < static_cast<Rpp32f>(255)) ? pixel : (static_cast<Rpp32f>(255)));
    return pixel;
}

inline void convert_output_bitdepth_to_u8(void *output, Rpp8u *outputu8, int inputBitDepth, Rpp64u oBufferSize, Rpp64u outputBufferSize, RpptDescPtr dstDescPtr, Rpp32f invConversionFactor)
{
    if (inputBitDepth == 0)
    {
        memcpy(outputu8, output, outputBufferSize);
    }
    else if ((inputBitDepth == 1) || (inputBitDepth == 3))
    {
        Rpp8u *outputTemp = outputu8 + dstDescPtr->offsetInBytes;
        Rpp16f *outputf16Temp = reinterpret_cast<Rpp16f *>(static_cast<Rpp8u *>(output) + dstDescPtr->offsetInBytes);
        for (int i = 0; i < oBufferSize; i++)
        {
            *outputTemp = static_cast<Rpp8u>(validate_pixel_range(static_cast<float>(*outputf16Temp) * invConversionFactor));
            outputf16Temp++;
            outputTemp++;
        }
    }
    else if ((inputBitDepth == 2) || (inputBitDepth == 4))
    {
        Rpp8u *outputTemp = outputu8 + dstDescPtr->offsetInBytes;
        Rpp32f *outputf32Temp = reinterpret_cast<Rpp32f *>(static_cast<Rpp8u *>(output) + dstDescPtr->offsetInBytes);
        for (int i = 0; i < oBufferSize; i++)
        {
            *outputTemp = static_cast<Rpp8u>(validate_pixel_range(*outputf32Temp * invConversionFactor));
            outputf32Temp++;
            outputTemp++;
        }
    }
    else if ((inputBitDepth == 5) || (inputBitDepth == 6))
    {
        Rpp8u *outputTemp = outputu8 + dstDescPtr->offsetInBytes;
        Rpp8s *outputi8Temp = static_cast<Rpp8s *>(output) + dstDescPtr->offsetInBytes;
        for (int i = 0; i < oBufferSize; i++)
        {
            *outputTemp = static_cast<Rpp8u>(validate_pixel_range((static_cast<Rpp32s>(*outputi8Temp) + 128)));
            outputi8Temp++;
            outputTemp++;
        }
    }
}

inline void convert_roi(RpptROI *roiTensorPtrSrc, RpptRoiType roiType, int batchSize)
{
    if(roiType == RpptRoiType::LTRB)
    {
        for (int i = 0; i < batchSize; i++)
        {
            RpptRoiXywh roi = roiTensorPtrSrc[i].xywhROI;
            roiTensorPtrSrc[i].ltrbROI = {roi.xy.x, roi.xy.y, roi.roiWidth - roi.xy.x, roi.roiHeight - roi.xy.y};
        }
    }
    else
    {
        for (int i = 0; i < batchSize; i++)
        {
            RpptRoiLtrb roi = roiTensorPtrSrc[i].ltrbROI;
            roiTensorPtrSrc[i].xywhROI = {roi.lt.x, roi.lt.y, roi.rb.x - roi.lt.x + 1, roi.rb.y - roi.lt.y + 1};
        }
    }
}

// converts image data from PKD3 to PLN3
inline void convert_pkd3_to_pln3(Rpp8u *input, RpptDescPtr descPtr)
{
    unsigned long long bufferSize = ((unsigned long long)descPtr->h * (unsigned long long)descPtr->w * (unsigned long long)descPtr->c * (unsigned long long)descPtr->n) + descPtr->offsetInBytes;
    Rpp8u *inputCopy = (Rpp8u *)calloc(bufferSize, sizeof(Rpp8u));
    memcpy(inputCopy, input, bufferSize * sizeof(Rpp8u));

    Rpp8u *inputTemp, *inputCopyTemp;
    inputTemp = input + descPtr->offsetInBytes;

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(descPtr->n)
    for (int count = 0; count < descPtr->n; count++)
    {
        Rpp8u *inputTempR, *inputTempG, *inputTempB;
        inputTempR = inputTemp + count * descPtr->strides.nStride;
        inputTempG = inputTempR + descPtr->strides.cStride;
        inputTempB = inputTempG + descPtr->strides.cStride;
        Rpp8u *inputCopyTemp = inputCopy + descPtr->offsetInBytes + count * descPtr->strides.nStride;

        for (int i = 0; i < descPtr->h; i++)
        {
            for (int j = 0; j < descPtr->w; j++)
            {
                *inputTempR = *inputCopyTemp;
                inputCopyTemp++;
                inputTempR++;
                *inputTempG = *inputCopyTemp;
                inputCopyTemp++;
                inputTempG++;
                *inputTempB = *inputCopyTemp;
                inputCopyTemp++;
                inputTempB++;
            }
        }
    }

    free(inputCopy);
}

// Opens a folder and recursively search for files with given extension
void open_folder(const string& folderPath, vector<string>& imageNames, vector<string>& imageNamesPath, string extension)
{
    auto src_dir = opendir(folderPath.c_str());
    struct dirent* entity;
    std::string fileName = " ";

    if (src_dir == nullptr)
        std::cerr << "\n ERROR: Failed opening the directory at " <<folderPath;

    while((entity = readdir(src_dir)) != nullptr)
    {
        string entry_name(entity->d_name);
        if (entry_name == "." || entry_name == "..")
            continue;
        fileName = entity->d_name;
        std::string filePath = folderPath;
        filePath.append("/");
        filePath.append(entity->d_name);
        fs::path pathObj(filePath);
        if(fs::exists(pathObj) && fs::is_directory(pathObj))
            open_folder(filePath, imageNames, imageNamesPath, extension);

        if (fileName.size() > 4 && fileName.substr(fileName.size() - 4) == extension)
        {
            imageNamesPath.push_back(filePath);
            imageNames.push_back(entity->d_name);
        }
    }
    if(imageNames.empty())
        std::cerr << "\n Did not load any file from " << folderPath;

    closedir(src_dir);
}

// Write a batch of images using the OpenCV library
inline void write_image_batch_opencv(string outputFolder, Rpp8u *output, RpptDescPtr dstDescPtr, vector<string>::const_iterator imagesNamesStart, RpptImagePatch *dstImgSizes, int maxImageDump)
{
    // create output folder
    mkdir(outputFolder.c_str(), 0700);
    outputFolder += "/";
    static int cnt = 1;
    static int imageCnt = 0;

    Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
    Rpp8u *offsettedOutput = output + dstDescPtr->offsetInBytes;
    for (int j = 0; (j < dstDescPtr->n) && (imageCnt < maxImageDump) ; j++, imageCnt++)
    {
        Rpp32u height = dstImgSizes[j].height;
        Rpp32u width = dstImgSizes[j].width;
        Rpp32u elementsInRow = width * dstDescPtr->c;
        Rpp32u outputSize = height * width * dstDescPtr->c;
        Rpp8u *tempOutput = (Rpp8u *)calloc(outputSize, sizeof(Rpp8u));
        Rpp8u *tempOutputRow = tempOutput;
        Rpp8u *outputRow = offsettedOutput + j * dstDescPtr->strides.nStride;
        for (int k = 0; k < height; k++)
        {
            memcpy(tempOutputRow, outputRow, elementsInRow * sizeof(Rpp8u));
            tempOutputRow += elementsInRow;
            outputRow += elementsInRowMax;
        }
        string outputImagePath = outputFolder + *(imagesNamesStart + j);
        Mat matOutputImage, matOutputImageRgb;
        if (dstDescPtr->c == 1)
            matOutputImage = Mat(height, width, CV_8UC1, tempOutput);
        else if (dstDescPtr->c == 2)
            matOutputImage = Mat(height, width, CV_8UC2, tempOutput);
        else if (dstDescPtr->c == 3)
        {
            matOutputImageRgb = Mat(height, width, CV_8UC3, tempOutput);
            cvtColor(matOutputImageRgb, matOutputImage, COLOR_RGB2BGR);
        }

        fs::path pathObj(outputImagePath);
        if (fs::exists(pathObj))
        {
            std::string outPath = outputImagePath.substr(0, outputImagePath.find_last_of('.')) + "_" + to_string(cnt) + outputImagePath.substr(outputImagePath.find_last_of('.'));
            imwrite(outPath, matOutputImage);
            cnt++;
        }
        else
            imwrite(outputImagePath, matOutputImage);
        free(tempOutput);
    }
}

// sets roi xywh values and dstImg sizes
inline void  set_src_and_dst_roi(vector<string>::const_iterator imagePathsStart, vector<string>::const_iterator imagePathsEnd, RpptROI *roiTensorPtrSrc, RpptROI *roiTensorPtrDst, RpptImagePatchPtr dstImgSizes)
{
    tjhandle tjInstance = tjInitDecompress();
    int i = 0;
    for (auto imagePathIter = imagePathsStart; imagePathIter != imagePathsEnd; ++imagePathIter, i++)
    {
        const string& imagePath = *imagePathIter;
        FILE* jpegFile = fopen(imagePath.c_str(), "rb");
        if (!jpegFile) {
            std::cerr << "Error opening file: " << imagePath << std::endl;
            continue;
        }

        fseek(jpegFile, 0, SEEK_END);
        long fileSize = ftell(jpegFile);
        fseek(jpegFile, 0, SEEK_SET);

        std::vector<unsigned char> jpegBuffer(fileSize);
        fread(jpegBuffer.data(), 1, fileSize, jpegFile);
        fclose(jpegFile);

        int jpegSubsamp;
        int width, height;
        if (tjDecompressHeader2(tjInstance, jpegBuffer.data(), jpegBuffer.size(), &width, &height, &jpegSubsamp) == -1) {
            std::cerr << "Error decompressing file: " << imagePath << std::endl;
            continue;
        }

        roiTensorPtrSrc[i].xywhROI = {0, 0, width, height};
        roiTensorPtrDst[i].xywhROI = {0, 0, width, height};
        dstImgSizes[i].width = roiTensorPtrDst[i].xywhROI.roiWidth;
        dstImgSizes[i].height = roiTensorPtrDst[i].xywhROI.roiHeight;
    }
    tjDestroy(tjInstance);
}

inline size_t get_size_of_data_type(RpptDataType dataType)
{
    if(dataType == RpptDataType::U8)
        return sizeof(Rpp8u);
    else if(dataType == RpptDataType::I8)
        return sizeof(Rpp8s);
    else if(dataType == RpptDataType::F16)
        return sizeof(Rpp16f);
    else if(dataType == RpptDataType::F32)
        return sizeof(Rpp32f);
    else
        return 0;
}


// sets descriptor dimensions and strides of src/dst
inline void set_descriptor_dims_and_strides(RpptDescPtr descPtr, int noOfImages, int maxHeight, int maxWidth, int numChannels, int offsetInBytes)
{
    descPtr->numDims = 4;
    descPtr->offsetInBytes = offsetInBytes;
    descPtr->n = noOfImages;
    descPtr->h = maxHeight;
    descPtr->w = maxWidth;
    descPtr->c = numChannels;

    // Optionally set w stride as a multiple of 8 for src/dst
    descPtr->w = ((descPtr->w / 8) * 8) + 8;
    // set strides
    if (descPtr->layout == RpptLayout::NHWC)
    {
        descPtr->strides.nStride = descPtr->c * descPtr->w * descPtr->h;
        descPtr->strides.hStride = descPtr->c * descPtr->w;
        descPtr->strides.wStride = descPtr->c;
        descPtr->strides.cStride = 1;
    }
    else if(descPtr->layout == RpptLayout::NCHW)
    {
        descPtr->strides.nStride = descPtr->c * descPtr->w * descPtr->h;
        descPtr->strides.cStride = descPtr->w * descPtr->h;
        descPtr->strides.hStride = descPtr->w;
        descPtr->strides.wStride = 1;
    }
}

// Read a batch of images using the turboJpeg decoder
inline void read_image_batch_turbojpeg(Rpp8u *input, RpptDescPtr descPtr, vector<string>::const_iterator imagesNamesStart)
{
    tjhandle m_jpegDecompressor = tjInitDecompress();

    // Loop through the input images
    for (int i = 0; i < descPtr->n; i++)
    {
        // Read the JPEG compressed data from a file
        std::string inputImagePath = *(imagesNamesStart + i);
        FILE* fp = fopen(inputImagePath.c_str(), "rb");
        if(!fp)
            std::cerr << "\n unable to open file : "<<inputImagePath;
        fseek(fp, 0, SEEK_END);
        long jpegSize = ftell(fp);
        rewind(fp);
        unsigned char* jpegBuf = (unsigned char*)calloc(jpegSize, sizeof(Rpp8u));
        fread(jpegBuf, 1, jpegSize, fp);
        fclose(fp);

        // Decompress the JPEG data into an RGB image buffer
        int width, height, subsamp, color_space;
        if(tjDecompressHeader2(m_jpegDecompressor, jpegBuf, jpegSize, &width, &height, &color_space) != 0)
            std::cerr << "\n Jpeg image decode failed in tjDecompressHeader2";
        Rpp8u* rgbBuf;
        int elementsInRow;
        if(descPtr->c == 3)
        {
            elementsInRow = width * descPtr->c;
            rgbBuf= (Rpp8u*)calloc(width * height * 3, sizeof(Rpp8u));
            if(tjDecompress2(m_jpegDecompressor, jpegBuf, jpegSize, rgbBuf, width, width * 3, height, TJPF_RGB, TJFLAG_ACCURATEDCT) != 0)
                std::cerr << "\n Jpeg image decode failed ";
        }
        else
        {
            elementsInRow = width;
            rgbBuf= (Rpp8u*)calloc(width * height, sizeof(Rpp8u));
            if(tjDecompress2(m_jpegDecompressor, jpegBuf, jpegSize, rgbBuf, width, width, height, TJPF_GRAY, 0) != 0)
                std::cerr << "\n Jpeg image decode failed ";
        }
        // Copy the decompressed image buffer to the RPP input buffer
        Rpp8u *inputTemp = input + descPtr->offsetInBytes + (i * descPtr->strides.nStride);
        for (int j = 0; j < height; j++)
        {
            memcpy(inputTemp, rgbBuf + j * elementsInRow, elementsInRow * sizeof(Rpp8u));
            inputTemp += descPtr->w * descPtr->c;
        }
        // Clean up
        free(jpegBuf);
        free(rgbBuf);
    }

    // Clean up
    tjDestroy(m_jpegDecompressor);
}

// Read a batch of images using the OpenCV library
inline void read_image_batch_opencv(Rpp8u *input, RpptDescPtr descPtr, vector<string>::const_iterator imagesNamesStart)
{
    for(int i = 0; i < descPtr->n; i++)
    {
        Rpp8u *inputTemp = input + (i * descPtr->strides.nStride);
        string inputImagePath = *(imagesNamesStart + i);
        Mat image, imageBgr;
        if (descPtr->c == 3)
        {
            imageBgr = imread(inputImagePath, 1);
            cvtColor(imageBgr, image, COLOR_BGR2RGB);
        }
        else if (descPtr->c == 1)
            image = imread(inputImagePath, 0);

        int width = image.cols;
        int height = image.rows;
        Rpp32u elementsInRow = width * descPtr->c;
        Rpp8u *inputImage = image.data;
        for (int j = 0; j < height; j++)
        {
            memcpy(inputTemp, inputImage, elementsInRow * sizeof(Rpp8u));
            inputImage += elementsInRow;
            inputTemp += descPtr->w * descPtr->c;;
        }
    }
}

// Convert inputs to correponding bit depth specified by user
inline void convert_input_bitdepth(void *input, void *input_second, Rpp8u *inputu8, Rpp8u *inputu8Second, int inputBitDepth, Rpp64u ioBufferSize, Rpp64u inputBufferSize, RpptDescPtr srcDescPtr, bool dualInputCase, Rpp32f conversionFactor)
{
    if (inputBitDepth == 0 || inputBitDepth == 3 || inputBitDepth == 4)
    {
        memcpy(input, inputu8, inputBufferSize);
        if(dualInputCase)
            memcpy(input_second, inputu8Second, inputBufferSize);
    }
    else if (inputBitDepth == 1)
    {
        Rpp8u *inputTemp, *inputSecondTemp;
        Rpp16f *inputf16Temp, *inputf16SecondTemp;
        inputTemp = inputu8 + srcDescPtr->offsetInBytes;
        inputf16Temp = reinterpret_cast<Rpp16f *>(static_cast<Rpp8u *>(input) + srcDescPtr->offsetInBytes);
        for (int i = 0; i < ioBufferSize; i++)
            *inputf16Temp++ = static_cast<Rpp16f>((static_cast<float>(*inputTemp++)) * conversionFactor);

        if(dualInputCase)
        {
            inputSecondTemp = inputu8Second + srcDescPtr->offsetInBytes;
            inputf16SecondTemp = reinterpret_cast<Rpp16f *>(static_cast<Rpp8u *>(input_second) + srcDescPtr->offsetInBytes);
            for (int i = 0; i < ioBufferSize; i++)
                *inputf16SecondTemp++ = static_cast<Rpp16f>((static_cast<float>(*inputSecondTemp++)) * conversionFactor);
        }
    }
    else if (inputBitDepth == 2)
    {
        Rpp8u *inputTemp, *inputSecondTemp;
        Rpp32f *inputf32Temp, *inputf32SecondTemp;
        inputTemp = inputu8 + srcDescPtr->offsetInBytes;
        inputf32Temp = reinterpret_cast<Rpp32f *>(static_cast<Rpp8u *>(input) + srcDescPtr->offsetInBytes);
        for (int i = 0; i < ioBufferSize; i++)
            *inputf32Temp++ = (static_cast<Rpp32f>(*inputTemp++)) * conversionFactor;

        if(dualInputCase)
        {
            inputSecondTemp = inputu8Second + srcDescPtr->offsetInBytes;
            inputf32SecondTemp = reinterpret_cast<Rpp32f *>(static_cast<Rpp8u *>(input_second) + srcDescPtr->offsetInBytes);
            for (int i = 0; i < ioBufferSize; i++)
                *inputf32SecondTemp++ = (static_cast<Rpp32f>(*inputSecondTemp++)) * conversionFactor;
        }
    }
    else if (inputBitDepth == 5)
    {
        Rpp8u *inputTemp, *inputSecondTemp;
        Rpp8s *inputi8Temp, *inputi8SecondTemp;

        inputTemp = inputu8 + srcDescPtr->offsetInBytes;
        inputi8Temp = static_cast<Rpp8s *>(input) + srcDescPtr->offsetInBytes;
        for (int i = 0; i < ioBufferSize; i++)
            *inputi8Temp++ = static_cast<Rpp8s>((static_cast<Rpp32s>(*inputTemp++)) - 128);

        if(dualInputCase)
        {
            inputSecondTemp = inputu8Second + srcDescPtr->offsetInBytes;
            inputi8SecondTemp = static_cast<Rpp8s *>(input_second) + srcDescPtr->offsetInBytes;
            for (int i = 0; i < ioBufferSize; i++)
                *inputi8SecondTemp++ = static_cast<Rpp8s>((static_cast<Rpp32s>(*inputSecondTemp++)) - 128);
        }
    }
}

// sets values of maxHeight and maxWidth
inline void set_max_dimensions(vector<string>imagePaths, int& maxHeight, int& maxWidth, int imagesMixed)
{
    tjhandle tjInstance = tjInitDecompress();
    for (const std::string& imagePath : imagePaths)
    {
        FILE* jpegFile = fopen(imagePath.c_str(), "rb");
        if (!jpegFile) {
            std::cerr << "Error opening file: " << imagePath << std::endl;
            continue;
        }

        fseek(jpegFile, 0, SEEK_END);
        long fileSize = ftell(jpegFile);
        fseek(jpegFile, 0, SEEK_SET);

        std::vector<unsigned char> jpegBuffer(fileSize);
        fread(jpegBuffer.data(), 1, fileSize, jpegFile);
        fclose(jpegFile);

        int jpegSubsamp;
        int width, height;
        if (tjDecompressHeader2(tjInstance, jpegBuffer.data(), jpegBuffer.size(), &width, &height, &jpegSubsamp) == -1) {
            std::cerr << "Error decompressing file: " << imagePath << std::endl;
            continue;
        }

        if((maxWidth && maxWidth != width) || (maxHeight && maxHeight != height))
            imagesMixed = 1;

        maxWidth = max(maxWidth, width);
        maxHeight = max(maxHeight, height);
    }
    tjDestroy(tjInstance);
}

void search_files_recursive(const string& folder_path, vector<string>& imageNames, vector<string>& imageNamesPath, string extension)
{
    vector<string> entry_list;
    string full_path = folder_path;
    auto sub_dir = opendir(folder_path.c_str());
    if (!sub_dir)
    {
        std::cerr << "ERROR: Failed opening the directory at "<< folder_path << std::endl;
        exit(0);
    }

    struct dirent* entity;
    while ((entity = readdir(sub_dir)) != nullptr)
    {
        string entry_name(entity->d_name);
        if (entry_name == "." || entry_name == "..")
            continue;
        entry_list.push_back(entry_name);
    }
    closedir(sub_dir);
    sort(entry_list.begin(), entry_list.end());

    for (unsigned dir_count = 0; dir_count < entry_list.size(); ++dir_count)
    {
        string subfolder_path = full_path + "/" + entry_list[dir_count];
        fs::path pathObj(subfolder_path);
        if (fs::exists(pathObj) && fs::is_regular_file(pathObj))
        {
            // ignore files with extensions .tar, .zip, .7z
            auto file_extension_idx = subfolder_path.find_last_of(".");
            if (file_extension_idx != std::string::npos)
            {
                std::string file_extension = subfolder_path.substr(file_extension_idx+1);
                if ((file_extension == "tar") || (file_extension == "zip") || (file_extension == "7z") || (file_extension == "rar"))
                    continue;
            }
            if (entry_list[dir_count].size() > 4 && entry_list[dir_count].substr(entry_list[dir_count].size() - 4) == extension)
            {
                imageNames.push_back(entry_list[dir_count]);
                imageNamesPath.push_back(subfolder_path);
            }
        }
        else if (fs::exists(pathObj) && fs::is_directory(pathObj))
            open_folder(subfolder_path, imageNames, imageNamesPath, extension);
    }
}

//returns function type
inline string set_function_type(int layoutType, int pln1OutTypeCase, int outputFormatToggle, string backend)
{
    string funcType;
    if(layoutType == 0)
    {
        funcType = "Tensor_" + backend + "_PKD3";
        if (pln1OutTypeCase)
            funcType += "_toPLN1";
        else
        {
            if (outputFormatToggle)
                funcType += "_toPLN3";
            else
                funcType += "_toPKD3";
        }
    }
    else if (layoutType == 1)
    {
        funcType = "Tensor_" + backend + "_PLN3";
        if (pln1OutTypeCase)
            funcType += "_toPLN1";
        else
        {
            if (outputFormatToggle)
                funcType += "_toPKD3";
            else
                funcType += "_toPLN3";
        }
    }
    else
    {
       funcType = "Tensor_" + backend + "_PLN1";
       funcType += "_toPLN1";
    }

    return funcType;
}

// sets descriptor layout of src/dst
inline void set_descriptor_layout( RpptDescPtr srcDescPtr, RpptDescPtr dstDescPtr, int layoutType, bool pln1OutTypeCase, int outputFormatToggle)
{
    if(layoutType == 0)
    {
        srcDescPtr->layout = RpptLayout::NHWC;
        // Set src/dst layouts in tensor descriptors
        if (pln1OutTypeCase)
            dstDescPtr->layout = RpptLayout::NCHW;
        else
        {
            if (outputFormatToggle == 0)
                dstDescPtr->layout = RpptLayout::NHWC;
            else if (outputFormatToggle == 1)
                dstDescPtr->layout = RpptLayout::NCHW;
        }
    }
    else if(layoutType == 1)
    {
        srcDescPtr->layout = RpptLayout::NCHW;
        // Set src/dst layouts in tensor descriptors
        if (pln1OutTypeCase)
            dstDescPtr->layout = RpptLayout::NCHW;
        else
        {
            if (outputFormatToggle == 0)
                dstDescPtr->layout = RpptLayout::NCHW;
            else if (outputFormatToggle == 1)
                dstDescPtr->layout = RpptLayout::NHWC;
        }
    }
    else
    {
        // Set src/dst layouts in tensor descriptors
        srcDescPtr->layout = RpptLayout::NCHW;
        dstDescPtr->layout = RpptLayout::NCHW;
    }
}

// sets descriptor data types of src/dst
inline void set_descriptor_data_type(int ip_bitDepth, string &funcName, RpptDescPtr srcDescPtr, RpptDescPtr dstDescPtr)
{
    if (ip_bitDepth == 0)
    {
        funcName += "_u8_";
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::U8;
    }
    else if (ip_bitDepth == 1)
    {
        funcName += "_f16_";
        srcDescPtr->dataType = RpptDataType::F16;
        dstDescPtr->dataType = RpptDataType::F16;
    }
    else if (ip_bitDepth == 2)
    {
        funcName += "_f32_";
        srcDescPtr->dataType = RpptDataType::F32;
        dstDescPtr->dataType = RpptDataType::F32;
    }
    else if (ip_bitDepth == 3)
    {
        funcName += "_u8_f16_";
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::F16;
    }
    else if (ip_bitDepth == 4)
    {
        funcName += "_u8_f32_";
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::F32;
    }
    else if (ip_bitDepth == 5)
    {
        funcName += "_i8_";
        srcDescPtr->dataType = RpptDataType::I8;
        dstDescPtr->dataType = RpptDataType::I8;
    }
    else if (ip_bitDepth == 6)
    {
        funcName += "_u8_i8_";
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::I8;
    }
}

int main(int argc, char **argv) {

    RpptDesc srcDesc, dstDesc;
    RpptDescPtr srcDescPtr = &srcDesc;
    RpptDescPtr dstDescPtr = &dstDesc;

    char *src = argv[1];
    string dst = argv[2];
    unsigned int outputFormatToggle = atoi(argv[3]);
    int layoutType = atoi(argv[4]);

    string funcName = augmentationMap[38];
    int inputBitDepth = atoi(argv[5]);
    int batchSize = 1;
    int roiHeightList[batchSize], roiWidthList[batchSize];
    int maxHeight = 0, maxWidth = 0;
    int inputChannels = set_input_channels(layoutType);
    Rpp32u outputChannels = inputChannels;

    int roiList[4] = {0, 0, 0, 0};

    // Set ROI tensors types for src/dst

    vector<string> imageNames, imageNamesPath;
    search_files_recursive(src, imageNames, imageNamesPath, ".jpg");

    // Determine the type of function to be used based on the specified layout type
    string funcType = set_function_type(layoutType, 0, outputFormatToggle, "HIP");
    set_descriptor_layout( srcDescPtr, dstDescPtr, layoutType, 0, outputFormatToggle);
    set_descriptor_data_type(inputBitDepth, funcName, srcDescPtr, dstDescPtr);
    set_max_dimensions(imageNamesPath, maxHeight, maxWidth, 0);
    set_descriptor_dims_and_strides(srcDescPtr, batchSize, maxHeight, maxWidth, inputChannels, 0);
    set_descriptor_dims_and_strides(dstDescPtr, batchSize, maxHeight, maxWidth, outputChannels,0);

    void *input, *output, *input_second;
    void *d_input, *d_output;
    Rpp64u ioBufferSize = 0;
    Rpp64u oBufferSize = 0;

    // Set buffer sizes in pixels for src/dst
    ioBufferSize = (Rpp64u)srcDescPtr->h * (Rpp64u)srcDescPtr->w * (Rpp64u)srcDescPtr->c * (Rpp64u)batchSize;
    oBufferSize = (Rpp64u)dstDescPtr->h * (Rpp64u)dstDescPtr->w * (Rpp64u)dstDescPtr->c * (Rpp64u)batchSize;

    // Set buffer sizes in bytes for src/dst (including offsets)
    Rpp64u ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
    Rpp64u oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;
    Rpp64u inputBufferSize = ioBufferSize * get_size_of_data_type(srcDescPtr->dataType) + srcDescPtr->offsetInBytes;
    Rpp64u outputBufferSize = oBufferSize * get_size_of_data_type(dstDescPtr->dataType) + dstDescPtr->offsetInBytes;

    // Initialize 8u host buffers for src/dst
    Rpp8u *inputu8 = static_cast<Rpp8u *>(calloc(ioBufferSizeInBytes_u8, 1));
    Rpp8u *inputu8Second = static_cast<Rpp8u *>(calloc(ioBufferSizeInBytes_u8, 1));
    Rpp8u *outputu8 = static_cast<Rpp8u *>(calloc(oBufferSizeInBytes_u8, 1));

    input = static_cast<Rpp8u *>(calloc(inputBufferSize, 1));
    input_second = static_cast<Rpp8u *>(calloc(inputBufferSize, 1));
    output = static_cast<Rpp8u *>(calloc(outputBufferSize, 1));

    CHECK_RETURN_STATUS(hipMalloc(&d_input, inputBufferSize));
    CHECK_RETURN_STATUS(hipMalloc(&d_output, outputBufferSize));

    RpptROI *roiTensorPtrSrc, *roiTensorPtrDst;
    CHECK_RETURN_STATUS(hipHostMalloc(&roiTensorPtrSrc, batchSize * sizeof(RpptROI)));
    CHECK_RETURN_STATUS(hipHostMalloc(&roiTensorPtrDst, batchSize * sizeof(RpptROI)));
    RpptImagePatch *dstImgSizes;
    CHECK_RETURN_STATUS(hipHostMalloc(&dstImgSizes, batchSize * sizeof(RpptImagePatch)));

    RpptRoiType roiTypeSrc, roiTypeDst;
    roiTypeSrc = RpptRoiType::XYWH;
    roiTypeDst = RpptRoiType::XYWH;

    vector<string>::const_iterator imagesPathStart = imageNamesPath.begin();
    vector<string>::const_iterator imagesPathEnd = imagesPathStart + batchSize;
    vector<string>::const_iterator imageNamesStart = imageNames.begin();
    vector<string>::const_iterator imageNamesEnd = imageNamesStart + batchSize;

    // Set ROIs for src/dst
    set_src_and_dst_roi(imagesPathStart, imagesPathEnd, roiTensorPtrSrc, roiTensorPtrDst, dstImgSizes);
    read_image_batch_turbojpeg(inputu8, srcDescPtr, imagesPathStart);
    if (layoutType == 1)
        convert_pkd3_to_pln3(inputu8, srcDescPtr);

    bool dualInputCase = false;
    Rpp32f conversionFactor = 1.0f;
    Rpp32f invConversionFactor = 1.0f / conversionFactor;
    convert_input_bitdepth(input, input_second, inputu8, inputu8Second, inputBitDepth, ioBufferSize, inputBufferSize, srcDescPtr, dualInputCase, conversionFactor);
    CHECK_RETURN_STATUS(hipMemcpy(d_input, input, inputBufferSize, hipMemcpyHostToDevice));
    CHECK_RETURN_STATUS(hipMemcpy(d_output, output, outputBufferSize, hipMemcpyHostToDevice));

    for(int i = 0; i < batchSize; i++)
    {
        roiList[0] = 10;
        roiList[1] = 10;
        roiWidthList[i] = roiTensorPtrSrc[i].xywhROI.roiWidth / 2;
        roiHeightList[i] = roiTensorPtrSrc[i].xywhROI.roiHeight / 2;
    }


    Rpp32f multiplier[batchSize * srcDescPtr->c];
    Rpp32f offset[batchSize * srcDescPtr->c];
    Rpp32u mirror[batchSize];
    if (srcDescPtr->c == 3)
    {
        Rpp32f meanParam[3] = { 60.0f, 80.0f, 100.0f };
        Rpp32f stdDevParam[3] = { 0.9f, 0.9f, 0.9f };
        Rpp32f offsetParam[3] = { - meanParam[0] / stdDevParam[0], - meanParam[1] / stdDevParam[1], - meanParam[2] / stdDevParam[2] };
        Rpp32f multiplierParam[3] = {  1.0f / stdDevParam[0], 1.0f / stdDevParam[1], 1.0f / stdDevParam[2] };

        int i = 0;
        int j = 0;
        for (i = 0, j = 0; i < batchSize; i++, j += 3)
        {
            multiplier[j] = multiplierParam[0];
            offset[j] = offsetParam[0];
            multiplier[j + 1] = multiplierParam[1];
            offset[j + 1] = offsetParam[1];
            multiplier[j + 2] = multiplierParam[2];
            offset[j + 2] = offsetParam[2];
            mirror[i] = 1;
        }
    }
    else if(srcDescPtr->c == 1)
    {
        Rpp32f meanParam = 100.0f;
        Rpp32f stdDevParam = 0.9f;
        Rpp32f offsetParam = - meanParam / stdDevParam;
        Rpp32f multiplierParam = 1.0f / stdDevParam;

        for (int i = 0; i < batchSize; i++)
        {
            multiplier[i] = multiplierParam;
            offset[i] = offsetParam;
            mirror[i] = 1;
        }
    }

    for (int i = 0; i < batchSize; i++)
    {
        roiTensorPtrDst[i].xywhROI.xy.x = roiList[0];
        roiTensorPtrDst[i].xywhROI.xy.y = roiList[1];
        dstImgSizes[i].width = roiTensorPtrDst[i].xywhROI.roiWidth = roiWidthList[i];
        dstImgSizes[i].height = roiTensorPtrDst[i].xywhROI.roiHeight = roiHeightList[i];
    }

    rppHandle_t handle;
    hipStream_t stream;
    CHECK_RETURN_STATUS(hipStreamCreate(&stream));
    rppCreateWithStreamAndBatchSize(&handle, stream, batchSize);

    // Note: Shift is 128 to center at 128 instead of 0, since centering around
    // 0 will cause underflow on uint8_t.
    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 3 || inputBitDepth == 4 || inputBitDepth == 5)
        rppt_crop_mirror_normalize_gpu(d_input, srcDescPtr, d_output, dstDescPtr, offset, multiplier, mirror, roiTensorPtrDst, roiTypeSrc, handle);

    CHECK_RETURN_STATUS(hipDeviceSynchronize());
    CHECK_RETURN_STATUS(hipMemcpy(output, d_output, outputBufferSize, hipMemcpyDeviceToHost));
    convert_output_bitdepth_to_u8(output, outputu8, inputBitDepth, oBufferSize, outputBufferSize, dstDescPtr, invConversionFactor);


    if (roiTypeSrc == RpptRoiType::LTRB)
        convert_roi(roiTensorPtrDst, RpptRoiType::XYWH, dstDescPtr->n);
    RpptROI roiDefault;
    RpptROIPtr roiPtrDefault = &roiDefault;
    roiPtrDefault->xywhROI =  {0, 0, static_cast<Rpp32s>(dstDescPtr->w), static_cast<Rpp32s>(dstDescPtr->h)};
    for (int i = 0; i < dstDescPtr->n; i++)
    {
        roiTensorPtrDst[i].xywhROI.roiWidth = std::min(roiPtrDefault->xywhROI.roiWidth - roiTensorPtrDst[i].xywhROI.xy.x, roiTensorPtrDst[i].xywhROI.roiWidth);
        roiTensorPtrDst[i].xywhROI.roiHeight = std::min(roiPtrDefault->xywhROI.roiHeight - roiTensorPtrDst[i].xywhROI.xy.y, roiTensorPtrDst[i].xywhROI.roiHeight);
        roiTensorPtrDst[i].xywhROI.xy.x = std::max(roiPtrDefault->xywhROI.xy.x, roiTensorPtrDst[i].xywhROI.xy.x);
        roiTensorPtrDst[i].xywhROI.xy.y = std::max(roiPtrDefault->xywhROI.xy.y, roiTensorPtrDst[i].xywhROI.xy.y);
    } 
    if (layoutType == 0 || layoutType == 1)
    {
        if ((dstDescPtr->c == 3) && (dstDescPtr->layout == RpptLayout::NCHW))
            convert_pln3_to_pkd3(outputu8, dstDescPtr);
    }
    write_image_batch_opencv(dst, outputu8, dstDescPtr, imageNamesStart, dstImgSizes, MAX_IMAGE_DUMP);
    rppDestroyGPU(handle);
    CHECK_RETURN_STATUS(hipHostFree(roiTensorPtrSrc));
    CHECK_RETURN_STATUS(hipHostFree(roiTensorPtrDst));
    CHECK_RETURN_STATUS(hipHostFree(dstImgSizes));
    free(input);
    free(input_second);
    free(output);
    free(inputu8);
    free(inputu8Second);
    free(outputu8);
    CHECK_RETURN_STATUS(hipFree(d_input));
    CHECK_RETURN_STATUS(hipFree(d_output));
}