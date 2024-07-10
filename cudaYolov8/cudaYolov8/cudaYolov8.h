#pragma once

#include <iostream>
#include <map>
#include "include/config.h"
#include "include/model.h"
#include "include/utils.h"
#include "include/process.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>


#ifdef DLL_CUDAYOLOV8
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT __declspec(dllimport)
#endif

class DLL_EXPORT cudaYolov8
{
public:
	
	cudaYolov8(const std::string engineFilePath, const std::string class_txt);

	~cudaYolov8();


public:

	int Inference(cv::Mat& image);   // 返回耗时
	void serializeEngine(const int& kBatchSize, std::string& wts_name, std::string& engine_name, std::string& sub_type);

	
protected:
	void init(std::string class_file);  // init cuda


private:

	nvinfer1::IRuntime* runtime;
	nvinfer1::ICudaEngine* engine;
	nvinfer1::IExecutionContext* context;
	cudaStream_t stream;

	float* device_buffers[2];          
	float* output_buffer_host;
	std::map<int, std::string> labels;    // label

	uint8_t* image_device;
    int kOutputSize;

	std::string engine_file;               //  模型地址
	std::string class_file;                //  类别地址

};

