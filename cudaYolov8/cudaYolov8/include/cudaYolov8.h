#pragma once
#include "pch.h"
#include <iostream>


class __declspec(dllexport) cudaYolov8
{
public:
	//cudaYolov8();
	//~cudaYolov8();

	enum class detType
	{
		image = 0,   // det Pic
		video = 1,     // det Video
		camera = 2,     // det Camera
	};


public:

	void Inference(std::string& engine_name, std::string& class_file);
	void serializeEngine(const int& kBatchSize, std::string& wts_name, std::string& engine_name, std::string& sub_type);

	int runInfer(std::string enginePath, std::string classFiles, detType detT);

};

