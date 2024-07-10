
#include "cudaYolov8.h"



Logger gLogger;

cudaYolov8::cudaYolov8(const std::string engineFilePath, const std::string class_txt) 
    : engine_file(engineFilePath), class_file(class_txt)
    , runtime(nullptr)
    , engine(nullptr)
    , context(nullptr)  
{
 
    readEngineFile(engine_file, runtime, engine, context);

    // 创建CUDA流
    cudaStreamCreate(&stream);
    init(class_txt);
}

cudaYolov8::~cudaYolov8() {

    // 销毁CUDA流
    cudaStreamDestroy(stream);
    // 释放TensorRT资源
    delete context;
    delete engine;
    delete runtime;

    //Release stream and buffers
    cudaFree(device_buffers[0]);
    cudaFree(device_buffers[1]);
    delete[] output_buffer_host;
    
}


void cudaYolov8::serializeEngine(const int& kBatchSize, std::string& wts_name, std::string& engine_name, std::string& sub_type) {
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    nvinfer1::IHostMemory* serialized_engine = nullptr;


    if (sub_type == "n") {
        serialized_engine = buildEngineYolov8n(kBatchSize, builder, config, nvinfer1::DataType::kFLOAT, wts_name);
    }
    else if (sub_type == "s") {
        serialized_engine = buildEngineYolov8s(kBatchSize, builder, config, nvinfer1::DataType::kFLOAT, wts_name);
    }
    else if (sub_type == "m") {
        serialized_engine = buildEngineYolov8m(kBatchSize, builder, config, nvinfer1::DataType::kFLOAT, wts_name);
    }
    else if (sub_type == "l") {
        serialized_engine = buildEngineYolov8l(kBatchSize, builder, config, nvinfer1::DataType::kFLOAT, wts_name);
    }
    else if (sub_type == "x") {
        serialized_engine = buildEngineYolov8x(kBatchSize, builder, config, nvinfer1::DataType::kFLOAT, wts_name);
    }

    assert(serialized_engine);
    std::ofstream p(engine_name, std::ios::binary);
    if (!p) {
        std::cout << "could not open plan output file" << std::endl;
        assert(false);
    }
    p.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());

    delete builder;
    delete config;
    delete serialized_engine;
}

void cudaYolov8::init(std::string class_file) {

    kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

    output_buffer_host = new float[kBatchSize * kOutputSize];
    assert(engine->getNbBindings() == 2);
    const int inputIndex = engine->getBindingIndex(kInputTensorName);
    const int outputIndex = engine->getBindingIndex(kOutputTensorName);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    cudaMalloc((void**)&image_device, kMaxInputImageSize * 3);
    cudaMalloc((void**)&device_buffers[0], kBatchSize * 3 * kInputH * kInputW * sizeof(float));
    cudaMalloc((void**)&device_buffers[1], kBatchSize * kOutputSize * sizeof(float));

    readClassFile(class_file, this->labels);

}

int cudaYolov8::Inference(cv::Mat& image){
    auto t_beg = std::chrono::high_resolution_clock::now();
      
    float scale = 1.0;
    int img_size = image.cols * image.rows * 3;
    cudaMemcpyAsync(image_device, image.data, img_size, cudaMemcpyHostToDevice, stream);
    preprocess(image_device, image.cols, image.rows, device_buffers[0], kInputW, kInputH, stream, scale);
    context->enqueue(kBatchSize, (void**)device_buffers, stream, nullptr);
    cudaMemcpyAsync(output_buffer_host, device_buffers[1], kBatchSize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    std::vector<Detection> res;
    NMS(res, output_buffer_host, kConfThresh, kNmsThresh);
    drawBbox(image, res, scale, labels);

    auto t_end = std::chrono::high_resolution_clock::now();
    float total_inf = std::chrono::duration<float, std::milli>(t_end - t_beg).count();
    //std::cout << "Inference time: " << int(total_inf) << std::endl;
    return int(total_inf);
}
