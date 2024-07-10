#define USE_FP16
//#define USE_INT8

#ifndef CONFIG_PARA
#define CONFIG_PARA

const static char* kInputTensorName = "images";
const static char* kOutputTensorName = "output";

const static int kNumClass = 80;
const static int kBatchSize = 1;

const static int kInputH = 640;
const static int kInputW = 640;

const static float kNmsThresh = 0.45f;
const static float kConfThresh = 0.5f;

const static int kMaxInputImageSize = 3000*3000;
const static int kMaxNumOutputBbox = 1000;

struct alignas(float) Detection {
  float bbox[4];
  float conf;
  float class_id;
};

#endif