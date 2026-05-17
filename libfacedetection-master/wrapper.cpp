#include "facedetectcnn.h"
extern "C" {
    int* facedetect_cnn_c(unsigned char* result_buffer, unsigned char* rgb_image_data, int width, int height, int step) {
        return facedetect_cnn(result_buffer, rgb_image_data, width, height, step);
    }
}
