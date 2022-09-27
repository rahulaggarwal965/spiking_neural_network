#define STB_IMAGE_IMPLEMENTATION

#include "snn_internal.h"
#include "stb/stb_image.h"

void generate_spikes() {

    int width, height, channels;
    u8 *image_data = stbi_load("/home/infinity/Downloads/mnist_png/mnist_png/testing/3/18.png", &width, &height, &channels, 1);

    if (image_data == NULL) {
        return;
    }



}


