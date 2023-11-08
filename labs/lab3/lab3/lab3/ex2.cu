#include <jetson-utils/videoSource.h>
#include <jetson-utils/videoOutput.h>

// CUDA kernel for RGB to grayscale conversion
__global__ void rgb2grayKernel(uchar4* image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index = y * width + x;
        uchar4 pixel = image[index];

        // Calculate grayscale value
        unsigned char grayscale = 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;

        // Create a new uchar4 with the grayscale value for all components
        image[index] = make_uchar4(grayscale, grayscale, grayscale, 255); // Assuming 255 for alpha (fully opaque)
    }
}

int main(int argc, char** argv) {
    // create input/output streams
    videoSource* input = videoSource::Create(argc, argv, ARG_POSITION(0));
    videoOutput* output = videoOutput::Create(argc, argv, ARG_POSITION(1));

    if (!input)
        return 0;

    // capture/display loop
    while (true) {
        uchar4* image = nullptr;
        int status = 0;

        if (!input->Capture(&image, 1000, &status)) {
            if (status == videoSource::TIMEOUT)
                continue;
            // 1000ms timeout (default)
            break; // EOS
        }

        if (output != nullptr) {
            // Define grid and block size for CUDA kernel launch
            dim3 blockDim(16, 16);
            dim3 gridDim((input->GetWidth() + blockDim.x - 1) / blockDim.x, (input->GetHeight() + blockDim.y - 1) / blockDim.y);

            // Launch the grayscale conversion kernel
            rgb2grayKernel<<<gridDim, blockDim>>>(image, input->GetWidth(), input->GetHeight());

            output->Render(image, input->GetWidth(), input->GetHeight());

            // Update status bar
            char str[256];
            sprintf(str, "Camera Viewer (%ux%u) | %0.1f FPS", input->GetWidth(), input->GetHeight(), output->GetFrameRate());
            output->SetStatus(str);

            if (!output->IsStreaming()) // check if the user quit
                break;
        }
    }

    return 0;
}
