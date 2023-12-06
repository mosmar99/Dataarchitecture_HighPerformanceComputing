#include <jetson-utils/videoSource.h>
#include <jetson-utils/videoOutput.h>

typedef unsigned int hist_t;
// https://forums.developer.nvidia.com/t/gstreamer-gstdecoder-failed-to-retrieve-next-image-buffer/195168

__global__ void calcHistogramKernel(uchar4* gray_image, hist_t* histogram_device, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // calculate index
    int index = threadIdx.y * blockDim.x + threadIdx.x;

    __shared__ hist_t histo_local[256]; 
    histo_local[index] = 0;
    __syncthreads();
 
    // calculate histogram_device
    if (x < width && y < height) {
        unsigned char gray = gray_image[y*width + x].x;
        atomicAdd(&(histo_local[gray]), 1);
    }
    __syncthreads();

    // write to global memory   
    atomicAdd(&(histogram_device[index]), histo_local[index]);
    __syncthreads();
}

__global__ void plotHistogramKernel(uchar4* image, unsigned int* histogram_device, int width, int height, int max_freq)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    uchar4 white_pixel = make_uchar4(255, 255, 255, 255);
    uchar4 black_pixel = make_uchar4(0, 0, 0, 255);
    unsigned char semi_black_val = 0;

    if (index < 256)
    {
        int freq = histogram_device[index] * 256 / max_freq;
        for (int i = 0; i < 256; i++)
        {
            int row = height - i - 1;
            semi_black_val = image[row * width + 2*index].x / 4;
            black_pixel = make_uchar4(semi_black_val, semi_black_val, semi_black_val, 255);
            if ( i <= freq) {
                image[row * width + 2*index]   = white_pixel;
                image[row * width + 2*index+1] = white_pixel;
            }            
            else
            {
                image[row * width + 2*index]   = black_pixel;
                image[row * width + 2*index+1] = black_pixel;
            }  
        }
    }
}

// CUDA kernel for RGB to grayscale conversion
__global__ void rgb2grayKernel(uchar4* in_image, uchar4* out_image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index = y * width + x;
        uchar4 pixel = in_image[index];

        // Calculate grayscale value
        unsigned char grayscale = 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;

        // Create a new uchar4 with the grayscale value for all components
        out_image[index] = make_uchar4(grayscale, grayscale, grayscale, 255); // Assuming 255 for alpha (fully opaque)
    }
}

int main(int argc, char** argv) {
    // create input/output streams
    videoSource* input = videoSource::Create(argc, argv, ARG_POSITION(0));
    videoOutput* output2 = videoOutput::Create(argc, argv, ARG_POSITION(2));

    uchar4* out_image = NULL;
    uint32_t N = input->GetWidth() * input->GetHeight();
    cudaMalloc(&out_image, N*sizeof(uchar4));

    hist_t *histogram_device = NULL;
    cudaMalloc(&histogram_device, 256*sizeof(hist_t));
    hist_t histogram_host[256] = {0};

    unsigned int max_freq = 20000;

    if (!input)
        return 0;

    // capture/display loop
    while (true) {
        uchar4* in_image = nullptr;
        int status = 0;

        if (!input->Capture(&in_image, 1000, &status)) {
            if (status == videoSource::TIMEOUT)
                continue;
            // 1000ms timeout (default)
            break; // EOS
        }

        if (output2 != nullptr) {
            // Define grid and block size for CUDA kernel launch
            dim3 blockDim(16, 16);
            dim3 gridDim((input->GetWidth() + blockDim.x - 1) / blockDim.x, (input->GetHeight() + blockDim.y - 1) / blockDim.y);

            // Launch the grayscale conversion kernel
            rgb2grayKernel<<<gridDim, blockDim>>>(in_image, out_image, input->GetWidth(), input->GetHeight());
           
            // calculate histogram_device of grayscale image
            memset(histogram_host, 0, 256*sizeof(hist_t));
            cudaMemcpy(histogram_device, histogram_host, 256*sizeof(hist_t), cudaMemcpyHostToDevice);
            calcHistogramKernel<<<gridDim, blockDim>>>(out_image, histogram_device, input->GetWidth(), input->GetHeight());
            cudaMemcpy(histogram_host, histogram_device, 256*sizeof(hist_t), cudaMemcpyDeviceToHost);

            // // print histogram_host
            // for (int i = 0; i < 256; i++)
            // {
            //     printf("%d ", histogram_host[i]);
            // }
            // printf("\n");

            // plot histogram
            plotHistogramKernel<<<1, 256>>>(out_image, histogram_device, input->GetWidth(), input->GetHeight(), max_freq);

            // output final image 
            output2->Render(out_image, input->GetWidth(), input->GetHeight());

            // Update status bar
            char str[256];
            sprintf(str, "Camera Viewer (%ux%u) | %0.1f FPS", input->GetWidth(), input->GetHeight(), output2->GetFrameRate());
            output2->SetStatus(str);

            if (!output2->IsStreaming()) // check if the user quit
                break;            
        }
    }

    // destroy resources
    delete input;
    delete output2;

    return 0;
}