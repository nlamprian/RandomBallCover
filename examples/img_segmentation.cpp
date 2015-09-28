/*! \file img_segmentation.cpp
 *  \brief An application examining the potential of the `Random Ball Cover` 
 *         data structure for image segmentation.
 *  \details This application was developed for testing the eligibility of `RBC` for 
 *           image segmentation. Try changing the `step` variable for configuring the 
 *           number of representatives, and/or the `weight` variable for adjusting the 
 *           influence of the color intensity, and examine the effects on the image. 
 *           By creating a graph with the representatives, and adjusting the edge weights, 
 *           connected components could be created that would define areas of interest.
 *  \author Nick Lamprianidis
 *  \version 1.2.1
 *  \date 2015
 *  \copyright The MIT License (MIT)
 *  \par
 *  Copyright (c) 2015 Nick Lamprianidis
 *  \par
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *  \par
 *  The above copyright notice and this permission notice shall be included in
 *  all copies or substantial portions of the Software.
 *  \par
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include <CLUtils.hpp>
#include <RBC/algorithms.hpp>
#include <RBC/tests/helper_funcs.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


// Kernel filenames
const std::vector<std::string> kernel_files = { "kernels/rbc_kernels.cl", 
                                                "kernels/scan_kernels.cl" };


int main (int argc, const char **argv)
{
    try
    {
        cv::Mat image = cv::imread ("../data/demo2.jpg", CV_LOAD_IMAGE_GRAYSCALE);

        const unsigned int step = 16;
        const float weight = 3.f;
        const unsigned int cols = image.cols;
        const unsigned int rows = image.rows;
        const unsigned int nx = cols * rows;
        const unsigned int rmd = 64 - ((cols / step) * (rows / step)) % 64;
        const unsigned int nr = (cols / step) * (rows / step) + rmd;  // must be multiple of 64
        const unsigned int d = 4;

        // Display original image
        cv::namedWindow ("Original", cv::WINDOW_AUTOSIZE);
        cv::imshow ("Original", image);
        cv::moveWindow ("Original", 0, 0);

        // Setup the OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addProgram (0, kernel_files);

        // Configure kernel execution parameters
        clutils::CLEnvInfo<1> info (0, 0, 0, { 0 }, 0);
        const cl_algo::RBC::KernelTypeC K = cl_algo::RBC::KernelTypeC::SHARED_NONE;
        const cl_algo::RBC::RBCPermuteConfig P = cl_algo::RBC::RBCPermuteConfig::GENERIC;
        cl_algo::RBC::RBCConstruct<K, P> rbc (clEnv, info);
        rbc.init (nx, nr, d);

        // Initialize data (writes on staging buffers directly)
        unsigned int idx = 0;
        std::transform (image.data, image.data + nx, (cl_float4 *) rbc.hPtrInX, 
            [&](unsigned char val)
            {
                cl_float p[4];
                p[0] = idx % cols; p[1] = idx / cols;  // Coordinates (x, y)
                p[2] = val / weight;  p[3] = 0.f;      // Intensity and dummy dimension
                idx++;
                return *((cl_float4 *) p);
            }
        );
        
        idx = 0;
        std::copy_if ((cl_float4 *) rbc.hPtrInX, (cl_float4 *) rbc.hPtrInX + nx, (cl_float4 *) rbc.hPtrInR, 
            [&](const cl_float4 &val) -> bool  // One representative every #step pixels
            {
                unsigned int x = idx % cols, y = idx / cols;
                idx++;
                return ((x + step / 2) % step == 0) && ((y + step / 2) % step == 0);
            }
        );
        for (int j = d * (nr - rmd); j < d * nr; ++j)
            rbc.hPtrInR[j] = std::numeric_limits<cl_float>::infinity ();

        // Copy data to device
        rbc.write (cl_algo::RBC::RBCConstruct<K, P>::Memory::D_IN_X);
        rbc.write (cl_algo::RBC::RBCConstruct<K, P>::Memory::D_IN_R);
        
        rbc.run ();  // Execute kernels (~ 26 ms)
        
        // Copy results to host
        rbc_dist_id *ID = (rbc_dist_id *) rbc.read (cl_algo::RBC::RBCConstruct<K, P>::Memory::H_OUT_ID);

        // Assign to each pixel the intensity of its representative
        idx = 0;
        std::transform ((cl_float4 *) rbc.hPtrInX, (cl_float4 *) rbc.hPtrInX + nx, image.data, 
            [&](const cl_float4 &val)
            {
                return (unsigned char) (weight * rbc.hPtrInR[ID[idx++].id * d + 2]);
            }
        );

        // Display processed image
        cv::namedWindow ("Image Segmentation", cv::WINDOW_AUTOSIZE);
        cv::imshow ("Image Segmentation", image);
        cv::moveWindow ("Image Segmentation", 640, 0);
        
        cv::waitKey (0);
    }
    catch (const cl::Error &error)
    {
        std::cerr << error.what ()
                  << " (" << clutils::getOpenCLErrorCodeString (error.err ()) 
                  << ")"  << std::endl;
        exit (EXIT_FAILURE);
    }

    return 0;
}
