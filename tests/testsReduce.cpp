/*! \file testsReduce.cpp
 *  \brief Google Test Unit Tests for the `Reduce` kernels.
 *  \note Use the `--profiling` flag to enable profiling of the kernels.
 *  \note The benchmarks in these tests are against naive CPU implementations 
 *        of the associated algorithms. They are used only for testing purposes, 
 *        and not for examining the performance of their GPU alternatives.
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
#include <chrono>
#include <random>
#include <limits>
#include <cmath>
#include <gtest/gtest.h>
#include <CLUtils.hpp>
#include <RBC/algorithms.hpp>
#include <RBC/tests/helper_funcs.hpp>


// Kernel filenames
const std::string kernel_filename_reduce { "kernels/reduce_kernels.cl" };

// Uniform random number generators
namespace RBC
{
    extern std::function<unsigned char ()> rNum_0_255;
    extern std::function<unsigned short ()> rNum_0_10000;
    extern std::function<float ()> rNum_R_0_1;
}

bool profiling;  // Flag to enable profiling of the kernels (--profiling)


/*! \brief Tests the **reduce_min** kernel.
 *  \details The kernel computes the minimum element of each row of an array. 
 */
TEST (Reduce, reduce_min)
{
    try
    {
        const unsigned int rows = 1024;
        const unsigned int cols = 1024;
        const unsigned int bufferInSize = cols * rows * sizeof (cl_float);
        const unsigned int bufferOutSize = rows * sizeof (cl_float);

        // Setup the OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addProgram (0, kernel_filename_reduce);

        // Configure kernel execution parameters
        clutils::CLEnvInfo<1> info (0, 0, 0, { 0 }, 0);
        cl_algo::RBC::Reduce<cl_algo::RBC::ReduceConfig::MIN, cl_float> rMin (clEnv, info);
        rMin.init (cols, rows);

        // Initialize data (writes on staging buffer directly)
        std::generate (rMin.hPtrIn, rMin.hPtrIn + bufferInSize / sizeof (cl_float), RBC::rNum_R_0_1);
        // RBC::printBufferF ("Original:", rMin.hPtrIn, cols, rows, 3);

        rMin.write ();  // Copy data to device
        
        rMin.run ();  // Execute kernels (~ 45 us)
        
        cl_float *results = (cl_float *) rMin.read ();  // Copy results to host
        // RBC::printBufferF ("Received:", results, 1, rows, 3);

        // Produce reference array of distances
        cl_float *refMin = new cl_float[rows];
        auto func = [](cl_float a, cl_float b) -> bool { return a < b; };
        RBC::cpuReduce<cl_float> (rMin.hPtrIn, refMin, cols, rows, func);
        // RBC::printBufferF ("Expected:", refMin, 1, rows, 3);

        // Verify blurred output
        float eps = std::numeric_limits<float>::epsilon ();  // 1.19209e-07
        for (uint i = 0; i < rows; ++i)
            ASSERT_LT (std::abs (refMin[i] - results[i]), eps);

        // Profiling ===========================================================
        if (profiling)
        {
            const int nRepeat = 1;  /* Number of times to perform the tests. */

            // CPU
            clutils::CPUTimer<double, std::milli> cTimer;
            clutils::ProfilingInfo<nRepeat> pCPU ("CPU");
            for (int i = 0; i < nRepeat; ++i)
            {
                cTimer.start ();
                RBC::cpuReduce<cl_float> (rMin.hPtrIn, refMin, cols, rows, func);
                pCPU[i] = cTimer.stop ();
            }
            
            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = rMin.run (gTimer);

            // Benchmark
            pGPU.print (pCPU, "Reduce<MIN>");
        }

    }
    catch (const cl::Error &error)
    {
        std::cerr << error.what ()
                  << " (" << clutils::getOpenCLErrorCodeString (error.err ()) 
                  << ")"  << std::endl;
        exit (EXIT_FAILURE);
    }
}


/*! \brief Tests the **reduce_max** kernel.
 *  \details The kernel computes the maximum element of each row of an array. 
 */
TEST (Reduce, reduce_max)
{
    try
    {
        const unsigned int rows = 1024;
        const unsigned int cols = 1024;
        const unsigned int bufferInSize = cols * rows * sizeof (cl_uint);
        const unsigned int bufferOutSize = rows * sizeof (cl_uint);

        // Setup the OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addProgram (0, kernel_filename_reduce);

        // Configure kernel execution parameters
        clutils::CLEnvInfo<1> info (0, 0, 0, { 0 }, 0);
        cl_algo::RBC::Reduce<cl_algo::RBC::ReduceConfig::MAX, cl_uint> rMax (clEnv, info);
        rMax.init (cols, rows);

        // Initialize data (writes on staging buffer directly)
        std::generate (rMax.hPtrIn, rMax.hPtrIn + bufferInSize / sizeof (cl_uint), RBC::rNum_R_0_1);
        // RBC::printBuffer ("Original:", rMax.hPtrIn, cols, rows);

        rMax.write ();  // Copy data to device
        
        rMax.run ();  // Execute kernels (~ 44 us)
        
        cl_uint *results = (cl_uint *) rMax.read ();  // Copy results to host
        // RBC::printBuffer ("Received:", results, 1, rows);

        // Produce reference array of distances
        cl_uint *refMax = new cl_uint[rows];
        auto func = [](cl_uint a, cl_uint b) -> bool { return a > b; };
        RBC::cpuReduce<cl_uint> (rMax.hPtrIn, refMax, cols, rows, func);
        // RBC::printBuffer ("Expected:", refMax, 1, rows);

        // Verify blurred output
        float eps = std::numeric_limits<float>::epsilon ();  // 1.19209e-07
        for (uint i = 0; i < rows; ++i)
            ASSERT_LT (std::abs (refMax[i] - results[i]), eps);

        // Profiling ===========================================================
        if (profiling)
        {
            const int nRepeat = 1;  /* Number of times to perform the tests. */

            // CPU
            clutils::CPUTimer<double, std::milli> cTimer;
            clutils::ProfilingInfo<nRepeat> pCPU ("CPU");
            for (int i = 0; i < nRepeat; ++i)
            {
                cTimer.start ();
                RBC::cpuReduce<cl_uint> (rMax.hPtrIn, refMax, cols, rows, func);
                pCPU[i] = cTimer.stop ();
            }
            
            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = rMax.run (gTimer);

            // Benchmark
            pGPU.print (pCPU, "Reduce<MAX>");
        }

    }
    catch (const cl::Error &error)
    {
        std::cerr << error.what ()
                  << " (" << clutils::getOpenCLErrorCodeString (error.err ()) 
                  << ")"  << std::endl;
        exit (EXIT_FAILURE);
    }
}


int main (int argc, char **argv)
{
    profiling = RBC::setProfilingFlag (argc, argv);

    ::testing::InitGoogleTest (&argc, argv);

    return RUN_ALL_TESTS ();
}
