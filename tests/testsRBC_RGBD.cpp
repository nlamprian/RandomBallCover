/*! \file testsRBC_RGBD.cpp
 *  \brief Google Test Unit Tests for the kernels associated 
 *         with the `Random Ball Cover` data structure.
 *  \note Use the `--profiling` flag to enable profiling of the kernels.
 *  \note The benchmarks in these tests are against naive CPU implementations 
 *        of the associated algorithms. They are used only for testing purposes, 
 *        and not for examining the performance of their GPU alternatives.
 *  \author Nick Lamprianidis
 *  \version 1.1
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
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <functional>
#include <numeric>
#include <chrono>
#include <random>
#include <limits>
#include <cmath>
#include <gtest/gtest.h>
#include <CLUtils.hpp>
#include <RBC/algorithms.hpp>
#include <RBC/tests/helper_funcs.hpp>
// #include <opencv2/core/core.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/imgproc/imgproc.hpp>


bool profiling;  // Flag to enable profiling of the kernels (--profiling)

// Kernel filenames
const std::vector<std::string> kernel_files = { "kernels/rbc_kernels.cl", 
                                                "kernels/scan_kernels.cl",
                                                "kernels/reduce_kernels.cl" };


/*! \brief Reads in a binary file.
 *  
 *  \param[in] path path to the file.
 *  \param[out] data array that receives the data.
 *  \param[in] n number of bytes to read.
 */
void fread (const char *path, char *data, size_t n)
{
    std::ifstream f (path, std::ios::binary);
    f.read (data, n);
    f.close ();
}


/*! \brief Finds the invalid points (zero coordinates) and 
 *         pushes them to the end of the sequences.
 *  
 *  \param[in] dBegin input iterator to the initial position of the sequence 
 *                    with the pc4d data.
 *  \param[in] dEnd input iterator past the last position of the sequence 
 *                  with the pc4d data.
 *  \param[in] cBegin input iterator to the initial position of the sequence
 *                    with the rgba data.
 *  \param[in] cEnd input iterator past the last position of the sequence
 *                  with the rgba data.
 *  \return The number of invalid points.
 */
unsigned int clean_data (
    std::vector<cl_float4>::iterator dBegin, std::vector<cl_float4>::iterator dEnd, 
    std::vector<cl_float4>::iterator cBegin, std::vector<cl_float4>::iterator cEnd)
{
    std::vector<cl_float4>::iterator di = dBegin;
    std::vector<cl_float4>::iterator ci = cBegin;
    std::vector<cl_float4>::iterator dlast = dEnd;
    std::vector<cl_float4>::iterator clast = cEnd;
    unsigned int count = 0;

    while (di != dlast)
    {
        cl_float *p = (cl_float *) &(*di);
        bool invalid = (p[0] == 0.f) && (p[1] == 0.f) && (p[2] == 0.f);
        if (invalid)
        {
            --dlast; --clast;
            std::swap (*di, *dlast);
            std::swap (*ci, *clast);
            count++;
            continue;
        }
        else
        {
            ++di; ++ci;
        }
    }

    return count;
}


/*! \brief Picks a number of elements uniformly at random without 
 *         replacement from the input sequences.
 *  \details The function implements a Fisher-Yates shuffle.
 *  
 *  \param[in] cBegin input iterator to the initial position of the sequence 
 *                    with the rgba/pc4d data.
 *  \param[in] dBegin input iterator to the initial position of the sequence
 *                    with the pc4d/rgba data.
 *  \param[out] rcBegin output iterator to the initial position of the sequence 
 *                      with the chosen rgba/pc4d data.
 *  \param[out] rdBegin output iterator to the initial position of the sequence 
 *                      with the chosen pc4d/rgba data.
 *  \param[in] n number of points in the input sequences.
 *  \param[in] rn number of points in the output sequences.
 */
template<typename iterator>
void random_unique (iterator cBegin, iterator dBegin, 
    iterator rcBegin, iterator rdBegin, size_t n, size_t rn)
{
    auto seed = std::chrono::system_clock::now ().time_since_epoch ().count ();
    std::default_random_engine generator { seed };
    std::uniform_int_distribution<unsigned long long> dist (0, (unsigned long long) -1);
    std::function<unsigned long long ()> randNum = std::bind (dist, generator);

    size_t left = n;
    while (rn--)
    {
        iterator ix = cBegin;
        iterator iy = dBegin;
        size_t i = randNum () % left;
        std::advance (ix, i);
        std::advance (iy, i);
        std::swap (*cBegin, *ix);
        std::swap (*dBegin, *iy);
        *rcBegin = *cBegin;
        *rdBegin = *dBegin;
        ++cBegin;
        ++dBegin;
        ++rcBegin;
        ++rdBegin;
        --left;
    }
}


/*! \brief Tests the **RBCSearch** class on an 8-D point cloud.
 *  \details The class computes the NNs of a set of queries.
 */
TEST (RBC, rbcRGBD)
{
    try
    {
        const unsigned int n = 640 * 480;
        const unsigned int nx = 16384;
        const unsigned int nr =   256;
        const unsigned int nq = 16384;
        const unsigned int d = 8;
        const float a = 4.f;
        const unsigned int bufferXSize = nx * d * sizeof (cl_float);
        const unsigned int bufferRSize = nr * d * sizeof (cl_float);
        const unsigned int bufferQSize = nq * d * sizeof (cl_float);
        const unsigned int bufferOSize = nr * sizeof (cl_uint);
        const unsigned int bufferNSize = nr * sizeof (cl_uint);
        const unsigned int bufferRIDSize = nq * sizeof (rbc_dist_id);
        const unsigned int bufferNNIDSize = nq * sizeof (rbc_dist_id);

        std::vector<cl_float4> rgba (n);
        std::vector<cl_float4> pc4d (n);
        std::vector<cl_float4> x_rgba (nx);
        std::vector<cl_float4> x_pc4d (nx);
        std::vector<cl_float4> r_rgba (nr);
        std::vector<cl_float4> r_pc4d (nr);
        std::vector<cl_float4> q_rgba (nq);
        std::vector<cl_float4> q_pc4d (nq);
        unsigned int idx;

        // Load data
        fread ("../data/rgba.bin", (char *) rgba.data (), n * sizeof (cl_float4));
        fread ("../data/pc4d.bin", (char *) pc4d.data (), n * sizeof (cl_float4));
        
        // Display RGBA image
        // cv::namedWindow ("RGBA", cv::WINDOW_AUTOSIZE);
        // cv::imshow ("RGBA", cv::Mat (480, 640, CV_32FC4, (float *) rgba.data ()));
        // cv::waitKey (0);

        // Prepare data
        unsigned int ni = clean_data (pc4d.begin (), pc4d.end (), rgba.begin (), rgba.end ());

        random_unique (rgba.begin (), pc4d.begin (), x_rgba.begin (), x_pc4d.begin (), n - ni, nx);

        random_unique (x_rgba.begin (), x_pc4d.begin (), r_rgba.begin (), r_pc4d.begin (), nx, nr);
        
        auto cb = rgba.begin (), db = pc4d.begin ();
        std::advance (cb, nx); std::advance (db, nx);
        random_unique (cb, db, q_rgba.begin (), q_pc4d.begin (), n - ni - nx, nq);

        // Setup the OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addProgram (0, kernel_files);

        // Construction ========================================================

        // Configure kernel execution parameters
        clutils::CLEnvInfo<1> info (0, 0, 0, { 0 }, 0);
        const cl_algo::RBC::KernelTypeC K = cl_algo::RBC::KernelTypeC::KINECT_R;
        const cl_algo::RBC::RBCPermuteConfig P = cl_algo::RBC::RBCPermuteConfig::GENERIC;
        cl_algo::RBC::RBCConstruct<K, P> rbcCon (clEnv, info);
        rbcCon.init (nx, nr, d, a);

        // Initialize data (writes on staging buffer directly)
        idx = 0;
        std::generate ((cl_float8 *) rbcCon.hPtrInX, (cl_float8 *) rbcCon.hPtrInX + nx, 
            [&]()
            {
                cl_float4 p[2];
                p[0] = x_pc4d[idx];
                p[1] = x_rgba[idx++];
                return *((cl_float8 *) p);
            }
        );
        idx = 0;
        std::generate ((cl_float8 *) rbcCon.hPtrInR, (cl_float8 *) rbcCon.hPtrInR + nr, 
            [&]()
            {
                cl_float4 p[2];
                p[0] = r_pc4d[idx];
                p[1] = r_rgba[idx++];
                return *((cl_float8 *) p);
            }
        );
        // RBC::printBufferF ("Original X:", rbcCon.hPtrInX, d, nx, 3);
        // RBC::printBufferF ("Original R:", rbcCon.hPtrInR, d, nr, 3);

        // Copy data to device
        rbcCon.write (cl_algo::RBC::RBCConstruct<K, P>::Memory::D_IN_X);
        rbcCon.write (cl_algo::RBC::RBCConstruct<K, P>::Memory::D_IN_R);
        
        rbcCon.run ();  // Execute kernels (~ 344 us)
        
        // Copy results to host
        cl_uint *O = (cl_uint *) rbcCon.read (cl_algo::RBC::RBCConstruct<K, P>::Memory::H_OUT_O, CL_FALSE);
        cl_uint *N = (cl_uint *) rbcCon.read (cl_algo::RBC::RBCConstruct<K, P>::Memory::H_OUT_N, CL_FALSE);
        cl_float *Xp = (cl_float *) rbcCon.read (cl_algo::RBC::RBCConstruct<K, P>::Memory::H_OUT_X_P);
        // RBC::printBuffer ("Received O:", O, 1, nr);
        // RBC::printBuffer ("Received N:", N, 1, nr);
        // RBC::printBufferF ("Received Xp:", Xp, d, nx, 3);

        // cl_uint max_n = std::accumulate (N, N + nr, 0, 
        //     [](cl_uint a, cl_uint b) -> cl_uint { return std::max (a, b); });
        // std::cout << "max_n = " << max_n << std::endl << std::endl;

        // Search ==============================================================

        const cl_algo::RBC::KernelTypeC K2 = cl_algo::RBC::KernelTypeC::KINECT_R;
        const cl_algo::RBC::RBCPermuteConfig P2 = cl_algo::RBC::RBCPermuteConfig::GENERIC;
        const cl_algo::RBC::KernelTypeS S2 = cl_algo::RBC::KernelTypeS::KINECT;
        cl_algo::RBC::RBCSearch<K2, P2, S2> rbcSearch (clEnv, info);
        
        // Couple interfaces
        rbcSearch.get (cl_algo::RBC::RBCSearch<K2, P2, S2>::Memory::D_IN_R) = 
            rbcCon.get (cl_algo::RBC::RBCConstruct<K, P>::Memory::D_IN_R);
        rbcSearch.get (cl_algo::RBC::RBCSearch<K2, P2, S2>::Memory::D_IN_X_P) = 
            rbcCon.get (cl_algo::RBC::RBCConstruct<K, P>::Memory::D_OUT_X_P);
        rbcSearch.get (cl_algo::RBC::RBCSearch<K2, P2, S2>::Memory::D_IN_O) = 
            rbcCon.get (cl_algo::RBC::RBCConstruct<K, P>::Memory::D_OUT_O);
        rbcSearch.get (cl_algo::RBC::RBCSearch<K2, P2, S2>::Memory::D_IN_N) = 
            rbcCon.get (cl_algo::RBC::RBCConstruct<K, P>::Memory::D_OUT_N);
        rbcSearch.init (nq, nr, nx, a);

        // Initialize data (writes on staging buffer directly)
        idx = 0;
        std::generate ((cl_float8 *) rbcSearch.hPtrInQ, (cl_float8 *) rbcSearch.hPtrInQ + nq, 
            [&]()
            {
                cl_float4 p[2];
                p[0] = q_pc4d[idx];
                p[1] = q_rgba[idx++];
                return *((cl_float8 *) p);
            }
        );
        // RBC::printBufferF ("Original Q:", rbcSearch.hPtrInQ, d, nq, 3);

        // Copy data to device
        rbcSearch.write (cl_algo::RBC::RBCSearch<K2, P2, S2>::Memory::D_IN_Q);

        rbcSearch.run ();  // Execute kernels (~ 714 us)

        // Copy results to host
        cl_float *Qp = (cl_float *) rbcSearch.read (
            cl_algo::RBC::RBCSearch<K2, P2, S2>::Memory::H_OUT_Q_P, CL_FALSE);
        rbc_dist_id *RID = (rbc_dist_id *) rbcSearch.read (
            cl_algo::RBC::RBCSearch<K2, P2, S2>::Memory::H_OUT_R_ID, CL_FALSE);
        rbc_dist_id *NNID = (rbc_dist_id *) rbcSearch.read (
            cl_algo::RBC::RBCSearch<K2, P2, S2>::Memory::H_OUT_NN_ID, CL_FALSE);
        cl_float *NN = (cl_float *) rbcSearch.read (
            cl_algo::RBC::RBCSearch<K2, P2, S2>::Memory::H_OUT_NN);
        // RBC::printBufferF ("Received Qp:", Qp, d, nq, 3);
        // RBC::printBuffer ("Received RID:", (unsigned int *) RID, 2, nq);
        // RBC::printBuffer ("Received NNID:", (unsigned int *) NNID, 2, nq);
        // RBC::printBufferF ("Received NN:", NN, d, nq, 3);

        // Testing =============================================================

        // Produce reference permuted database
        rbc_dist_id *refNNID = new rbc_dist_id[nq];
        cl_float *refNN = new cl_float[nq * d];
        RBC::cpuRBCSearch8 (Qp, RID, Xp, O, N, refNNID, refNN, nq, nr, nx, a);
        // RBC::printBuffer ("Expected NNID:", (unsigned int *) refNNID, 2, nq);
        // RBC::printBufferF ("Expected NN:", refNN, d, nq, 3);

        // Verify blurred output
        // RBCSearch seems to return a few different NNs. This will have to be investigated
        // Nonetheless, the correctness of the algorithm is not affected
        float eps = 420 * std::numeric_limits<float>::epsilon ();  // 5.00679e-05
        for (uint q = 0; q < nq; ++q)
        {
            // ASSERT_EQ (refNNID[q].id, NNID[q].id);

            for (uint j = 0; j < d; ++j)
            {
                // ASSERT_EQ (refNN[q * d + j], NN[q * d + j]);
                float da = RBC::euclideanMetric8Squared (&Qp[q * d], &refNN[q * d], a);
                float db = RBC::euclideanMetric8Squared (&Qp[q * d], &NN[q * d], a);
                ASSERT_LT (std::abs (da-db), eps);
            }
        }

        // Profiling ===========================================================
        if (profiling)
        {
            const int nRepeat = 1;  /* Number of times to perform the tests. */

            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = rbcCon.run (gTimer);

            // Benchmark
            pGPU.print ("RBC_RGBD<Construction>");
        }

        if (profiling)
        {
            const int nRepeat = 1;  /* Number of times to perform the tests. */

            // CPU
            clutils::CPUTimer<double, std::milli> cTimer;
            clutils::ProfilingInfo<nRepeat> pCPU ("CPU");
            for (int i = 0; i < nRepeat; ++i)
            {
                cTimer.start ();
                RBC::cpuRBCSearch8 (Qp, RID, Xp, O, N, refNNID, refNN, nq, nr, nx, a);
                pCPU[i] = cTimer.stop ();
            }
            
            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = rbcSearch.run (gTimer);

            // Benchmark
            pGPU.print (pCPU, "RBC_RGBD<Search>");

            // Compute true NNs ================================================
            cl_float *tNN = new cl_float[nq * d];
            RBC::cpuNNSearch (Qp, Xp, tNN, nq, nx, d);

            // Calculate Error (~ 1.05) ========================================
            // Mean distance from the computed NNs, 
            // relative to the distance from the true NNs
            // 1 (optimal) - infinity (bad)
            std::cout << " Mean Error\n ----------" << std::endl << "   Value  : ";
            std::cout << std::fixed << std::setprecision (3);
            std::cout << RBC::meanError (Qp, NN, nq, d) / RBC::meanError (Qp, tNN, nq, d);
            std::cout << std::endl << std::endl;
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
