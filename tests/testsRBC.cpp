/*! \file testsRBC.cpp
 *  \brief Google Test Unit Tests for the kernels associated 
 *         with the `Random Ball Cover` data structure.
 *  \note Use the `--profiling` flag to enable profiling of the kernels.
 *  \note The benchmarks in these tests are against naive CPU implementations 
 *        of the associated algorithms. They are used only for testing purposes, 
 *        and not for examining the performance of their GPU alternatives.
 *  \author Nick Lamprianidis
 *  \version 1.0
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
#include <numeric>
#include <chrono>
#include <random>
#include <limits>
#include <cmath>
#include <gtest/gtest.h>
#include <CLUtils.hpp>
#include <RBC/algorithms.hpp>
#include <RBC/tests/helper_funcs.hpp>


bool profiling;  // Flag to enable profiling of the kernels (--profiling)

// Kernel filenames
const std::string kernel_filename_rbc    { "kernels/rbc_kernels.cl"    };
const std::string kernel_filename_scan   { "kernels/scan_kernels.cl"   };
const std::string kernel_filename_reduce { "kernels/reduce_kernels.cl" };

// Uniform random number generators
namespace RBC
{
    extern std::function<unsigned char ()> rNum_0_255;
    extern std::function<unsigned short ()> rNum_0_10000;
    extern std::function<float ()> rNum_R_0_1;
}

std::default_random_engine generator { 1234ul };
std::uniform_real_distribution<float> distributionR1 { 0.f, 1.f };
std::function<float ()> rNum_R_0_1_ = std::bind (distributionR1, generator);

/*! \brief Tests the **rbcComputeDists_{ShareNone,SharedR,SharedXR}** kernel.
 *  \details The kernel computes the distances between the points of two sets.
 */
TEST (RBC, rbcComputeDists)
{
    try
    {
        const unsigned int nx = 1 << 14;  // 16384
        const unsigned int nr = 1 << 7;   //   128
        const unsigned int d = 8;
        const unsigned int bufferSizeX = nx * d * sizeof (cl_float);
        const unsigned int bufferSizeR = nr * d * sizeof (cl_float);

        // Setup the OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addProgram (0, kernel_filename_rbc);

        // Configure kernel execution parameters
        clutils::CLEnvInfo<1> info (0, 0, 0, { 0 }, 0);
        cl_algo::RBC::RBCComputeDists cd (clEnv, info, cl_algo::RBC::RBCComputeDists::KernelType::SHARED_X_R);
        cd.init (nx, nr, d);

        // Initialize data (writes on staging buffers directly)
        std::generate (cd.hPtrInX, cd.hPtrInX + bufferSizeX / sizeof (cl_float), RBC::rNum_R_0_1);
        std::generate (cd.hPtrInR, cd.hPtrInR + bufferSizeR / sizeof (cl_float), RBC::rNum_R_0_1);
        // RBC::printBufferF ("Original X:", cd.hPtrInX, d, nx, 3);
        // RBC::printBufferF ("Original R:", cd.hPtrInR, d, nr, 3);

        // Copy data to device
        cd.write (cl_algo::RBC::RBCComputeDists::Memory::D_IN_X);
        cd.write (cl_algo::RBC::RBCComputeDists::Memory::D_IN_R);

        // Execute kernels
        //~ rbcComputeDists_SharedNone:  86 us
        //~ rbcComputeDists_SharedR   : 134 us
        //~ rbcComputeDists_SharedXR  : 156 us
        cd.run ();
        
        cl_float *results = (cl_float *) cd.read ();  // Copy results to host
        // RBC::printBufferF ("Received:", results, nr, nx, 3);

        // Produce reference array of distances
        cl_float *refCD = new cl_float[nr * nx];
        RBC::cpuRBCComputeDists (cd.hPtrInX, cd.hPtrInR, refCD, nx, nr, d);
        // RBC::printBufferF ("Expected:", refCD, nr, nx, 3);

        // Verify blurred output
        float eps = 42 * std::numeric_limits<float>::epsilon ();  // 5.00679e-06
        for (uint x = 0; x < nx; ++x)
            for (uint r = 0; r < nr; ++r)
                ASSERT_LT (std::abs (refCD[x * nr + r] - results[x * nr + r]), eps);

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
                RBC::cpuRBCComputeDists (cd.hPtrInX, cd.hPtrInR, refCD, nx, nr, d);
                pCPU[i] = cTimer.stop ();
            }
            
            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = cd.run (gTimer);

            // Benchmark
            pGPU.print (pCPU, "RBCComputeDists");
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


/*! \brief Tests the **rbcComputeDists_Kinect** kernel.
 *  \details The kernel computes the distances between the 
 *           points of two sets of Kinect 8-D data.
 */
TEST (RBC, rbcComputeDists_Kinect)
{
    try
    {
        const unsigned int nx = 1 << 14;  // 16384
        const unsigned int nr = 1 << 7;   //   128
        const unsigned int d = 8;
        const unsigned int bufferSizeX = nx * d * sizeof (cl_float);
        const unsigned int bufferSizeR = nr * d * sizeof (cl_float);

        // Setup the OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addProgram (0, kernel_filename_rbc);

        // Configure kernel execution parameters
        clutils::CLEnvInfo<1> info (0, 0, 0, { 0 }, 0);
        cl_algo::RBC::RBCComputeDists cd (clEnv, info);
        cd.init (nx, nr);

        // Initialize data (writes on staging buffers directly)
        std::generate (cd.hPtrInX, cd.hPtrInX + bufferSizeX / sizeof (cl_float), RBC::rNum_R_0_1);
        std::generate (cd.hPtrInR, cd.hPtrInR + bufferSizeR / sizeof (cl_float), RBC::rNum_R_0_1);
        // RBC::printBufferF ("Original X:", cd.hPtrInX, d, nx, 3);
        // RBC::printBufferF ("Original R:", cd.hPtrInR, d, nr, 3);

        // Copy data to device
        cd.write (cl_algo::RBC::RBCComputeDists::Memory::D_IN_X);
        cd.write (cl_algo::RBC::RBCComputeDists::Memory::D_IN_R);
        
        // Execute kernels
        //~ rbcComputeDists_SharedNone:  88 us
        //~ rbcComputeDists_Kinect    : 114 us
        //~ rbcComputeDists_Kinect_2  : 127 us
        cd.run ();
        
        cl_float *results = (cl_float *) cd.read ();  // Copy results to host
        // RBC::printBufferF ("Received:", results, nr, nx, 3);

        // Produce reference array of distances
        cl_float *refCD = new cl_float[nr * nx];
        RBC::cpuRBCComputeDists (cd.hPtrInX, cd.hPtrInR, refCD, nx, nr, d);
        // RBC::printBufferF ("Expected:", refCD, nr, nx, 3);

        // Verify blurred output
        float eps = 42 * std::numeric_limits<float>::epsilon ();  // 5.00679e-06
        for (uint x = 0; x < nx; ++x)
            for (uint r = 0; r < nr; ++r)
                ASSERT_LT (std::abs (refCD[x * nr + r] - results[x * nr + r]), eps);

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
                RBC::cpuRBCComputeDists (cd.hPtrInX, cd.hPtrInR, refCD, nx, nr, d);
                pCPU[i] = cTimer.stop ();
            }
            
            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = cd.run (gTimer);

            // Benchmark
            pGPU.print (pCPU, "RBCComputeDists[Kinect]");
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


/*! \brief Tests the **rbcMinDists** kernel.
 *  \details The kernel computes the minimum element of each row of an array.
 */
TEST (RBC, rbcMinDists)
{
    try
    {
        const unsigned int nx = 1 << 14;  // 16384
        const unsigned int nr = 1 << 7;   //   128
        const unsigned int bufferInSize = nr * nx * sizeof (cl_float);
        const unsigned int bufferOutSize = nx * sizeof (rbc_dist_id);

        // Setup the OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addProgram (0, kernel_filename_rbc);

        // Configure kernel execution parameters
        clutils::CLEnvInfo<1> info (0, 0, 0, { 0 }, 0);
        const cl_algo::RBC::RBCMinConfig C = cl_algo::RBC::RBCMinConfig::CONSTRUCTION;
        cl_algo::RBC::RBCMin<C> rbcMin (clEnv, info);
        rbcMin.init (nr, nx);

        // Initialize data (writes on staging buffer directly)
        std::generate (rbcMin.hPtrInD, rbcMin.hPtrInD + bufferInSize / sizeof (cl_float), RBC::rNum_R_0_1);
        // RBC::printBufferF ("Original:", rbcMin.hPtrInD, nr, nx, 3);

        rbcMin.write ();  // Copy data to device
        
        rbcMin.run ();  // Execute kernels (~ 123 us)
        
        // Copy results to host
        rbc_dist_id *ID = (rbc_dist_id *) rbcMin.read (cl_algo::RBC::RBCMin<C>::Memory::H_OUT_ID, CL_FALSE);
        cl_uint *Rnk = (cl_uint *) rbcMin.read (cl_algo::RBC::RBCMin<C>::Memory::H_OUT_RNK, CL_FALSE);
        cl_uint *N = (cl_uint *) rbcMin.read (cl_algo::RBC::RBCMin<C>::Memory::H_OUT_N);
        // RBC::printBuffer ("Received ID:", (unsigned int *) ID, 2, nx);
        // RBC::printBuffer ("Received Rnk:", Rnk, 1, nx);
        // RBC::printBuffer ("Received N:", N, 1, nr);

        // Produce reference representative ids
        rbc_dist_id *refMin = new rbc_dist_id[nx];
        cl_uint *refRnk = new cl_uint[nx];
        cl_uint *refN = new cl_uint[nr];
        RBC::cpuRBCMinDists (rbcMin.hPtrInD, refMin, refN, refRnk, nr, nx, true);
        // RBC::printBuffer ("Expected ID:", (unsigned int *) refMin, 2, nx);
        // RBC::printBuffer ("Expected Rnk:", refRnk, 1, nx);
        // RBC::printBuffer ("Expected N:", refN, 1, nr);

        // Verify blurred output
        float eps = std::numeric_limits<float>::epsilon ();  // 1.19209e-07
        for (uint i = 0; i < nr; ++i)
            ASSERT_EQ (refN[i], N[i]);
        for (uint i = 0; i < nx; ++i)
        {
            ASSERT_LT (std::abs (refMin[i].dist - ID[i].dist), eps);
            ASSERT_EQ (refMin[i].id, ID[i].id);
            // ASSERT_EQ (refRnk[i], Rnk[i]);  // They are not processed in the same order!
        }

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
                RBC::cpuRBCMinDists (rbcMin.hPtrInD, refMin, refN, refRnk, nr, nx, true);
                pCPU[i] = cTimer.stop ();
            }
            
            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = rbcMin.run (gTimer);

            // Benchmark
            pGPU.print (pCPU, "RBCMin[rbcMinDists]");
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


/*! \brief Tests the **rbcGroupMinDists** kernel.
 *  \details The kernel computes the minimum element of each row of an array.
 */
TEST (RBC, rbcGroupMinDists)
{
    try
    {
        const unsigned int nx = 1 << 14; // 16384
        const unsigned int nr = 5 << 7;  //   640
        const unsigned int bufferInSize = nr * nx * sizeof (cl_float);
        const unsigned int bufferOutSize = nx * sizeof (rbc_dist_id);

        // Setup the OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addProgram (0, kernel_filename_rbc);

        // Configure kernel execution parameters
        clutils::CLEnvInfo<1> info (0, 0, 0, { 0 }, 0);
        const cl_algo::RBC::RBCMinConfig C = cl_algo::RBC::RBCMinConfig::CONSTRUCTION;
        cl_algo::RBC::RBCMin<C> rbcMin (clEnv, info);
        rbcMin.init (nr, nx);

        // Initialize data (writes on staging buffer directly)
        std::generate (rbcMin.hPtrInD, rbcMin.hPtrInD + bufferInSize / sizeof (cl_float), RBC::rNum_R_0_1);
        // RBC::printBufferF ("Original:", rbcMin.hPtrInD, nr, nx, 3);

        rbcMin.write ();  // Copy data to device
        
        rbcMin.run ();  // Execute kernels (~ 419 us)
        
        // Copy results to host
        rbc_dist_id *ID = (rbc_dist_id *) rbcMin.read (cl_algo::RBC::RBCMin<C>::Memory::H_OUT_ID, CL_FALSE);
        cl_uint *Rnk = (cl_uint *) rbcMin.read (cl_algo::RBC::RBCMin<C>::Memory::H_OUT_RNK, CL_FALSE);
        cl_uint *N = (cl_uint *) rbcMin.read (cl_algo::RBC::RBCMin<C>::Memory::H_OUT_N);
        // RBC::printBuffer ("Received ID:", (unsigned int *) ID, 2, nx);
        // RBC::printBuffer ("Received N:", N, 1, nr);
        // RBC::printBuffer ("Received Rnk:", Rnk, 1, nx);

        // Produce reference representative ids
        rbc_dist_id *refMin = new rbc_dist_id[nx];
        cl_uint *refRnk = new cl_uint[nx];
        cl_uint *refN = new cl_uint[nr];
        RBC::cpuRBCMinDists (rbcMin.hPtrInD, refMin, refN, refRnk, nr, nx, true);
        // RBC::printBuffer ("Expected ID:", (unsigned int *) refMin, 2, nx);
        // RBC::printBuffer ("Expected Rnk:", refRnk, 1, nx);
        // RBC::printBuffer ("Expected N:", refN, 1, nr);

        // Verify blurred output
        float eps = std::numeric_limits<float>::epsilon ();  // 1.19209e-07
        for (uint i = 0; i < nr; ++i)
            ASSERT_EQ (refN[i], N[i]);
        for (uint i = 0; i < nx; ++i)
        {
            ASSERT_LT (std::abs (refMin[i].dist - ID[i].dist), eps);
            ASSERT_EQ (refMin[i].id, ID[i].id);
            // ASSERT_EQ (refRnk[i], Rnk[i]);  // They are not processed in the same order!
        }

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
                RBC::cpuRBCMinDists (rbcMin.hPtrInD, refMin, refN, refRnk, nr, nx, true);
                pCPU[i] = cTimer.stop ();
            }
            
            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = rbcMin.run (gTimer);

            // Benchmark
            pGPU.print (pCPU, "RBCMin[rbcGroupMinDists]");
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


/*! \brief Tests the **rbcPermute** kernel.
 *  \details The kernel permutes the database points to form the representative lists.
 */
TEST (RBC, rbcPermute)
{
    try
    {
        const std::vector<std::string> kernel_files = { kernel_filename_rbc, 
                                                        kernel_filename_scan };

        const unsigned int nx = 1 << 14;  // 16384
        const unsigned int nr = 1 << 7;   //   128
        const unsigned int d = 8;
        const unsigned int bufferXSize = nx * d * sizeof (cl_float);
        const unsigned int bufferRSize = nr * d * sizeof (cl_float);
        const unsigned int bufferDSize = nr * nx * sizeof (cl_float);
        const unsigned int bufferIDSize = nx * sizeof (rbc_dist_id);
        const unsigned int bufferRnkSize = nx * sizeof (cl_uint);
        const unsigned int bufferOSize = nr * sizeof (cl_uint);

        // Setup the OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addProgram (0, kernel_files);

        clutils::CLEnvInfo<1> info (0, 0, 0, { 0 }, 0);

        // Prepare data ========================================================

        // Compute matrix of distances -----------------------------------------

        cl_algo::RBC::RBCComputeDists cDist (clEnv, info, cl_algo::RBC::RBCComputeDists::KernelType::SHARED_NONE);
        cDist.init (nx, nr, d);
        cl_float *X = cDist.hPtrInX;
        cl_float *R = cDist.hPtrInR;
        
        std::generate (cDist.hPtrInX, cDist.hPtrInX + bufferXSize / sizeof (cl_float), RBC::rNum_R_0_1);
        std::generate (cDist.hPtrInR, cDist.hPtrInR + bufferRSize / sizeof (cl_float), RBC::rNum_R_0_1);
        // RBC::printBufferF ("X:", X, d, nx, 1);
        // RBC::printBufferF ("R:", R, d, nr, 1);

        cDist.write (cl_algo::RBC::RBCComputeDists::Memory::D_IN_X);
        cDist.write (cl_algo::RBC::RBCComputeDists::Memory::D_IN_R);

        cDist.run ();

        cl_float *D = (cl_float *) cDist.read ();
        // RBC::printBufferF ("D:", D, nr, nx, 1);

        // Reduce distances (find representative ids for each db point) --------

        const cl_algo::RBC::RBCMinConfig C1 = cl_algo::RBC::RBCMinConfig::CONSTRUCTION;
        cl_algo::RBC::RBCMin<C1> rbcMin (clEnv, info);
        rbcMin.init (nr, nx);

        rbcMin.write (cl_algo::RBC::RBCMin<C1>::Memory::D_IN_D, D);
        
        rbcMin.run ();
        
        rbc_dist_id *ID = (rbc_dist_id *) rbcMin.read (cl_algo::RBC::RBCMin<C1>::Memory::H_OUT_ID, CL_FALSE);
        cl_uint *Rnk = (cl_uint *) rbcMin.read (cl_algo::RBC::RBCMin<C1>::Memory::H_OUT_RNK, CL_FALSE);
        cl_uint *N = (cl_uint *) rbcMin.read (cl_algo::RBC::RBCMin<C1>::Memory::H_OUT_N);
        // RBC::printBuffer ("ID:", (unsigned int *) ID, 2, nx);
        // RBC::printBuffer ("Rnk:", Rnk, nx, 1);
        // RBC::printBuffer ("N:", N, nr, 1);

        // Scan list cardinalities (compute list offsets) ----------------------

        const cl_algo::RBC::ScanConfig C2 = cl_algo::RBC::ScanConfig::EXCLUSIVE;
        cl_algo::RBC::Scan<C2> scan (clEnv, info);
        scan.init (nr, 1);

        scan.write (cl_algo::RBC::Scan<C2>::Memory::D_IN, N);

        scan.run ();

        cl_uint *O = (cl_uint *) scan.read ();
        // RBC::printBuffer ("O:", O, nr, 1);

        // Test rbcPermute ===================================================

        // Configure kernel execution parameters
        cl_algo::RBC::RBCPermute pDB (clEnv, info);
        pDB.init (nx, nr, d);

        // Copy data to device
        pDB.write (cl_algo::RBC::RBCPermute::Memory::D_IN_X, X);
        pDB.write (cl_algo::RBC::RBCPermute::Memory::D_IN_ID, ID);
        pDB.write (cl_algo::RBC::RBCPermute::Memory::D_IN_RNK, Rnk);
        pDB.write (cl_algo::RBC::RBCPermute::Memory::D_IN_O, O);

        pDB.run ();  // Execute kernel (~ 14 us)

        cl_float *Xp = (cl_float *) pDB.read ();  // Copy results to host
        // RBC::printBufferF ("Received Xp:", Xp, d, nx, 3);

        // Produce reference permuted database
        cl_float *refXp = new cl_float[nx * d];
        RBC::cpuRBCPermute (X, ID, O, Rnk, refXp, nx, nr, d);
        // RBC::printBufferF ("Expected Xp:", refXp, d, nx, 3);

        // Verify blurred output
        float eps = std::numeric_limits<float>::epsilon ();  // 1.19209e-07
        for (uint x = 0; x < nx; ++x)
            for (uint j = 0; j < d; ++j)
                ASSERT_LT (std::abs (refXp[x * d + j] - Xp[x * d + j]), eps);

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
                RBC::cpuRBCPermute (X, ID, O, Rnk, refXp, nx, nr, d);
                pCPU[i] = cTimer.stop ();
            }
            
            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = pDB.run (gTimer);

            // Benchmark
            pGPU.print (pCPU, "RBCPermute");
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


/*! \brief Tests the **RBCConstruct** class.
 *  \details The class constructs the representative lists.
 */
TEST (RBC, rbcConstruct)
{
    try
    {
        const std::vector<std::string> kernel_files = { kernel_filename_rbc, 
                                                        kernel_filename_scan };

        const unsigned int nx = 1 << 14;  // 16384
        const unsigned int nr = 1 << 8;   //   256
        const unsigned int d = 8;
        const unsigned int bufferXSize = nx * d * sizeof (cl_float);
        const unsigned int bufferRSize = nr * d * sizeof (cl_float);
        const unsigned int bufferDSize = nr * nx * sizeof (cl_float);
        const unsigned int bufferIDSize = nx * sizeof (rbc_dist_id);
        const unsigned int bufferRnkSize = nx * sizeof (cl_uint);
        const unsigned int bufferOSize = nr * sizeof (cl_uint);

        // Setup the OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addProgram (0, kernel_files);

        // Configure kernel execution parameters
        clutils::CLEnvInfo<1> info (0, 0, 0, { 0 }, 0);
        cl_algo::RBC::RBCConstruct rbcCon (clEnv, info);
        rbcCon.init (nx, nr, d);

        // Initialize data (writes on staging buffer directly)
        std::generate (rbcCon.hPtrInX, rbcCon.hPtrInX + bufferXSize / sizeof (cl_float), RBC::rNum_R_0_1);
        std::generate (rbcCon.hPtrInR, rbcCon.hPtrInR + bufferRSize / sizeof (cl_float), RBC::rNum_R_0_1);
        // RBC::printBufferF ("Original X:", rbcCon.hPtrInX, d, nx, 3);
        // RBC::printBufferF ("Original R:", rbcCon.hPtrInR, d, nr, 3);

        // Copy data to device
        rbcCon.write (cl_algo::RBC::RBCConstruct::Memory::D_IN_X);
        rbcCon.write (cl_algo::RBC::RBCConstruct::Memory::D_IN_R);
        
        rbcCon.run ();  // Execute kernels (~ 330 us)
        
        // Copy results to host
        rbc_dist_id *ID = (rbc_dist_id *) rbcCon.read (cl_algo::RBC::RBCConstruct::Memory::H_OUT_ID, CL_FALSE);
        cl_uint *Rnk = (cl_uint *) rbcCon.read (cl_algo::RBC::RBCConstruct::Memory::H_OUT_RNK, CL_FALSE);
        cl_uint *O = (cl_uint *) rbcCon.read (cl_algo::RBC::RBCConstruct::Memory::H_OUT_O, CL_FALSE);
        cl_float *Xp = (cl_float *) rbcCon.read (cl_algo::RBC::RBCConstruct::Memory::H_OUT_X_P);
        // RBC::printBufferF ("Received:", Xp, d, nx, 3);

        // Produce reference permuted database
        cl_float *refXp = new cl_float[nx * d];
        RBC::cpuRBCPermute (rbcCon.hPtrInX, ID, O, Rnk, refXp, nx, nr, d);
        // RBC::printBufferF ("Expected:", refXp, d, nx, 3);

        // Verify blurred output
        for (uint x = 0; x < nx; ++x)
            for (uint j = 0; j < d; ++j)
                ASSERT_EQ (refXp[x * d + j], Xp[x * d + j]);

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
                RBC::cpuRBCPermute (rbcCon.hPtrInX, ID, O, Rnk, refXp, nx, nr, d);
                pCPU[i] = cTimer.stop ();
            }
            
            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = rbcCon.run (gTimer);

            // Benchmark
            pGPU.print (pCPU, "RBCConstruct");
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


/*! \brief Tests the **RBCSearch** class.
 *  \details The class computes the NNs of a set of queries.
 */
TEST (RBC, rbcSearch)
{
    try
    {
        const std::vector<std::string> kernel_files = { kernel_filename_rbc, 
                                                        kernel_filename_scan,
                                                        kernel_filename_reduce };

        const unsigned int nx = 1 << 14;  // 16384
        const unsigned int nr = 1 <<  8;  //   256
        const unsigned int nq = 1 << 14;  // 16384
        const unsigned int d = 8;
        const unsigned int bufferXSize = nx * d * sizeof (cl_float);
        const unsigned int bufferRSize = nr * d * sizeof (cl_float);
        const unsigned int bufferQSize = nq * d * sizeof (cl_float);
        const unsigned int bufferOSize = nr * sizeof (cl_uint);
        const unsigned int bufferNSize = nr * sizeof (cl_uint);
        const unsigned int bufferDSize = nr * nx * sizeof (cl_float);
        const unsigned int bufferRIDSize = nx * sizeof (rbc_dist_id);
        const unsigned int bufferNNIDSize = nq * sizeof (rbc_dist_id);
        const unsigned int bufferRnkSize = nx * sizeof (cl_uint);

        // Setup the OpenCL environment
        clutils::CLEnv clEnv;
        clEnv.addContext (0);
        clEnv.addQueue (0, 0, CL_QUEUE_PROFILING_ENABLE);
        clEnv.addProgram (0, kernel_files);
        cl::Context &context = clEnv.getContext (0);

        // Construction ========================================================

        // Configure kernel execution parameters
        clutils::CLEnvInfo<1> info (0, 0, 0, { 0 }, 0);
        cl_algo::RBC::RBCConstruct rbcCon (clEnv, info);

        rbcCon.get (cl_algo::RBC::RBCConstruct::Memory::D_OUT_X_P) = 
            cl::Buffer (context, CL_MEM_READ_WRITE, bufferXSize);
        rbcCon.init (nx, nr, d);

        // Initialize data (writes on staging buffer directly)
        std::generate (rbcCon.hPtrInX, rbcCon.hPtrInX + bufferXSize / sizeof (cl_float), RBC::rNum_R_0_1);
        std::generate (rbcCon.hPtrInR, rbcCon.hPtrInR + bufferRSize / sizeof (cl_float), RBC::rNum_R_0_1);
        // RBC::printBufferF ("Original X:", rbcCon.hPtrInX, d, nx, 3);
        // RBC::printBufferF ("Original R:", rbcCon.hPtrInR, d, nr, 3);

        // Copy data to device
        rbcCon.write (cl_algo::RBC::RBCConstruct::Memory::D_IN_X);
        rbcCon.write (cl_algo::RBC::RBCConstruct::Memory::D_IN_R);
        
        rbcCon.run ();  // Execute kernels
        
        // Copy results to host
        rbc_dist_id *ID = (rbc_dist_id *) rbcCon.read (cl_algo::RBC::RBCConstruct::Memory::H_OUT_ID, CL_FALSE);
        cl_uint *O = (cl_uint *) rbcCon.read (cl_algo::RBC::RBCConstruct::Memory::H_OUT_O, CL_FALSE);
        cl_uint *N = (cl_uint *) rbcCon.read (cl_algo::RBC::RBCConstruct::Memory::H_OUT_N, CL_FALSE);
        cl_float *Xp = (cl_float *) rbcCon.read (cl_algo::RBC::RBCConstruct::Memory::H_OUT_X_P);
        // RBC::printBufferF ("Received:", Xp, d, nx, 3);

        // Search ==============================================================

        cl_algo::RBC::RBCSearch rbcSearch (clEnv, info);
        
        // Couple interfaces
        rbcSearch.get (cl_algo::RBC::RBCSearch::Memory::D_IN_R) = 
            rbcCon.get (cl_algo::RBC::RBCConstruct::Memory::D_IN_R);
        rbcSearch.get (cl_algo::RBC::RBCSearch::Memory::D_IN_X_P) = 
            rbcCon.get (cl_algo::RBC::RBCConstruct::Memory::D_OUT_X_P);
        rbcSearch.get (cl_algo::RBC::RBCSearch::Memory::D_IN_O) = 
            rbcCon.get (cl_algo::RBC::RBCConstruct::Memory::D_OUT_O);
        rbcSearch.get (cl_algo::RBC::RBCSearch::Memory::D_IN_N) = 
            rbcCon.get (cl_algo::RBC::RBCConstruct::Memory::D_OUT_N);
        rbcSearch.init (nq, nr, nx, d);

        // Initialize data (writes on staging buffer directly)
        std::generate (rbcSearch.hPtrInQ, rbcSearch.hPtrInQ + bufferQSize / sizeof (cl_float), rNum_R_0_1_);
        // RBC::printBufferF ("Original Q:", rbcSearch.hPtrInQ, d, nq, 3);

        // Copy data to device
        rbcSearch.write (cl_algo::RBC::RBCSearch::Memory::D_IN_Q);

        rbcSearch.run ();  // Execute kernels (~ 1100 us)

        // Copy results to host
        rbc_dist_id *RID = (rbc_dist_id *) rbcSearch.read (cl_algo::RBC::RBCSearch::Memory::H_OUT_R_ID, CL_FALSE);
        rbc_dist_id *NNID = (rbc_dist_id *) rbcSearch.read (cl_algo::RBC::RBCSearch::Memory::H_OUT_NN_ID, CL_FALSE);
        cl_float *NN = (cl_float *) rbcSearch.read (cl_algo::RBC::RBCSearch::Memory::H_OUT_NN);
        // RBC::printBuffer ("Received RID:", (unsigned int *) RID, 2, nq);
        // RBC::printBuffer ("Received NNID:", (unsigned int *) NNID, 2, nq);
        // RBC::printBufferF ("Received NN:", NN, d, nq, 3);

        // Testing =============================================================

        // Produce reference permuted database
        rbc_dist_id *refRID = new rbc_dist_id[nq];
        rbc_dist_id *refNNID = new rbc_dist_id[nq];
        cl_float *refNN = new cl_float[nq * d];
        RBC::cpuRBCSearch (rbcSearch.hPtrInQ, rbcCon.hPtrInR, Xp, O, N, refRID, refNNID, refNN, nq, nr, nx, d);
        // RBC::printBuffer ("Expected RID:", (unsigned int *) refRID, 2, nq);
        // RBC::printBuffer ("Expected NNID:", (unsigned int *) refNNID, 2, nq);
        // RBC::printBufferF ("Expected NN:", refNN, d, nq, 3);

        // Verify blurred output
        for (uint q = 0; q < nq; ++q)
            ASSERT_EQ (refRID[q].id, RID[q].id);
        for (uint q = 0; q < nq; ++q)
            ASSERT_EQ (refNNID[q].id, NNID[q].id);
        for (uint q = 0; q < nq; ++q)
            for (uint j = 0; j < d; ++j)
                ASSERT_EQ (refNN[q * d + j], NN[q * d + j]);

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
                RBC::cpuRBCSearch (rbcSearch.hPtrInQ, rbcCon.hPtrInR, Xp, O, N, refRID, refNNID, refNN, nq, nr, nx, d);
                pCPU[i] = cTimer.stop ();
            }
            
            // GPU
            clutils::GPUTimer<std::milli> gTimer (clEnv.devices[0][0]);
            clutils::ProfilingInfo<nRepeat> pGPU ("GPU");
            for (int i = 0; i < nRepeat; ++i)
                pGPU[i] = rbcSearch.run (gTimer);

            // Benchmark
            pGPU.print (pCPU, "RBCSearch");
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
