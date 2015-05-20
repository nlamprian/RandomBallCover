/*! \file algorithms.hpp
 *  \brief Declares classes that organize the execution of OpenCL kernels.
 *  \details Each class hides the details of kernel execution. They
 *           initialize the necessary buffers, set up the workspaces, 
 *           and run the kernels.
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

#ifndef ALGORITHMS_HPP
#define ALGORITHMS_HPP

#include <CLUtils.hpp>
#include <RBC/data_types.hpp>
#include <RBC/common.hpp>


/*! \brief Offers classes which set up kernel execution parameters and 
 *         provide interfaces for the handling of memory objects.
 */
namespace cl_algo
{
namespace RBC
{

    /*! \brief Enumerates the kernels available for computing 
     *         the array of distances in the construction step.
     *  
     *  \param SHARED_NONE refers to the `rbcComputeDists_SharedNone` kernel.
     *  \param SHARED_R refers to the `rbcComputeDists_SharedR` kernel.
     *  \param SHARED_X_R refers to the `rbcComputeDists_SharedXR` kernel.
     *  \param KINECT refers to the `rbcComputeDists_Kinect` kernel.
     *  \param KINECT_R refers to the `rbcComputeDists_Kinect_R` kernel.
     *  \param KINECT_XR refers to the `rbcComputeDists_Kinect_XR` kernel.
     */
    enum class KernelTypeC : uint8_t { SHARED_NONE, SHARED_R, SHARED_X_R, KINECT, KINECT_R, KINECT_X_R };


    /*! \brief Interface class for the `rbcComputeDists` kernels.
     *  \details The `rbcComputeDists` kernels compute the distances between 
     *           two sets of points in a brute force way. For more details, 
     *           look at the kernel's documentation.
     *  \note The `rbcComputeDists` kernels are available 
     *        in `kernels/rbc_kernels.cl`.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created by 
     *        a `RBCComputeDists` instance:<br>
     *        |  Name  | Type | Placement | I/O | Use | Properties | Size |
     *        |  ---   |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN_X | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$n_x*d * sizeof\ (cl\_float)\f$ |
     *        | H_IN_R | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$n_r*d * sizeof\ (cl\_float)\f$ |
     *        | H_OUT_D| Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$n_x*n_r*sizeof\ (cl\_float)\f$ |
     *        | D_IN_X | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$n_x*d * sizeof\ (cl\_float)\f$ |
     *        | D_IN_R | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$n_r*d * sizeof\ (cl\_float)\f$ |
     *        | D_OUT_D| Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$n_x*n_r*sizeof\ (cl\_float)\f$ |
     *        
     *  \tparam K configures the class for using one of the available kernels.
     */
    template <KernelTypeC K = KernelTypeC::SHARED_NONE>
    class RBCComputeDists
    {
    public:
        /*! \brief Enumerates the memory objects handled by the class.
         *  \note `H_*` names refer to staging buffers on the host.
         *  \note `D_*` names refer to buffers on the device.
         *  
         *  \param H_IN_X input staging buffer.
         *  \param H_IN_R input staging buffer.
         *  \param H_OUT_D output staging buffer for channel R.
         *  \param D_IN_X input buffer.
         *  \param D_IN_R input buffer.
         *  \param D_OUT_D output buffer for channel R.
         */
        enum class Memory : uint8_t { H_IN_X, H_IN_R, H_OUT_D, D_IN_X, D_IN_R, D_OUT_D };

        /*! \brief Configures an OpenCL environment as specified by `_info`. */
        RBCComputeDists (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info);
        /*! \brief Returns a reference to an internal memory object. */
        cl::Memory& get (RBCComputeDists::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (unsigned int _nx, unsigned int _nr, unsigned int _d = 8, float _a = 1.f, Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (RBCComputeDists::Memory mem = RBCComputeDists::Memory::D_IN_X, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (RBCComputeDists::Memory mem = RBCComputeDists::Memory::H_OUT_D, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Gets the scaling factor \f$ \alpha \f$. */
        float getAlpha ();
        /*! \brief Sets the scaling factor \f$ \alpha \f$. */
        void setAlpha (float _a);

        cl_float *hPtrInX;  /*!< Mapping of the input staging buffer for X. */
        cl_float *hPtrInR;  /*!< Mapping of the input staging buffer for R. */
        cl_float *hPtrOutD;  /*!< Mapping of the output staging buffer. */

    private:
        clutils::CLEnv &env;
        clutils::CLEnvInfo<1> info;
        cl::Context context;
        cl::CommandQueue queue;
        cl::Kernel kernel;
        cl::NDRange global, local;
        Staging staging;
        float a;
        unsigned int nx, nr, d;
        unsigned int bufferXSize, bufferRSize, bufferDSize;
        cl::Buffer hBufferInX, hBufferInR, hBufferOutD;
        cl::Buffer dBufferInX, dBufferInR, dBufferOutD;

    public:
        /*! \brief Executes the necessary kernels.
         *  \details This `run` instance is used for profiling.
         *  
         *  \param[in] timer `GPUTimer` that does the profiling of the kernel executions.
         *  \param[in] events a wait-list of events.
         *  \return Τhe total execution time measured by the timer.
         */
        template <typename period>
        double run (clutils::GPUTimer<period> &timer, const std::vector<cl::Event> *events = nullptr)
        {
            queue.enqueueNDRangeKernel (kernel, cl::NullRange, global, local, events, &timer.event ());
            queue.flush (); timer.wait ();

            return timer.duration ();
        }

    };


    /*! \brief Enumerates configurations for the `Reduce` class.
     * 
     *  \param MIN identifies the case of `min` reduce.
     *  \param MAX identifies the case of `max` reduce.
     */
    enum class ReduceConfig : uint8_t { MIN, MAX };


    /*! \brief Interface class for the `reduce` kernels.
     *  \details The `reduce` kernels reduce each row of an array to a single element. 
     *           For more details, look at the kernels' documentation.
     *  \note The `reduce` kernela are available in `kernels/reduce_kernels.cl`.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created by a `Reduce` instance:<br>
     *        |  Name  | Type | Placement | I/O | Use | Properties | Size |
     *        |  ---   |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN   | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$columns*rows*sizeof\ (T)\f$ |
     *        | H_OUT  | Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$        rows*sizeof\ (T)\f$ |
     *        | D_IN   | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$columns*rows*sizeof\ (T)\f$ |
     *        | D_OUT  | Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$        rows*sizeof\ (T)\f$ |
     *  
     *  \tparam C configures the class for different types of reduction.
     *  \tparam T configures the class to work with different types of data.
     */
    template <ReduceConfig C, typename T = cl_float>
    class Reduce
    {
    public:
        /*! \brief Enumerates the memory objects handled by the class.
         *  \note `H_*` names refer to staging buffers on the host.
         *  \note `D_*` names refer to buffers on the device.
         *  
         *  \param H_IN input staging buffer.
         *  \param H_OUT output staging buffer.
         *  \param D_IN input buffer.
         *  \param D_REC buffer of reduced elements per work-group.
         *  \param D_OUT output buffer.
         */
        enum class Memory : uint8_t { H_IN, H_OUT, D_IN, D_REC, D_OUT };

        /*! \brief Configures an OpenCL environment as specified by `_info`. */
        Reduce (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info);
        /*! \brief Returns a reference to an internal memory object. */
        cl::Memory& get (Reduce::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (unsigned int _cols, unsigned int _rows, Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (Reduce::Memory mem = Reduce::Memory::D_IN, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (Reduce::Memory mem = Reduce::Memory::H_OUT, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);

        T *hPtrIn;  /*!< Mapping of the input staging buffer. */
        T *hPtrOut;  /*!< Mapping of the output staging buffer. */

    private:
        clutils::CLEnv &env;
        clutils::CLEnvInfo<1> info;
        cl::Context context;
        cl::CommandQueue queue;
        cl::Kernel recKernel, groupRecKernel;
        cl::NDRange globalR, globalGR, local;
        Staging staging;
        size_t wgMultiple, wgXdim;
        unsigned int cols, rows;
        unsigned int bufferInSize, bufferGRSize, bufferOutSize;
        cl::Buffer hBufferIn, hBufferOut;
        cl::Buffer dBufferIn, dBufferRec, dBufferOut;

    public:
        /*! \brief Executes the necessary kernels.
         *  \details This `run` instance is used for profiling.
         *  
         *  \param[in] timer `GPUTimer` that does the profiling of the kernel executions.
         *  \param[in] events a wait-list of events.
         *  \return Τhe total execution time measured by the timer.
         */
        template <typename period>
        double run (clutils::GPUTimer<period> &timer, const std::vector<cl::Event> *events = nullptr)
        {
            double pTime;

            if (wgXdim == 1)
            {
                queue.enqueueNDRangeKernel (recKernel, cl::NullRange, globalR, local, events, &timer.event ());
                queue.flush (); timer.wait ();
                pTime = timer.duration ();
            }
            else
            {
                queue.enqueueNDRangeKernel (recKernel, cl::NullRange, globalR, local, events, &timer.event ());
                queue.flush (); timer.wait ();
                pTime = timer.duration ();

                queue.enqueueNDRangeKernel (groupRecKernel, cl::NullRange, globalGR, local, nullptr, &timer.event ());
                queue.flush (); timer.wait ();
                pTime += timer.duration ();
            }

            return pTime;
        }

    };


    /*! \brief Interface class for the `rbcMinDists` kernel.
     *  \details `rbcMinDists` computes the minimum element, and its corresponding 
     *           column id, of each row in an array. It also builds a histogram of 
     *           the id values. And lastly, it stores the rank (order of insert) 
     *           of each minimum element within its corresponding histogram bin.
     *           For more details, look at the kernel's documentation.
     *  \note The `rbcMinDists` kernel is available in `kernels/rbc_kernels.cl`.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created by 
     *        a `RBCMin` instance:<br>
     *        |  Name  | Type | Placement | I/O | Use | Properties | Size |
     *        |  ---   |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN_D    | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$columns*rows*sizeof\ (cl\_float)\f$ |
     *        | H_OUT_ID  | Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$rows    *sizeof\ (rbc\_dist\_id)\f$ |
     *        | H_OUT_RNK | Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$rows         *sizeof\ (cl\_uint)\f$ |
     *        | H_OUT_N   | Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$columns      *sizeof\ (cl\_uint)\f$ |
     *        | D_IN_D    | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$columns*rows*sizeof\ (cl\_float)\f$ |
     *        | D_OUT_ID  | Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$rows    *sizeof\ (rbc\_dist\_id)\f$ |
     *        | D_OUT_RNK | Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$rows         *sizeof\ (cl\_uint)\f$ |
     *        | D_OUT_N   | Buffer | Device | O | Processing  | CL_MEM_READ_WRITE | \f$columns      *sizeof\ (cl\_uint)\f$ |
     *        `rbc_dist_id` is documented in `rbcDataTypes.hpp`.
     */
    class RBCMin
    {
    public:
        /*! \brief Enumerates the memory objects handled by the class.
         *  \note `H_*` names refer to staging buffers on the host.
         *  \note `D_*` names refer to buffers on the device.
         *  
         *  \param H_IN_D input staging buffer for the array of distances.
         *  \param H_OUT_ID output staging buffer with the minimum elements and id values.
         *  \param H_OUT_RNK output staging buffer with the indices of each database point within the associated list.
         *  \param H_OUT_N output staging buffer with the cardinalities of the representative lists.
         *  \param D_IN_D input buffer for the array of distances.
         *  \param D_MINS buffer of minimum elements and id values per work-group.
         *  \param D_OUT_ID output buffer with the minimum elements and id values.
         *  \param D_OUT_RNK output buffer with the indices of each database point within the associated list.
         *  \param D_OUT_N output buffer with the cardinalities of the representative lists.
         */
        enum class Memory : uint8_t { H_IN_D, H_OUT_ID, H_OUT_RNK, H_OUT_N, 
                                      D_IN_D, D_MINS, D_OUT_ID, D_OUT_RNK, D_OUT_N };

        /*! \brief Configures an OpenCL environment as specified by `_info`. */
        RBCMin (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info);
        /*! \brief Returns a reference to an internal memory object. */
        cl::Memory& get (RBCMin::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (unsigned int _cols, unsigned int _rows, int _accCounters = 1, Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (RBCMin::Memory mem = RBCMin::Memory::D_IN_D, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (RBCMin::Memory mem = RBCMin::Memory::H_OUT_ID, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);

        cl_float *hPtrInD;  /*!< Mapping of the input staging buffer. */
        rbc_dist_id *hPtrOutID;  /*!< Mapping of the output staging buffer with the minimum elements and id values. */
        cl_uint *hPtrOutRnk;  /*!< Mapping of the output staging buffer with the indices of each database point within the associated list. */
        cl_uint *hPtrOutN;  /*!< Mapping of the output staging buffer with the number of elements per representative list. */

    private:
        clutils::CLEnv &env;
        clutils::CLEnvInfo<1> info;
        cl::Context context;
        cl::CommandQueue queue;
        cl::Kernel initKernel, minsKernel, groupMinsKernel;
        cl::NDRange globalInit, globalM, globalGM, local;
        Staging staging;
        size_t wgMultiple, wgXdim;
        int accCounters;
        unsigned int cols, rows;
        unsigned int bufferDSize, bufferGMSize, bufferIDSize, bufferRnkSize, bufferNSize;
        cl::Buffer hBufferInD, hBufferOutID, hBufferOutRnk, hBufferOutN;
        cl::Buffer dBufferInD, dBufferGM, dBufferOutID, dBufferOutRnk, dBufferOutN;

    public:
        /*! \brief Executes the necessary kernels.
         *  \details This `run` instance is used for profiling.
         *  
         *  \param[in] timer `GPUTimer` that does the profiling of the kernel executions.
         *  \param[in] events a wait-list of events.
         *  \return Τhe total execution time measured by the timer.
         */
        template <typename period>
        double run (clutils::GPUTimer<period> &timer, const std::vector<cl::Event> *events = nullptr)
        {
            double pTime = 0.0;

            if (accCounters == 1)
            {
                queue.enqueueNDRangeKernel (initKernel, cl::NullRange, globalInit, cl::NullRange, events, &timer.event ());
                queue.flush (); timer.wait ();
                pTime = timer.duration ();
            }

            if (wgXdim == 1)
            {
                queue.enqueueNDRangeKernel (minsKernel, cl::NullRange, globalM, local, events, &timer.event ());
                queue.flush (); timer.wait ();
                pTime += timer.duration ();
            }
            else
            {
                queue.enqueueNDRangeKernel (minsKernel, cl::NullRange, globalM, local, events, &timer.event ());
                queue.flush (); timer.wait ();
                pTime += timer.duration ();

                queue.enqueueNDRangeKernel (groupMinsKernel, cl::NullRange, globalGM, local, nullptr, &timer.event ());
                queue.flush (); timer.wait ();
                pTime += timer.duration ();
            }

            return pTime;
        }

    };


    /*! \brief Enumerates configurations for the `Scan` class.
     * 
     *  \param INCLUSIVE identifies the case of `inclusive` scan.
     *  \param EXCLUSIVE identifies the case of `exclusive` scan.
     */
    enum class ScanConfig : uint8_t { INCLUSIVE, EXCLUSIVE };


    /*! \brief Interface class for the `scan` kernel.
     *  \details `scan` performs a scan operation on each row in an array. 
     *           For more details, look at the kernel's documentation.
     *  \note The `scan` kernel is available in `kernels/scan_kernels.cl`.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created by a `Scan` instance:<br>
     *        | Name | Type | Placement | I/O | Use | Properties | Size |
     *        | ---  |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$width*height*sizeof\ (cl\_int)\f$ |
     *        | H_OUT| Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$width*height*sizeof\ (cl\_int)\f$ |
     *        | D_IN | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$width*height*sizeof\ (cl\_int)\f$ |
     *        | D_OUT| Buffer | Device | O | Processing  | CL_MEM_READ_WRITE | \f$width*height*sizeof\ (cl\_int)\f$ |
     *  
     *  \tparam C configures the class to perform either `inclusive` or `exclusive` scan.
     *  \tparam T configures the class to work with different types of data.
     */
    template <ScanConfig C, typename T = cl_int>
    class Scan
    {
    public:
        /*! \brief Enumerates the memory objects handled by the class.
         *  \note `H_*` names refer to staging buffers on the host.
         *  \note `D_*` names refer to buffers on the device.
         *  
         *  \param H_IN input staging buffer.
         *  \param H_OUT output staging buffer.
         *  \param D_IN input buffer.
         *  \param D_SUMS buffer of partial group sums.
         *  \param D_OUT output buffer.
         */
        enum class Memory : uint8_t { H_IN, H_OUT, D_IN, D_SUMS, D_OUT };

        /*! \brief Configures an OpenCL environment as specified by `_info`. */
        Scan (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info);
        /*! \brief Returns a reference to an internal memory object. */
        cl::Memory& get (Scan::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (unsigned int _cols, unsigned int _rows, Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (Scan::Memory mem = Scan::Memory::D_IN, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (Scan::Memory mem = Scan::Memory::H_OUT, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);

        T *hPtrIn;  /*!< Mapping of the input staging buffer. */
        T *hPtrOut;  /*!< Mapping of the output staging buffer. */

    private:
        clutils::CLEnv &env;
        clutils::CLEnvInfo<1> info;
        cl::Context context;
        cl::CommandQueue queue;
        cl::Kernel kernelScan, kernelSumsScan, kernelAddSums;
        cl::NDRange globalScan, globalSumsScan, localScan;
        cl::NDRange globalAddSums, localAddSums, offsetAddSums;
        Staging staging;
        size_t wgMultiple, wgXdim;
        unsigned int cols, rows, bufferSize, bufferSumsSize;
        cl::Buffer hBufferIn, hBufferOut;
        cl::Buffer dBufferIn, dBufferOut, dBufferSums;

    public:
        /*! \brief Executes the necessary kernels.
         *  \details This `run` instance is used for profiling.
         *  
         *  \param[in] timer `GPUTimer` that does the profiling of the kernel executions.
         *  \param[in] events a wait-list of events.
         *  \return Τhe total execution time measured by the timer.
         */
        template <typename period>
        double run (clutils::GPUTimer<period> &timer, const std::vector<cl::Event> *events = nullptr)
        {
            double pTime;

            if (wgXdim == 1)
            {
                queue.enqueueNDRangeKernel (
                    kernelScan, cl::NullRange, globalScan, localScan, events, &timer.event ());
                queue.flush (); timer.wait ();
                pTime = timer.duration ();
            }
            else
            {
                queue.enqueueNDRangeKernel (
                    kernelScan, cl::NullRange, globalScan, localScan, events, &timer.event ());
                queue.flush (); timer.wait ();
                pTime = timer.duration ();

                queue.enqueueNDRangeKernel (
                    kernelSumsScan, cl::NullRange, globalSumsScan, localScan, nullptr, &timer.event ());
                queue.flush (); timer.wait ();
                pTime += timer.duration ();

                queue.enqueueNDRangeKernel (
                    kernelAddSums, offsetAddSums, globalAddSums, localAddSums, nullptr, &timer.event ());
                queue.flush (); timer.wait ();
                pTime += timer.duration ();
            }

            return pTime;
        }

    };


    /*! \brief Enumerates configurations for the `RBCPermute` class.
     * 
     *  \param GENERIC identifies the case of data of arbitrary dimensionality.
     *  \param KINECT identifies the case of Kinect point clouds in \f$ \mathbb{R}^8 \f$.
     */
    enum class RBCPermuteConfig : uint8_t { GENERIC, KINECT };


    /*! \brief Interface class for the `rbcPermute` kernel.
     *  \details `rbcPermute` permutes the database points to form the representative 
     *           lists and allow for coalesced access pattern during the search operation.
     *           For more details, look at the kernel's documentation.
     *  \note The `rbcPermute` kernel is available in `kernels/rbc_kernels.cl`.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created by a `RBCPermute` instance:<br>
     *        | Name | Type | Placement | I/O | Use | Properties | Size |
     *        | ---  |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN_X    | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$n_x*d  *sizeof\ (cl\_float)\f$ |
     *        | H_IN_ID   | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$n_x*sizeof\ (rbc\_dist\_id)\f$ |
     *        | H_IN_RNK  | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$n_x     *sizeof\ (cl\_uint)\f$ |
     *        | H_IN_O    | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$n_r     *sizeof\ (cl\_uint)\f$ |
     *        | H_OUT_X_P | Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$n_x*d  *sizeof\ (cl\_float)\f$ |
     *        | H_OUT_ID_P| Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$n_x*sizeof\ (rbc\_dist\_id)\f$ |
     *        | D_IN_X    | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$n_x*d  *sizeof\ (cl\_float)\f$ |
     *        | D_IN_ID   | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$n_x*sizeof\ (rbc\_dist\_id)\f$ |
     *        | D_IN_RNK  | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$n_x     *sizeof\ (cl\_uint)\f$ |
     *        | D_IN_O    | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$n_r     *sizeof\ (cl\_uint)\f$ |
     *        | D_OUT_X_P | Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$n_x*d  *sizeof\ (cl\_float)\f$ |
     *        | D_OUT_ID_P| Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$n_x*sizeof\ (rbc\_dist\_id)\f$ |
     *        
     *  \tparam C configures the class either for `generic` or `Kinect` data.
     */
    template <RBCPermuteConfig C = RBCPermuteConfig::GENERIC>
    class RBCPermute
    {
    public:
        /*! \brief Enumerates the memory objects handled by the class.
         *  \note `H_*` names refer to staging buffers on the host.
         *  \note `D_*` names refer to buffers on the device.
         *  
         *  \param H_IN_X input staging buffer for the database.
         *  \param H_IN_ID input staging buffer with the representative ids for each point.
         *  \param H_IN_RNK input staging buffer with the indices of the points within each rep. list.
         *  \param H_IN_O input staging buffer with the offsets of the representative lists within the db.
         *  \param H_OUT_X_P output staging buffer for the permuted database.
         *  \param H_OUT_ID_P output staging buffer with the representative ids for each point in the permuted db.
         *  \param D_IN_X input buffer for the database.
         *  \param D_IN_ID input buffer with the representative ids for each point.
         *  \param D_IN_RNK input buffer with the indices of the points within each rep. list.
         *  \param D_IN_O input buffer with the offsets of the representative lists within the db.
         *  \param D_OUT_X_P output buffer for the permuted database.
         *  \param D_OUT_ID_P output buffer with the representative ids for each point in the permuted db.
         */
        enum class Memory : uint8_t { H_IN_X, H_IN_ID, H_IN_RNK, H_IN_O, H_OUT_X_P, H_OUT_ID_P,  
                                      D_IN_X, D_IN_ID, D_IN_RNK, D_IN_O, D_OUT_X_P, D_OUT_ID_P };

        /*! \brief Configures an OpenCL environment as specified by `_info`. */
        RBCPermute (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info);
        /*! \brief Returns a reference to an internal memory object. */
        cl::Memory& get (RBCPermute::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (unsigned int _nx, unsigned int _nr, unsigned int _d = 8, int _permID = 0, Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (RBCPermute::Memory mem = RBCPermute::Memory::D_IN_X, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (RBCPermute::Memory mem = RBCPermute::Memory::H_OUT_X_P, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);

        cl_float *hPtrInX;  /*!< Mapping of the input staging buffer for the database. */
        rbc_dist_id *hPtrInID;  /*!< Mapping of the input staging buffer with the representative ids for each point. */
        cl_uint *hPtrInRnk;  /*!< Mapping of the input staging buffer with the indices of the points within each rep. list. */
        cl_uint *hPtrInO;  /*!< Mapping of the input staging buffer with the offsets of the representative lists within the db. */
        cl_float *hPtrOutXp;  /*!< Mapping of the output staging buffer for the permuted database. */
        rbc_dist_id *hPtrOutIDp;  /*!< Mapping of the output staging buffer with the representative ids for each point in the permuted db. */

    private:
        clutils::CLEnv &env;
        clutils::CLEnvInfo<1> info;
        cl::Context context;
        cl::CommandQueue queue;
        cl::Kernel kernel;
        cl::NDRange global;
        Staging staging;
        int permID;
        unsigned int nx, nr, d;
        unsigned int bufferXSize, bufferIDSize, bufferRnkSize, bufferOSize;
        cl::Buffer hBufferInX, hBufferInID, hBufferInRnk, hBufferInO, hBufferOutXp, hBufferOutIDp;
        cl::Buffer dBufferInX, dBufferInID, dBufferInRnk, dBufferInO, dBufferOutXp, dBufferOutIDp;

    public:
        /*! \brief Executes the necessary kernels.
         *  \details This `run` instance is used for profiling.
         *  
         *  \param[in] timer `GPUTimer` that does the profiling of the kernel executions.
         *  \param[in] events a wait-list of events.
         *  \return Τhe total execution time measured by the timer.
         */
        template <typename period>
        double run (clutils::GPUTimer<period> &timer, const std::vector<cl::Event> *events = nullptr)
        {
            queue.enqueueNDRangeKernel (kernel, cl::NullRange, global, cl::NullRange, events, &timer.event ());
            queue.flush (); timer.wait ();
            
            return timer.duration ();
        }

    };


    /*! \brief Interface class for constructing the `Random Ball Cover` data structure.
     *  \details The process involves finding the representative lists and permuting
     *           the database points to form the lists within the database.
     *           The `RBC` data structure built is the one described in the exact 
     *           search algorithm. For more details, see [here][2].
     *           [2]: http://www.lcayton.com/rbc.pdf
     *  \note The kernels used in this process are available in 
     *        `kernels/rbc_kernels.cl` and `kernels/scan_kernels.cl`.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created by a `RBCConstruct` instance:<br>
     *        | Name | Type | Placement | I/O | Use | Properties | Size |
     *        | ---  |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN_X    | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$n_x*d  *sizeof\ (cl\_float)\f$ |
     *        | H_IN_R    | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$n_r*d  *sizeof\ (cl\_float)\f$ |
     *        | H_OUT_ID  | Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$n_x*sizeof\ (rbc\_dist\_id)\f$ |
     *        | H_OUT_RNK | Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$n_x     *sizeof\ (cl\_uint)\f$ |
     *        | H_OUT_N   | Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$n_r     *sizeof\ (cl\_uint)\f$ |
     *        | H_OUT_O   | Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$n_r     *sizeof\ (cl\_uint)\f$ |
     *        | H_OUT_X_P | Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$n_x*d  *sizeof\ (cl\_float)\f$ |
     *        | H_OUT_ID_P| Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$n_x*sizeof\ (rbc\_dist\_id)\f$ |
     *        | D_IN_X    | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$n_x*d  *sizeof\ (cl\_float)\f$ |
     *        | D_IN_R    | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$n_r*d  *sizeof\ (cl\_float)\f$ |
     *        | D_OUT_ID  | Buffer | Device | O | Processing  | CL_MEM_READ_WRITE | \f$n_x*sizeof\ (rbc\_dist\_id)\f$ |
     *        | D_OUT_RNK | Buffer | Device | O | Processing  | CL_MEM_READ_WRITE | \f$n_x     *sizeof\ (cl\_uint)\f$ |
     *        | D_OUT_N   | Buffer | Device | O | Processing  | CL_MEM_READ_WRITE | \f$n_r     *sizeof\ (cl\_uint)\f$ |
     *        | D_OUT_O   | Buffer | Device | O | Processing  | CL_MEM_READ_WRITE | \f$n_r     *sizeof\ (cl\_uint)\f$ |
     *        | D_OUT_X_P | Buffer | Device | O | Processing  | CL_MEM_READ_WRITE | \f$n_x*d  *sizeof\ (cl\_float)\f$ |
     *        | D_OUT_ID_P| Buffer | Device | O | Processing  | CL_MEM_READ_WRITE | \f$n_x*sizeof\ (rbc\_dist\_id)\f$ |
     *        
     *  \tparam K configures the class for using one of the available kernels for computing the point distances.
     *  \tparam P configures the class for using one of the available kernels for the database permutation.
     */
    template <KernelTypeC K, RBCPermuteConfig P>
    class RBCConstruct
    {
    public:
        /*! \brief Enumerates the memory objects handled by the class.
         *  \note `H_*` names refer to staging buffers on the host.
         *  \note `D_*` names refer to buffers on the device.
         *  
         *  \param H_IN_X input staging buffer for the database.
         *  \param H_IN_R input staging buffer for the representatives.
         *  \param H_OUT_ID output staging buffer with the representative ids for each db point.
         *  \param H_OUT_RNK output staging buffer with the indices of the points within each rep. list.
         *  \param H_OUT_N outut staging buffer with the cardinalities of the representative lists.
         *  \param H_OUT_O outut staging buffer with the offsets of the representative lists within the db.
         *  \param H_OUT_X_P output staging buffer for the permuted database.
         *  \param H_OUT_ID_P output staging buffer with the representative ids for each point in the permuted db.
         *  \param D_IN_X input buffer for the database.
         *  \param D_IN_R input buffer for the representatives.
         *  \param D_OUT_D output buffer for the array of distances.
         *  \param D_OUT_ID output buffer with the representative ids for each db point.
         *  \param D_OUT_RNK output buffer with the indices of the points within each rep. list.
         *  \param D_OUT_N outut buffer with the cardinalities of the representative lists.
         *  \param D_OUT_O outut buffer with the offsets of the representative lists within the db.
         *  \param D_OUT_X_P output buffer for the permuted database.
         *  \param D_OUT_ID_P output buffer with the representative ids for each point in the permuted db.
         */
        enum class Memory : uint8_t { H_IN_X, H_IN_R, H_OUT_ID, H_OUT_RNK, H_OUT_N, H_OUT_O, H_OUT_X_P, H_OUT_ID_P, 
                                      D_IN_X, D_IN_R, D_OUT_D, D_OUT_ID, D_OUT_RNK, D_OUT_N, D_OUT_O, D_OUT_X_P, D_OUT_ID_P };

        /*! \brief Configures an OpenCL environment as specified by `_info`. */
        RBCConstruct (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info);
        /*! \brief Returns a reference to an internal memory object. */
        cl::Memory& get (RBCConstruct::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (unsigned int _nx, unsigned int _nr, unsigned int _d, float _a = 1.f, int _permID = 0, Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (RBCConstruct::Memory mem = RBCConstruct::Memory::D_IN_X, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (RBCConstruct::Memory mem = RBCConstruct::Memory::H_OUT_X_P, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Gets the scaling factor \f$ \alpha \f$. */
        float getAlpha ();
        /*! \brief Sets the scaling factor \f$ \alpha \f$. */
        void setAlpha (float _a);

        cl_float *hPtrInX;  /*!< Mapping of the input staging buffer for the database. */
        cl_float *hPtrInR;  /*!< Mapping of the input staging buffer for the representatives. */
        rbc_dist_id *hPtrOutID;  /*!< Mapping of the output staging buffer with the representative ids for each point. */
        cl_uint *hPtrOutRnk;  /*!< Mapping of the output staging buffer with the indices of the points within each rep. list. */
        cl_uint *hPtrOutN;  /*!< Mapping of the output staging buffer with the cardinalities of the representative lists. */
        cl_uint *hPtrOutO;  /*!< Mapping of the output staging buffer with the offsets of the representative lists within the db. */
        cl_float *hPtrOutXp;  /*!< Mapping of the output staging buffer for the permuted database. */
        rbc_dist_id *hPtrOutIDp;  /*!< Mapping of the output staging buffer with the representative ids for each point in the permuted db. */

    private:
        clutils::CLEnv &env;
        clutils::CLEnvInfo<1> info;
        cl::Context context;
        cl::CommandQueue queue;
        RBCComputeDists<K> rbcCompDists;
        RBCMin rbcMinDists;
        Scan<ScanConfig::EXCLUSIVE> rbcScanNLists;
        RBCPermute<P> rbcPermDB;
        Staging staging;
        int permID;
        unsigned int nx, nr, d;
        unsigned int bufferXSize, bufferRSize, bufferDSize, bufferIDSize;
        unsigned int bufferNSize, bufferOSize, bufferRnkSize;
        cl::Buffer hBufferInX, hBufferInR, hBufferOutID, hBufferOutRnk, hBufferOutN, hBufferOutO, hBufferOutXp, hBufferOutIDp;
        cl::Buffer dBufferInX, dBufferInR, dBufferOutID, dBufferOutRnk, dBufferOutN, dBufferOutO, dBufferOutXp, dBufferOutIDp;

    public:
        /*! \brief Executes the necessary kernels.
         *  \details This `run` instance is used for profiling.
         *  
         *  \param[in] timer `GPUTimer` that does the profiling of the kernel executions.
         *  \param[in] events a wait-list of events.
         *  \return Τhe total execution time measured by the timer.
         */
        template <typename period>
        double run (clutils::GPUTimer<period> &timer, const std::vector<cl::Event> *events = nullptr)
        {
            double pTime;

            pTime = rbcCompDists.run (timer, events);
            pTime += rbcMinDists.run (timer);
            pTime += rbcScanNLists.run (timer);
            pTime += rbcPermDB.run (timer);
            
            return pTime;
        }

    };


    /*! \brief Enumerates the kernels available for computing 
     *         the array of distances (`Q-X[L]`) during search.
     *  
     *  \param GENERIC refers to the `rbcComputeQXDists` kernel.
     *  \param KINECT refers to the `rbcComputeQXDists_Kinect` kernel.
     */
    enum class KernelTypeS : uint8_t { GENERIC, KINECT };


    /*! \brief Interface class for searching for nearest neighbors, of a set of 
     *         queries, in the `Random Ball Cover` data structure.
     *  \details The process involves finding the representative of each query and 
     *           then searching in their representative's list for their NN.
     *           The algorithm is the one described in the one shot algorithm. 
     *           For more details, see [here][http://www.lcayton.com/rbc.pdf].
     *  \note The kernels used in this process are available in `kernels/rbc_kernels.cl`, 
     *        `kernels/reduce_kernels.cl` and `kernels/scan_kernels.cl`.
     *  \note This is just a declaration. Look at the explicit template specializations
     *        for specific instantiations of the class.
     *        
     *  \tparam K configures the class for using one of the available kernels for computing the `Q-R` distances.
     *  \tparam P configures the class for using one of the available kernels for the database permutation.
     *  \tparam S configures the class for using one of the available kernels for computing the `Q-X[L]` distances.
     */
    template <KernelTypeC K, RBCPermuteConfig P, KernelTypeS S>
    class RBCSearch;


    /*! \brief Interface class for searching for nearest neighbors, of a set of 
     *         queries, in the `Random Ball Cover` data structure.
     *  \details The process involves finding the representative of each query and 
     *           then searching in their representative's list for their NN.
     *           The algorithm is the one described in the one shot algorithm. 
     *           For more details, see [here][http://www.lcayton.com/rbc.pdf].
     *  \note The kernels used in this process are available in `kernels/rbc_kernels.cl`, 
     *        `kernels/reduce_kernels.cl` and `kernels/scan_kernels.cl`.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created by a `RBCSearch` instance:<br>
     *        | Name | Type | Placement | I/O | Use | Properties | Size |
     *        | ---  |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN_Q      | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$n_q*d  *sizeof\ (cl\_float)\f$ |
     *        | H_IN_R      | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$n_r*d  *sizeof\ (cl\_float)\f$ |
     *        | H_IN_X_P    | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$n_x*d  *sizeof\ (cl\_float)\f$ |
     *        | H_IN_O      | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$n_r     *sizeof\ (cl\_uint)\f$ |
     *        | H_IN_N      | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$n_r     *sizeof\ (cl\_uint)\f$ |
     *        | H_OUT_R_ID  | Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$n_q*sizeof\ (rbc\_dist\_id)\f$ |
     *        | H_OUT_Q_P   | Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$n_q*d  *sizeof\ (cl\_float)\f$ |
     *        | H_OUT_NN_ID | Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$n_q*sizeof\ (rbc\_dist\_id)\f$ |
     *        | H_OUT_NN    | Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$n_q*d  *sizeof\ (cl\_float)\f$ |
     *        | D_IN_Q      | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$n_q*d  *sizeof\ (cl\_float)\f$ |
     *        | D_IN_R      | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$n_r*d  *sizeof\ (cl\_float)\f$ |
     *        | D_IN_X_P    | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$n_x*d  *sizeof\ (cl\_float)\f$ |
     *        | D_IN_O      | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$n_r     *sizeof\ (cl\_uint)\f$ |
     *        | D_IN_N      | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$n_r     *sizeof\ (cl\_uint)\f$ |
     *        | D_OUT_R_ID  | Buffer | Device | O | Processing  | CL_MEM_READ_WRITE | \f$n_q*sizeof\ (rbc\_dist\_id)\f$ |
     *        | D_OUT_Q_P   | Buffer | Device | O | Processing  | CL_MEM_READ_WRITE | \f$n_q*d  *sizeof\ (cl\_float)\f$ |
     *        | D_OUT_NN_ID | Buffer | Device | O | Processing  | CL_MEM_READ_WRITE | \f$n_q*sizeof\ (rbc\_dist\_id)\f$ |
     *        | D_OUT_NN    | Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$n_q*d  *sizeof\ (cl\_float)\f$ |
     *        
     *  \tparam K configures the class for using one of the available kernels for computing the `Q-R` distances.
     *  \tparam P configures the class for using one of the available kernels for the database permutation.
     */
    template <KernelTypeC K, RBCPermuteConfig P>
    class RBCSearch<K, P, KernelTypeS::GENERIC>
    {
    public:
        /*! \brief Enumerates the memory objects handled by the class.
         *  \note `H_*` names refer to staging buffers on the host.
         *  \note `D_*` names refer to buffers on the device.
         *  
         *  \param H_IN_Q input staging buffer for the queries.
         *  \param H_IN_R input staging buffer for the representatives.
         *  \param H_IN_X_P input staging buffer for the permuted database.
         *  \param H_IN_O input staging buffer with the offsets of the representative lists within the db.
         *  \param H_IN_N input staging buffer with the cardinalities of the representative lists.
         *  \param H_OUT_R_ID output staging buffer with the representative ids for each query in Qp.
         *  \param H_OUT_Q_P output staging buffer for the permuted queries.
         *  \param H_OUT_NN_ID output staging buffer with the NN ids (relative to the NN's rep list) 
         *                     for each query in Qp.
         *  \param H_OUT_NN output staging buffer with the queries' nearest neighbors. The NNs correspond
         *                  to the queries in the permuted array, Qp.
         *  \param D_IN_Q input buffer for the queries.
         *  \param D_IN_R input buffer for the representatives.
         *  \param D_IN_X_P input buffer for the permuted database.
         *  \param D_IN_O input buffer with the offsets of the representative lists within the db.
         *  \param D_IN_N input buffer with the cardinalities of the representative lists.
         *  \param D_OUT_R_ID output buffer with the representative ids for each query in Qp.
         *  \param D_OUT_Q_P output buffer for the permuted queries.
         *  \param D_OUT_NN_ID output buffer with the NN ids (relative to the NN's rep list) 
         *                     for each query in Qp.
         *  \param D_OUT_NN output buffer with the queries' nearest neighbors. The NNs correspond
         *                  to the queries in the permuted array, Qp.
         *  \param D_OUT_QR_D output buffer for the array of query distances from the representatives.
         *  \param D_OUT_QX_D output buffer for the array of query distances from the points in their rep's list.
         */
        enum class Memory : uint8_t { 
            H_IN_Q, H_IN_R, H_IN_X_P, H_IN_O, H_IN_N, H_OUT_R_ID, H_OUT_Q_P, H_OUT_NN_ID, H_OUT_NN, 
            D_IN_Q, D_IN_R, D_IN_X_P, D_IN_O, D_IN_N, D_OUT_R_ID, D_OUT_Q_P, D_OUT_NN_ID, D_OUT_NN, 
            D_OUT_QR_D, D_OUT_QX_D };

        /*! \brief Configures an OpenCL environment as specified by `_info`. */
        RBCSearch (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info);
        /*! \brief Returns a reference to an internal memory object. */
        cl::Memory& get (RBCSearch::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (unsigned int _nq, unsigned int _nr, unsigned int _nx, unsigned int _d, Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (RBCSearch::Memory mem = RBCSearch::Memory::D_IN_Q, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (RBCSearch::Memory mem = RBCSearch::Memory::H_OUT_NN, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);

        cl_float *hPtrInQ;  /*!< Mapping of the input staging buffer for the queries. */
        cl_float *hPtrInR;  /*!< Mapping of the input staging buffer for the representatives. */
        cl_float *hPtrInXp;  /*!< Mapping of the input staging buffer for the permuted database. */
        cl_uint *hPtrInO;  /*!< Mapping of the input staging buffer with the offsets of the representative lists within the db. */
        cl_uint *hPtrInN;  /*!< Mapping of the input staging buffer with the cardinalities of the representative lists. */
        rbc_dist_id *hPtrOutRID;  /*!< Mapping of the output staging buffer with the representative ids for each query. */
        cl_float *hPtrOutQp;  /*!< Mapping of the output staging buffer for the permuted queries. */
        rbc_dist_id *hPtrOutNNID;  /*!< Mapping of the output staging buffer with the NN ids for each query. */
        cl_float *hPtrOutNN;  /*!< Mapping of the output staging buffer for the query NNs. */

    private:
        void setExecParams ();

        clutils::CLEnv &env;
        clutils::CLEnvInfo<1> info;
        cl::Context context;
        cl::CommandQueue queue;
        cl::NDRange globalQXD, localQXD, globalNNID, globalGNNID, globalNN, local;
        cl::Kernel rbcCompQXDistsKernel, nnidMinsKernel, nnidGroupMinsKernel, rbcNNKernel;
        RBCConstruct<K, P> rbcCompRIDs;
        Reduce<ReduceConfig::MAX, cl_uint> compMaxN;
        Staging staging;
        size_t wgMultiple, wgXdim;
        unsigned int nq, nr, nx, d, max_n;
        unsigned int bufferQSize, bufferRSize, bufferXSize, bufferOSize, bufferNSize;
        unsigned int bufferQXDSize, bufferRIDSize;
        unsigned int bufferNNIDSize, bufferGNNIDSize, bufferNNSize;
        cl::Buffer hBufferInQ, hBufferInR, hBufferInXp, hBufferInO, hBufferInN;
        cl::Buffer dBufferInQ, dBufferInR, dBufferInXp, dBufferInO, dBufferInN;
        cl::Buffer hBufferOutRID, hBufferOutQp, hBufferOutNNID, hBufferOutNN;
        cl::Buffer dBufferOutRID, dBufferOutQp, dBufferOutNNID, dBufferOutNN;
        cl::Buffer dBufferOutQXD, dBufferOutGNNID;

    public:
        /*! \brief Executes the necessary kernels.
         *  \details This `run` instance is used for profiling.
         *  
         *  \param[in] timer `GPUTimer` that does the profiling of the kernel executions.
         *  \param[in] events a wait-list of events.
         *  \return Τhe total execution time measured by the timer.
         */
        template <typename period>
        double run (clutils::GPUTimer<period> &timer, const std::vector<cl::Event> *events = nullptr)
        {
            double pTime;

            // compMaxN is not profiled. Its cost is incurred 
            // only once, and it's expected to be insignificant
            if (wgXdim == 0) setExecParams ();

            // Compute nearest representatives
            pTime = rbcCompRIDs.run (timer, events);

            // Compute distances from the points in the representative lists
            queue.enqueueNDRangeKernel (rbcCompQXDistsKernel, 
                cl::NullRange, globalQXD, localQXD, nullptr, &timer.event ());
            queue.flush (); timer.wait ();
            pTime += timer.duration ();

            // Compute NN ids
            queue.enqueueNDRangeKernel (nnidMinsKernel, 
                cl::NullRange, globalNNID, local, nullptr, &timer.event ());
            queue.flush (); timer.wait ();
            pTime += timer.duration ();

            if (wgXdim > 1)
            {
                queue.enqueueNDRangeKernel (nnidGroupMinsKernel, 
                    cl::NullRange, globalGNNID, local, nullptr, &timer.event ());
                queue.flush (); timer.wait ();
                pTime += timer.duration ();
            }
            
            // Collect NNs
            queue.enqueueNDRangeKernel (rbcNNKernel, 
                cl::NullRange, globalNN, cl::NullRange, nullptr, &timer.event ());
            queue.flush (); timer.wait ();
            pTime += timer.duration ();

            return pTime;
        }

    };


    /*! \brief Interface class for searching for nearest neighbors, of a set of 
     *         queries, in the `Random Ball Cover` data structure.
     *  \details The process involves finding the representative of each query and 
     *           then searching in their representative's list for their NN.
     *           The algorithm is the one described in the one shot algorithm. 
     *           For more details, see [here][http://www.lcayton.com/rbc.pdf].
     *  \note The kernels used in this process are available in `kernels/rbc_kernels.cl`, 
     *        `kernels/reduce_kernels.cl` and `kernels/scan_kernels.cl`.
     *  \note The class creates its own buffers. If you would like to provide 
     *        your own buffers, call `get` to get references to the placeholders 
     *        within the class and assign them to your buffers. You will have to 
     *        do this strictly before the call to `init`. You can also call `get` 
     *        (after the call to `init`) to get a reference to a buffer within 
     *        the class and assign it to another kernel class instance further 
     *        down in your task pipeline.
     *  
     *        The following input/output `OpenCL` memory objects are created by a `RBCSearch` instance:<br>
     *        | Name | Type | Placement | I/O | Use | Properties | Size |
     *        | ---  |:---: |   :---:   |:---:|:---:|   :---:    |:---: |
     *        | H_IN_Q      | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$n_q*d  *sizeof\ (cl\_float)\f$ |
     *        | H_IN_R      | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$n_r*d  *sizeof\ (cl\_float)\f$ |
     *        | H_IN_X_P    | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$n_x*d  *sizeof\ (cl\_float)\f$ |
     *        | H_IN_O      | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$n_r     *sizeof\ (cl\_uint)\f$ |
     *        | H_IN_N      | Buffer | Host   | I | Staging     | CL_MEM_READ_WRITE | \f$n_r     *sizeof\ (cl\_uint)\f$ |
     *        | H_OUT_R_ID  | Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$n_q*sizeof\ (rbc\_dist\_id)\f$ |
     *        | H_OUT_Q_P   | Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$n_q*d  *sizeof\ (cl\_float)\f$ |
     *        | H_OUT_NN_ID | Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$n_q*sizeof\ (rbc\_dist\_id)\f$ |
     *        | H_OUT_NN    | Buffer | Host   | O | Staging     | CL_MEM_READ_WRITE | \f$n_q*d  *sizeof\ (cl\_float)\f$ |
     *        | D_IN_Q      | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$n_q*d  *sizeof\ (cl\_float)\f$ |
     *        | D_IN_R      | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$n_r*d  *sizeof\ (cl\_float)\f$ |
     *        | D_IN_X_P    | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$n_x*d  *sizeof\ (cl\_float)\f$ |
     *        | D_IN_O      | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$n_r     *sizeof\ (cl\_uint)\f$ |
     *        | D_IN_N      | Buffer | Device | I | Processing  | CL_MEM_READ_ONLY  | \f$n_r     *sizeof\ (cl\_uint)\f$ |
     *        | D_OUT_R_ID  | Buffer | Device | O | Processing  | CL_MEM_READ_WRITE | \f$n_q*sizeof\ (rbc\_dist\_id)\f$ |
     *        | D_OUT_Q_P   | Buffer | Device | O | Processing  | CL_MEM_READ_WRITE | \f$n_q*d  *sizeof\ (cl\_float)\f$ |
     *        | D_OUT_NN_ID | Buffer | Device | O | Processing  | CL_MEM_READ_WRITE | \f$n_q*sizeof\ (rbc\_dist\_id)\f$ |
     *        | D_OUT_NN    | Buffer | Device | O | Processing  | CL_MEM_WRITE_ONLY | \f$n_q*d  *sizeof\ (cl\_float)\f$ |
     *        
     *  \tparam K configures the class for using one of the available kernels for computing the `Q-R` distances.
     *  \tparam P configures the class for using one of the available kernels for the database permutation.
     */
    template <KernelTypeC K, RBCPermuteConfig P>
    class RBCSearch<K, P, KernelTypeS::KINECT>
    {
    public:
        /*! \brief Enumerates the memory objects handled by the class.
         *  \note `H_*` names refer to staging buffers on the host.
         *  \note `D_*` names refer to buffers on the device.
         *  
         *  \param H_IN_Q input staging buffer for the queries.
         *  \param H_IN_R input staging buffer for the representatives.
         *  \param H_IN_X_P input staging buffer for the permuted database.
         *  \param H_IN_O input staging buffer with the offsets of the representative lists within the db.
         *  \param H_IN_N input staging buffer with the cardinalities of the representative lists.
         *  \param H_OUT_R_ID output staging buffer with the representative ids for each query in Qp.
         *  \param H_OUT_Q_P output staging buffer for the permuted queries.
         *  \param H_OUT_NN_ID output staging buffer with the NN ids (relative to the NN's rep list) 
         *                     for each query in Qp.
         *  \param H_OUT_NN output staging buffer with the queries' nearest neighbors. The NNs correspond
         *                  to the queries in the permuted array, Qp.
         *  \param D_IN_Q input buffer for the queries.
         *  \param D_IN_R input buffer for the representatives.
         *  \param D_IN_X_P input buffer for the permuted database.
         *  \param D_IN_O input buffer with the offsets of the representative lists within the db.
         *  \param D_IN_N input buffer with the cardinalities of the representative lists.
         *  \param D_OUT_R_ID output buffer with the representative ids for each query in Qp.
         *  \param D_OUT_Q_P output buffer for the permuted queries.
         *  \param D_OUT_NN_ID output buffer with the NN ids (relative to the NN's rep list) 
         *                     for each query in Qp.
         *  \param D_OUT_NN output buffer with the queries' nearest neighbors. The NNs correspond
         *                  to the queries in the permuted array, Qp.
         *  \param D_OUT_QR_D output buffer for the array of query distances from the representatives.
         *  \param D_OUT_QX_D output buffer for the array of query distances from the points in their rep's list.
         */
        enum class Memory : uint8_t { 
            H_IN_Q, H_IN_R, H_IN_X_P, H_IN_O, H_IN_N, H_OUT_R_ID, H_OUT_Q_P, H_OUT_NN_ID, H_OUT_NN, 
            D_IN_Q, D_IN_R, D_IN_X_P, D_IN_O, D_IN_N, D_OUT_R_ID, D_OUT_Q_P, D_OUT_NN_ID, D_OUT_NN, 
            D_OUT_QR_D, D_OUT_QX_D };

        /*! \brief Configures an OpenCL environment as specified by `_info`. */
        RBCSearch (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info);
        /*! \brief Returns a reference to an internal memory object. */
        cl::Memory& get (RBCSearch::Memory mem);
        /*! \brief Configures kernel execution parameters. */
        void init (unsigned int _nq, unsigned int _nr, unsigned int _nx, float _a = 1.f, Staging _staging = Staging::IO);
        /*! \brief Performs a data transfer to a device buffer. */
        void write (RBCSearch::Memory mem = RBCSearch::Memory::D_IN_Q, void *ptr = nullptr, bool block = CL_FALSE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Performs a data transfer to a staging buffer. */
        void* read (RBCSearch::Memory mem = RBCSearch::Memory::H_OUT_NN, bool block = CL_TRUE, 
                    const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Executes the necessary kernels. */
        void run (const std::vector<cl::Event> *events = nullptr, cl::Event *event = nullptr);
        /*! \brief Gets the scaling factor \f$ \alpha \f$. */
        float getAlpha ();
        /*! \brief Sets the scaling factor \f$ \alpha \f$. */
        void setAlpha (float _a);

        cl_float *hPtrInQ;  /*!< Mapping of the input staging buffer for the queries. */
        cl_float *hPtrInR;  /*!< Mapping of the input staging buffer for the representatives. */
        cl_float *hPtrInXp;  /*!< Mapping of the input staging buffer for the permuted database. */
        cl_uint *hPtrInO;  /*!< Mapping of the input staging buffer with the offsets of the representative lists within the db. */
        cl_uint *hPtrInN;  /*!< Mapping of the input staging buffer with the cardinalities of the representative lists. */
        rbc_dist_id *hPtrOutRID;  /*!< Mapping of the output staging buffer with the representative ids for each query. */
        cl_float *hPtrOutQp;  /*!< Mapping of the output staging buffer for the permuted queries. */
        rbc_dist_id *hPtrOutNNID;  /*!< Mapping of the output staging buffer with the NN ids for each query. */
        cl_float *hPtrOutNN;  /*!< Mapping of the output staging buffer for the query NNs. */

    private:
        void setExecParams ();

        clutils::CLEnv &env;
        clutils::CLEnvInfo<1> info;
        cl::Context context;
        cl::CommandQueue queue;
        cl::NDRange globalQXD, localQXD, globalNNID, globalGNNID, globalNN, local;
        cl::Kernel rbcCompQXDistsKernel, nnidMinsKernel, nnidGroupMinsKernel, rbcNNKernel;
        RBCConstruct<K, P> rbcCompRIDs;
        Reduce<ReduceConfig::MAX, cl_uint> compMaxN;
        Staging staging;
        float a;
        size_t wgMultiple, wgXdim;
        unsigned int nq, nr, nx, d, max_n;
        unsigned int bufferQSize, bufferRSize, bufferXSize, bufferOSize, bufferNSize;
        unsigned int bufferQXDSize, bufferRIDSize;
        unsigned int bufferNNIDSize, bufferGNNIDSize, bufferNNSize;
        cl::Buffer hBufferInQ, hBufferInR, hBufferInXp, hBufferInO, hBufferInN;
        cl::Buffer dBufferInQ, dBufferInR, dBufferInXp, dBufferInO, dBufferInN;
        cl::Buffer hBufferOutRID, hBufferOutQp, hBufferOutNNID, hBufferOutNN;
        cl::Buffer dBufferOutRID, dBufferOutQp, dBufferOutNNID, dBufferOutNN;
        cl::Buffer dBufferOutQXD, dBufferOutGNNID;

    public:
        /*! \brief Executes the necessary kernels.
         *  \details This `run` instance is used for profiling.
         *  
         *  \param[in] timer `GPUTimer` that does the profiling of the kernel executions.
         *  \param[in] events a wait-list of events.
         *  \return Τhe total execution time measured by the timer.
         */
        template <typename period>
        double run (clutils::GPUTimer<period> &timer, const std::vector<cl::Event> *events = nullptr)
        {
            double pTime;

            // compMaxN is not profiled. Its cost is incurred 
            // only once, and it's expected to be insignificant
            if (wgXdim == 0) setExecParams ();

            // Compute nearest representatives
            pTime = rbcCompRIDs.run (timer, events);

            // Compute distances from the points in the representative lists
            queue.enqueueNDRangeKernel (rbcCompQXDistsKernel, 
                cl::NullRange, globalQXD, localQXD, nullptr, &timer.event ());
            queue.flush (); timer.wait ();
            pTime += timer.duration ();

            // Compute NN ids
            queue.enqueueNDRangeKernel (nnidMinsKernel, 
                cl::NullRange, globalNNID, local, nullptr, &timer.event ());
            queue.flush (); timer.wait ();
            pTime += timer.duration ();

            if (wgXdim > 1)
            {
                queue.enqueueNDRangeKernel (nnidGroupMinsKernel, 
                    cl::NullRange, globalGNNID, local, nullptr, &timer.event ());
                queue.flush (); timer.wait ();
                pTime += timer.duration ();
            }
            
            // Collect NNs
            queue.enqueueNDRangeKernel (rbcNNKernel, 
                cl::NullRange, globalNN, cl::NullRange, nullptr, &timer.event ());
            queue.flush (); timer.wait ();
            pTime += timer.duration ();

            return pTime;
        }

    };

}
}

#endif  // ALGORITHMS_HPP
