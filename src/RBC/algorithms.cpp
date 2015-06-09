/*! \file algorithms.cpp
 *  \brief Defines classes that organize the execution of OpenCL kernels.
 *  \details Each class hides the details of the execution of a kernel. They
 *           initialize the necessary buffers, set up the workspaces, and 
 *           run the kernels.
 *  \author Nick Lamprianidis
 *  \version 1.2.0
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
#include <sstream>
#include <cmath>
#include <CLUtils.hpp>
#include <RBC/algorithms.hpp>


/*! \note All the classes assume there is a fully configured `clutils::CLEnv` 
 *        environment. This means, there is a known context on which they will 
 *        operate, there is a known command queue which they will use, and all 
 *        the necessary kernel code has been compiled. For more info on **CLUtils**, 
 *        you can check the [online documentation](http://clutils.paign10.me/).
 */
namespace cl_algo
{
namespace RBC
{

    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. Specifies the context, queue, etc, to be used.
     */
    template <KernelTypeC K>
    RBCComputeDists<K>::RBCComputeDists (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0]))
    {
        std::string kernel_name;
        switch (K)
        {
            case KernelTypeC::SHARED_NONE:
                kernel_name += "rbcComputeDists_SharedNone";
                break;
            case KernelTypeC::SHARED_R:
                kernel_name += "rbcComputeDists_SharedR";
                break;
            case KernelTypeC::SHARED_X_R:
                kernel_name += "rbcComputeDists_SharedXR";
                break;
            case KernelTypeC::KINECT:
                kernel_name += "rbcComputeDists_Kinect";
                break;
            case KernelTypeC::KINECT_R:
                kernel_name += "rbcComputeDists_Kinect_R";
                break;
            case KernelTypeC::KINECT_X_R:
                kernel_name += "rbcComputeDists_Kinect_XR";
                break;
        }

        kernel = cl::Kernel (env.getProgram (info.pgIdx), kernel_name.c_str ());
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    template <KernelTypeC K>
    cl::Memory& RBCComputeDists<K>::get (RBCComputeDists::Memory mem)
    {
        switch (mem)
        {
            case RBCComputeDists::Memory::H_IN_X:
                return hBufferInX;
            case RBCComputeDists::Memory::H_IN_R:
                return hBufferInR;
            case RBCComputeDists::Memory::H_OUT_D:
                return hBufferOutD;
            case RBCComputeDists::Memory::D_IN_X:
                return dBufferInX;
            case RBCComputeDists::Memory::D_IN_R:
                return dBufferInR;
            case RBCComputeDists::Memory::D_OUT_D:
                return dBufferOutD;
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *        
     *  \param[in] _nx number of database points.
     *  \param[in] _nr number of representative points.
     *  \param[in] _d dimensionality of the associated points.
     *  \param[in] _a factor scaling the results of the distance calculations for the 
     *                geometric \f$ x_g \f$ and photometric \f$ x_p \f$ dimensions of 
     *                the \f$ x\epsilon\mathbb{R}^8 \f$ points. That is, \f$ \|x-x'\|_2^2= 
     *                f_g(a)\|x_g-x'_g\|_2^2+f_p(a)\|x_p-x'_p\|_2^2 \f$. This parameter is 
     *                applicable when involving the "Kinect" kernels. That is, when the 
     *                template parameter, `K`, gets the value `KINECT`, `KINECT_R`, or 
     *                `KINECT_X_R`. For more details, take a look at 
     *                `euclideanSquaredMetric8` in `kernels/rbc_kernels.cl`.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    template <KernelTypeC K>
    void RBCComputeDists<K>::init (unsigned int _nx, unsigned int _nr, unsigned int _d, float _a, Staging _staging)
    {
        nx = _nx; nr = _nr; d = _d; a = _a;
        bufferXSize = nx * d * sizeof (cl_float);
        bufferRSize = nr * d * sizeof (cl_float);
        bufferDSize = nx * nr * sizeof (cl_float);
        staging = _staging;

        cl::Device &device = env.devices[info.pIdx][info.dIdx];
        size_t maxLocalSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE> ();

        // Establish local dimensions
        size_t lXdim, lYdim;
        switch (K)
        {
            case KernelTypeC::KINECT_R:
            case KernelTypeC::KINECT_X_R:
                // Optimal configuration (found heuristically): (4, maxLocalSize / 4)
                // But (maxLocalSize / 4) forces a harder restriction on the number of
                // points in X. We'll try to relax this, if necessary
                for (lXdim = 4, lYdim = maxLocalSize / 4; 
                    ((nx % (4 * lYdim)) != 0) && (lXdim != lYdim); lXdim <<= 1, lYdim >>= 1) ;
                break;
            default:  // Square work-group
                for (lXdim = maxLocalSize; lXdim > (size_t) std::sqrt (maxLocalSize); lXdim >>= 1) ;
                lYdim = lXdim;
        }

        try
        {
            std::ostringstream ss;

            if (nx == 0 || nr == 0)
                throw std::string ("The number of points in X or R cannot be zero");

            unsigned int xMultiple, yMultiple;
            switch (K)
            {
                case KernelTypeC::SHARED_NONE:
                case KernelTypeC::KINECT:
                    xMultiple = 4;
                    yMultiple = 4;
                    break;
                default:
                    xMultiple = 4 * lXdim;
                    yMultiple = 4 * lYdim;
            }

            if (nx % yMultiple)
            {
                ss << "The number of points in X must be ";
                ss << "a multiple of " << yMultiple;
                throw ss.str ();
            }

            if (nr % xMultiple)
            {
                ss << "The number of points in R must be ";
                ss << "a multiple of " << xMultiple;
                throw ss.str ();
            }

            if (d % 4)
                throw std::string ("The dimensionality of the data must be a multiple of 4");

        }
        catch (const std::string &error)
        {
            std::cerr << "Error[RBCComputeDists]: " << error << std::endl;
            exit (EXIT_FAILURE);
        }

        size_t localSize = lXdim * lYdim;
        size_t wgMultiple = kernel.getWorkGroupInfo
                <CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE> (env.devices[info.pIdx][info.dIdx]);
        if (localSize % wgMultiple)
            std::cout << "Warning[RBCComputeDists]: The work-group size [" << localSize 
                      << "] is not a multiple of the preferred size [" 
                      << wgMultiple << "] on this device" << std::endl;

        // Set workspaces
        global = cl::NDRange (nr / 4, nx / 4);
        local = cl::NDRange (lXdim, lYdim);

        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrInX = nullptr;
                hPtrInR = nullptr;
                hPtrOutD = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferInX () == nullptr)
                    hBufferInX = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferXSize);
                if (hBufferInR () == nullptr)
                    hBufferInR = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferRSize);

                hPtrInX = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInX, CL_FALSE, CL_MAP_WRITE, 0, bufferXSize);
                hPtrInR = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInR, CL_FALSE, CL_MAP_WRITE, 0, bufferRSize);
                queue.enqueueUnmapMemObject (hBufferInX, hPtrInX);
                queue.enqueueUnmapMemObject (hBufferInR, hPtrInR);

                if (!io)
                {
                    queue.finish ();
                    hPtrOutD = nullptr;
                    break;
                }

            case Staging::O:
                if (hBufferOutD () == nullptr)
                    hBufferOutD = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferDSize);

                hPtrOutD = (cl_float *) queue.enqueueMapBuffer (
                    hBufferOutD, CL_FALSE, CL_MAP_READ, 0, bufferDSize);
                queue.enqueueUnmapMemObject (hBufferOutD, hPtrOutD);
                queue.finish ();

                if (!io) { hPtrInX = nullptr; hPtrInR = nullptr; }
                break;
        }

        // Create device buffers
        if (dBufferInX () == nullptr)
            dBufferInX = cl::Buffer (context, CL_MEM_READ_ONLY, bufferXSize);
        if (dBufferInR () == nullptr)
            dBufferInR = cl::Buffer (context, CL_MEM_READ_ONLY, bufferRSize);
        if (dBufferOutD () == nullptr)
            dBufferOutD = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferDSize);

        // Set kernel arguments
        kernel.setArg (0, dBufferInX);
        kernel.setArg (1, dBufferInR);
        kernel.setArg (2, dBufferOutD);

        switch (K)
        {
            case KernelTypeC::SHARED_NONE:
                a = std::numeric_limits<cl_float>::infinity ();
                kernel.setArg (3, d);
                break;
            case KernelTypeC::SHARED_R:
                a = std::numeric_limits<cl_float>::infinity ();
                kernel.setArg (3, cl::Local ((local[0] << 2) * (local[1] << 2) * sizeof (cl_float)));
                kernel.setArg (4, d);
                break;
            case KernelTypeC::SHARED_X_R:
                a = std::numeric_limits<cl_float>::infinity ();
                kernel.setArg (3, cl::Local ((local[0] << 2) * (local[1] << 2) * sizeof (cl_float)));
                kernel.setArg (4, cl::Local ((local[0] << 2) * (local[1] << 2) * sizeof (cl_float)));
                kernel.setArg (5, d);
                break;
            case KernelTypeC::KINECT:
                kernel.setArg (3, a);
                break;
            case KernelTypeC::KINECT_R:
                kernel.setArg (3, cl::Local (4 * local[0] * sizeof (cl_float8)));
                kernel.setArg (4, a);
                break;
            case KernelTypeC::KINECT_X_R:
                kernel.setArg (3, cl::Local (4 * local[1] * sizeof (cl_float8)));
                kernel.setArg (4, cl::Local (4 * local[0] * sizeof (cl_float8)));
                kernel.setArg (5, a);
                break;
        }
    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    template <KernelTypeC K>
    void RBCComputeDists<K>::write (RBCComputeDists::Memory mem, void *ptr, bool block, 
                                    const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case RBCComputeDists::Memory::D_IN_X:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + nx * d, hPtrInX);
                    queue.enqueueWriteBuffer (dBufferInX, block, 0, bufferXSize, hPtrInX, events, event);
                    break;
                case RBCComputeDists::Memory::D_IN_R:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + nr * d, hPtrInR);
                    queue.enqueueWriteBuffer (dBufferInR, block, 0, bufferRSize, hPtrInR, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    template <KernelTypeC K>
    void* RBCComputeDists<K>::read (RBCComputeDists::Memory mem, bool block, 
                                    const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case RBCComputeDists::Memory::H_OUT_D:
                    queue.enqueueReadBuffer (dBufferOutD, block, 0, bufferDSize, hPtrOutD, events, event);
                    return hPtrOutD;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the kernel execution.
     */
    template <KernelTypeC K>
    void RBCComputeDists<K>::run (const std::vector<cl::Event> *events, cl::Event *event)
    {
        queue.enqueueNDRangeKernel (kernel, cl::NullRange, global, local, events, event);
    }


    /*! \note If the template parameter, `K`, has the value `SHARED_NONE`, `SHARED_R`, or
     *        `SHARED_X_R`, the parameter \f$ \alpha \f$ is not applicable and the 
     *        value returned is \f$ \infty \f$.
     *  
     *  \return The scaling factor \f$ \alpha \f$.
     */
    template <KernelTypeC K>
    float RBCComputeDists<K>::getAlpha ()
    {
        return a;
    }


    /*! \details Updates the kernel argument for the scaling factor \f$ \alpha \f$.
     *  \note If the template parameter, `K`, has the value `SHARED_NONE`, `SHARED_R`, 
     *        or `SHARED_X_R`, the parameter \f$ \alpha \f$ is not applicable and the 
     *        function has no effect.
     *
     *  \param[in] _a scaling factor \f$ \alpha \f$.
     */
    template <KernelTypeC K>
    void RBCComputeDists<K>::setAlpha (float _a)
    {
        switch (K)
        {
            case KernelTypeC::SHARED_NONE:
            case KernelTypeC::SHARED_R:
            case KernelTypeC::SHARED_X_R:
                std::cerr << "The parameter alpha is not applicable in ";
                std::cerr << "this template instantiation" << std::endl;
                break;
            case KernelTypeC::KINECT:
                a = _a;
                kernel.setArg (3, a);
                break;
            case KernelTypeC::KINECT_R:
                a = _a;
                kernel.setArg (4, a);
                break;
            case KernelTypeC::KINECT_X_R:
                a = _a;
                kernel.setArg (5, a);
                break;
        }
    }


    /*! \brief Template instantiation for the case of the `rbcComputeDists_SharedNone` kernel. */
    template class RBCComputeDists<KernelTypeC::SHARED_NONE>;
    /*! \brief Template instantiation for the case of the `rbcComputeDists_SharedR` kernel. */
    template class RBCComputeDists<KernelTypeC::SHARED_R>;
    /*! \brief Template instantiation for the case of the `rbcComputeDists_SharedXR` kernel. */
    template class RBCComputeDists<KernelTypeC::SHARED_X_R>;
    /*! \brief Template instantiation for the case of the `rbcComputeDists_Kinect` kernel. */
    template class RBCComputeDists<KernelTypeC::KINECT>;
    /*! \brief Template instantiation for the case of the `rbcComputeDists_Kinect_R` kernel. */
    template class RBCComputeDists<KernelTypeC::KINECT_R>;
    /*! \brief Template instantiation for the case of the `rbcComputeDists_Kinect_XR` kernel. */
    template class RBCComputeDists<KernelTypeC::KINECT_X_R>;


    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. Specifies the context, queue, etc, to be used.
     */
    template <>
    Reduce<ReduceConfig::MIN, cl_float>::Reduce (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        recKernel (env.getProgram (info.pgIdx), "reduce_min_f"), 
        groupRecKernel (env.getProgram (info.pgIdx), "reduce_min_f")
    {
        wgMultiple = recKernel.getWorkGroupInfo
            <CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE> (env.devices[info.pIdx][info.dIdx]);
    }


    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. Specifies the context, queue, etc, to be used.
     */
    template <>
    Reduce<ReduceConfig::MAX, cl_uint>::Reduce (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        recKernel (env.getProgram (info.pgIdx), "reduce_max_ui"), 
        groupRecKernel (env.getProgram (info.pgIdx), "reduce_max_ui")
    {
        wgMultiple = recKernel.getWorkGroupInfo
            <CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE> (env.devices[info.pIdx][info.dIdx]);
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    template <ReduceConfig C, typename T>
    cl::Memory& Reduce<C, T>::get (Reduce::Memory mem)
    {
        switch (mem)
        {
            case Reduce::Memory::H_IN:
                return hBufferIn;
            case Reduce::Memory::H_OUT:
                return hBufferOut;
            case Reduce::Memory::D_IN:
                return dBufferIn;
            case Reduce::Memory::D_RED:
                return dBufferR;
            case Reduce::Memory::D_OUT:
                return dBufferOut;
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *        
     *  \param[in] _cols number of columns in the input array.
     *  \param[in] _rows number of rows in the input array.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    template <ReduceConfig C, typename T>
    void Reduce<C, T>::init (unsigned int _cols, unsigned int _rows, Staging _staging)
    {
        cols = _cols; rows = _rows;
        bufferInSize  = cols * rows * sizeof (T);
        bufferOutSize = rows * sizeof (T);
        staging = _staging;

        // Establish the number of work-groups per row
        wgXdim = std::ceil (cols / (float) (8 * wgMultiple));
        // Round up to a multiple of 4 (data are handled as float4)
        if ((wgXdim != 1) && (wgXdim % 4)) wgXdim += 4 - wgXdim % 4;

        bufferGRSize = wgXdim * rows * sizeof (T);

        try
        {
            if (wgXdim == 0)
                throw "The array cannot have zero columns";

            if (cols % 4 != 0)
                throw "The number of columns in the array must be a multiple of 4";

            // (8 * wgMultiple) elements per work-group
            // (8 * wgMultiple) work-groups maximum
            if (cols > std::pow (8 * wgMultiple, 2))
            {
                std::ostringstream ss;
                ss << "The current configuration of MinReduce supports arrays ";
                ss << "of up to " << std::pow (8 * wgMultiple, 2) << " columns";
                throw ss.str ().c_str ();
            }
        }
        catch (const char *error)
        {
            std::cerr << "Error[Reduce]: " << error << std::endl;
            exit (EXIT_FAILURE);
        }

        // Set workspaces
        globalR = cl::NDRange (wgXdim * wgMultiple, rows);
        globalGR = cl::NDRange (wgMultiple, rows);
        local = cl::NDRange (wgMultiple, 1);
        
        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrIn = nullptr;
                hPtrOut = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferIn () == nullptr)
                    hBufferIn = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferInSize);

                hPtrIn = (T *) queue.enqueueMapBuffer (
                    hBufferIn, CL_FALSE, CL_MAP_WRITE, 0, bufferInSize);
                queue.enqueueUnmapMemObject (hBufferIn, hPtrIn);

                if (!io)
                {
                    queue.finish ();
                    hPtrOut = nullptr;
                    break;
                }

            case Staging::O:
                if (hBufferOut () == nullptr)
                    hBufferOut = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferOutSize);

                hPtrOut = (T *) queue.enqueueMapBuffer (
                    hBufferOut, CL_FALSE, CL_MAP_READ, 0, bufferOutSize);
                queue.enqueueUnmapMemObject (hBufferOut, hPtrOut);
                queue.finish ();

                if (!io) hPtrIn = nullptr;
                break;
        }

        // Create device buffers
        if (dBufferIn () == nullptr)
            dBufferIn = cl::Buffer (context, CL_MEM_READ_ONLY, bufferInSize);
        if (dBufferR () == nullptr && wgXdim != 1)
            dBufferR = cl::Buffer (context, CL_MEM_READ_WRITE, bufferGRSize);
        if (dBufferOut () == nullptr)
            dBufferOut = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferOutSize);

        // Set kernel arguments
        if (wgXdim == 1)
        {
            recKernel.setArg (0, dBufferIn);
            recKernel.setArg (1, dBufferOut);
            recKernel.setArg (2, cl::Local (2 * local[0] * sizeof (T)));
            recKernel.setArg (3, cols / 4);
        }
        else
        {
            recKernel.setArg (0, dBufferIn);
            recKernel.setArg (1, dBufferR);
            recKernel.setArg (2, cl::Local (2 * local[0] * sizeof (T)));
            recKernel.setArg (3, cols / 4);

            groupRecKernel.setArg (0, dBufferR);
            groupRecKernel.setArg (1, dBufferOut);
            groupRecKernel.setArg (2, cl::Local (2 * local[0] * sizeof (T)));
            groupRecKernel.setArg (3, (cl_uint) (wgXdim / 4));
        }
    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    template <ReduceConfig C, typename T>
    void Reduce<C, T>::write (Reduce::Memory mem, void *ptr, bool block, 
                              const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case Reduce::Memory::D_IN:
                    if (ptr != nullptr)
                        std::copy ((T *) ptr, (T *) ptr + cols * rows, hPtrIn);
                    queue.enqueueWriteBuffer (dBufferIn, block, 0, bufferInSize, hPtrIn, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    template <ReduceConfig C, typename T>
    void* Reduce<C, T>::read (Reduce::Memory mem, bool block, 
                              const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case Reduce::Memory::H_OUT:
                    queue.enqueueReadBuffer (dBufferOut, block, 0, bufferOutSize, hPtrOut, events, event);
                    return hPtrOut;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the kernel execution.
     */
    template <ReduceConfig C, typename T>
    void Reduce<C, T>::run (const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (wgXdim == 1)
        {
            queue.enqueueNDRangeKernel (recKernel, cl::NullRange, globalR, local, events, event);
        }
        else
        {
            queue.enqueueNDRangeKernel (recKernel, cl::NullRange, globalR, local, events);
            queue.enqueueNDRangeKernel (groupRecKernel, cl::NullRange, globalGR, local, nullptr, event);
        }
    }


    /*! \brief Template instantiation for the case of `MIN` reduction and `float` data. */
    template class Reduce<ReduceConfig::MIN, cl_float>;
    /*! \brief Template instantiation for the case of `MAX` reduction and `uint` data. */
    template class Reduce<ReduceConfig::MAX, cl_uint>;


    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. Specifies the context, queue, etc, to be used.
     */
    RBCMin::RBCMin (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        initKernel (env.getProgram (info.pgIdx), "rbcNInit"), 
        minsKernel (env.getProgram (info.pgIdx), "rbcMinDists"), 
        groupMinsKernel (env.getProgram (info.pgIdx), "rbcGroupMinDists")
    {
        wgMultiple = minsKernel.getWorkGroupInfo
            <CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE> (env.devices[info.pIdx][info.dIdx]);
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    cl::Memory& RBCMin::get (RBCMin::Memory mem)
    {
        switch (mem)
        {
            case RBCMin::Memory::H_IN_D:
                return hBufferInD;
            case RBCMin::Memory::H_OUT_ID:
                return hBufferOutID;
            case RBCMin::Memory::H_OUT_RNK:
                return hBufferOutRnk;
            case RBCMin::Memory::H_OUT_N:
                return hBufferOutN;
            case RBCMin::Memory::D_IN_D:
                return dBufferInD;
            case RBCMin::Memory::D_MINS:
                return dBufferGM;
            case RBCMin::Memory::D_OUT_ID:
                return dBufferOutID;
            case RBCMin::Memory::D_OUT_RNK:
                return dBufferOutRnk;
            case RBCMin::Memory::D_OUT_N:
                return dBufferOutN;
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *        
     *  \param[in] _cols number of columns in the input array.
     *  \param[in] _rows number of rows in the input array.
     *  \param[in] _accCounters flag to indicate whether or not to involve in the computation 
     *                          the list element counters, `N`, and element ranks, `Rnk`.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    void RBCMin::init (unsigned int _cols, unsigned int _rows, int _accCounters, Staging _staging)
    {
        cols = _cols; rows = _rows;
        bufferDSize  = cols * rows * sizeof (cl_float);
        bufferIDSize = rows * sizeof (rbc_dist_id);
        bufferRnkSize = rows * sizeof (cl_uint);
        bufferNSize = cols * sizeof (cl_uint);
        accCounters = _accCounters;
        staging = _staging;

        // Establish the number of work-groups per row
        wgXdim = std::ceil (cols / (float) (8 * wgMultiple));

        bufferGMSize = wgXdim * (rows * sizeof (rbc_dist_id));

        try
        {
            if (cols == 0)
                throw "The array cannot have zero columns";

            if (cols % 4 != 0)
                throw "The number of columns in the array must be a multiple of 4";

            // (8 * wgMultiple) elements per work-group
            // (2 * wgMultiple) work-groups maximum
            if (cols > 16 * wgMultiple * wgMultiple)
            {
                std::ostringstream ss;
                ss << "The current configuration of RBCMin supports arrays ";
                ss << "of up to " << 16 * wgMultiple * wgMultiple << " columns";
                throw ss.str ().c_str ();
            }
        }
        catch (const char *error)
        {
            std::cerr << "Error[RBCMin]: " << error << std::endl;
            exit (EXIT_FAILURE);
        }

        // Set workspaces
        globalInit = cl::NDRange (cols / 4);
        globalM = cl::NDRange (wgXdim * wgMultiple, rows);
        globalGM = cl::NDRange (wgMultiple, rows);
        local = cl::NDRange (wgMultiple, 1);
        
        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrInD = nullptr;
                hPtrOutID = nullptr;
                hPtrOutRnk = nullptr;
                hPtrOutN = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferInD () == nullptr)
                    hBufferInD = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferDSize);

                hPtrInD = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInD, CL_FALSE, CL_MAP_WRITE, 0, bufferDSize);
                queue.enqueueUnmapMemObject (hBufferInD, hPtrInD);

                if (!io)
                {
                    queue.finish ();
                    hPtrOutID = nullptr;
                    hPtrOutRnk = nullptr;
                    hPtrOutN = nullptr;
                    break;
                }

            case Staging::O:
                if (hBufferOutID () == nullptr)
                    hBufferOutID = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferIDSize);
                
                hPtrOutID = (rbc_dist_id *) queue.enqueueMapBuffer (
                    hBufferOutID, CL_FALSE, CL_MAP_READ, 0, bufferIDSize);
                queue.enqueueUnmapMemObject (hBufferOutID, hPtrOutID);
                

                if (accCounters == 1)
                {
                    if (hBufferOutRnk () == nullptr)
                        hBufferOutRnk = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferRnkSize);
                    if (hBufferOutN () == nullptr)
                        hBufferOutN = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferNSize);

                    hPtrOutRnk = (cl_uint *) queue.enqueueMapBuffer (
                        hBufferOutRnk, CL_FALSE, CL_MAP_READ, 0, bufferRnkSize);
                    hPtrOutN = (cl_uint *) queue.enqueueMapBuffer (
                        hBufferOutN, CL_FALSE, CL_MAP_READ, 0, bufferNSize);
                    queue.enqueueUnmapMemObject (hBufferOutRnk, hPtrOutRnk);
                    queue.enqueueUnmapMemObject (hBufferOutN, hPtrOutN);
                }
                else
                {
                    hPtrOutRnk = nullptr;
                    hPtrOutN = nullptr;
                }

                queue.finish ();

                if (!io) hPtrInD = nullptr;
                break;
        }

        // Create device buffers
        if (dBufferInD () == nullptr)
            dBufferInD = cl::Buffer (context, CL_MEM_READ_ONLY, bufferDSize);
        if (dBufferGM () == nullptr && wgXdim != 1)
            dBufferGM = cl::Buffer (context, CL_MEM_READ_WRITE, bufferGMSize);
        if (dBufferOutID () == nullptr)
            dBufferOutID = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferIDSize);
        if (dBufferOutRnk () == nullptr && accCounters == 1)
            dBufferOutRnk = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferRnkSize);
        if (dBufferOutN () == nullptr && accCounters == 1)
            dBufferOutN = cl::Buffer (context, CL_MEM_READ_WRITE, bufferNSize);

        // Set kernel arguments
        if (accCounters == 1)
        {
            initKernel.setArg (0, dBufferOutN);
            initKernel.setArg (1, 0);
        }

        minsKernel.setArg (0, dBufferInD);
        minsKernel.setArg (1, dBufferOutID);
        minsKernel.setArg (2, dBufferOutN);    // Unused when wgXdim > 1        
        minsKernel.setArg (3, dBufferOutRnk);  // Unused when wgXdim > 1
        minsKernel.setArg (4, cl::Local (2 * local[0] * sizeof (rbc_dist_id)));
        minsKernel.setArg (5, cols / 4);
        minsKernel.setArg (6, accCounters);

        if (wgXdim != 1)
        {
            minsKernel.setArg (1, dBufferGM);
            minsKernel.setArg (6, 0);

            groupMinsKernel.setArg (0, dBufferGM);
            groupMinsKernel.setArg (1, dBufferOutID);
            groupMinsKernel.setArg (2, dBufferOutN);
            groupMinsKernel.setArg (3, dBufferOutRnk);
            groupMinsKernel.setArg (4, cl::Local (2 * local[0] * sizeof (rbc_dist_id)));
            groupMinsKernel.setArg (5, (cl_uint) wgXdim);
            groupMinsKernel.setArg (6, accCounters);
        }
    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    void RBCMin::write (RBCMin::Memory mem, void *ptr, bool block, 
                        const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case RBCMin::Memory::D_IN_D:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + cols * rows, hPtrInD);
                    queue.enqueueWriteBuffer (dBufferInD, block, 0, bufferDSize, hPtrInD, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    void* RBCMin::read (RBCMin::Memory mem, bool block, 
                        const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case RBCMin::Memory::H_OUT_ID:
                    queue.enqueueReadBuffer (dBufferOutID, block, 0, bufferIDSize, hPtrOutID, events, event);
                    return hPtrOutID;
                case RBCMin::Memory::H_OUT_RNK:
                    if (hPtrOutRnk != nullptr && accCounters == 1)
                        queue.enqueueReadBuffer (dBufferOutRnk, block, 0, bufferRnkSize, hPtrOutRnk, events, event);
                    return hPtrOutRnk;
                case RBCMin::Memory::H_OUT_N:
                    if (hPtrOutN != nullptr && accCounters == 1)
                        queue.enqueueReadBuffer (dBufferOutN, block, 0, bufferNSize, hPtrOutN, events, event);
                    return hPtrOutN;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the kernel execution.
     */
    void RBCMin::run (const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (accCounters == 1)
            queue.enqueueNDRangeKernel (initKernel, cl::NullRange, globalInit, cl::NullRange, events);

        if (wgXdim == 1)
        {
            queue.enqueueNDRangeKernel (minsKernel, cl::NullRange, globalM, local, nullptr, event);
        }
        else
        {
            queue.enqueueNDRangeKernel (minsKernel, cl::NullRange, globalM, local);
            queue.enqueueNDRangeKernel (groupMinsKernel, cl::NullRange, globalGM, local, nullptr, event);
        }
    }


    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. Specifies the context, queue, etc, to be used.
     */
    template <>
    Scan<ScanConfig::INCLUSIVE, cl_int>::Scan (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        kernelScan (env.getProgram (info.pgIdx), "inclusiveScan_i"), 
        kernelSumsScan (env.getProgram (info.pgIdx), "inclusiveScan_i"), 
        kernelAddSums (env.getProgram (info.pgIdx), "addGroupSums_i")
    {
        wgMultiple = kernelScan.getWorkGroupInfo
            <CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE> (env.devices[info.pIdx][info.dIdx]);
    }


    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. Specifies the context, queue, etc, to be used.
     */
    template <>
    Scan<ScanConfig::EXCLUSIVE, cl_int>::Scan (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        kernelScan (env.getProgram (info.pgIdx), "exclusiveScan_i"), 
        kernelSumsScan (env.getProgram (info.pgIdx), "inclusiveScan_i"), 
        kernelAddSums (env.getProgram (info.pgIdx), "addGroupSums_i")
    {
        wgMultiple = kernelScan.getWorkGroupInfo
            <CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE> (env.devices[info.pIdx][info.dIdx]);
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    template <ScanConfig C, typename T>
    cl::Memory& Scan<C, T>::get (Scan::Memory mem)
    {
        switch (mem)
        {
            case Scan::Memory::H_IN:
                return hBufferIn;
            case Scan::Memory::H_OUT:
                return hBufferOut;
            case Scan::Memory::D_IN:
                return dBufferIn;
            case Scan::Memory::D_SUMS:
                return dBufferSums;
            case Scan::Memory::D_OUT:
                return dBufferOut;
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *  \note Working with `float` elements and having large summations can be problematic.
     *        It is advised that a scaling is applied on the elements for better accuracy.
     *        
     *  \param[in] _cols number of columns in the input array.
     *  \param[in] _rows number of rows in the input array.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    template <ScanConfig C, typename T>
    void Scan<C, T>::init (unsigned int _cols, unsigned int _rows, Staging _staging)
    {
        cols = _cols; rows = _rows;
        bufferSize = cols * rows * sizeof (T);
        staging = _staging;

        // Establish the number of work-groups per row
        wgXdim = std::ceil (cols / (float) (8 * wgMultiple));
        // Round up to a multiple of 4 (data are handled as int4)
        if ((wgXdim != 1) && (wgXdim % 4)) wgXdim += 4 - wgXdim % 4;

        bufferSumsSize = wgXdim * rows * sizeof (T);

        try
        {
            if (wgXdim == 0)
                throw "The array cannot have zero columns";

            if (cols % 4 != 0)
                throw "The number of columns in the array must be a multiple of 4";

            // (8 * wgMultiple) elements per work-group
            // (8 * wgMultiple) work-groups maximum
            if (cols > std::pow (8 * wgMultiple, 2))
            {
                std::ostringstream ss;
                ss << "The current configuration of Scan supports arrays ";
                ss << "of up to " << std::pow (8 * wgMultiple, 2) << " columns";
                throw ss.str ().c_str ();
            }
        }
        catch (const char *error)
        {
            std::cerr << "Error[Scan]: " << error << std::endl;
            exit (EXIT_FAILURE);
        }

        // Set workspaces
        globalScan = cl::NDRange (wgXdim * wgMultiple, rows);
        localScan = cl::NDRange (wgMultiple, 1);
        globalSumsScan = cl::NDRange (wgMultiple, rows);
        globalAddSums = cl::NDRange (2 * (wgXdim - 1) * wgMultiple, rows);
        localAddSums = cl::NDRange (2 * wgMultiple, 1);
        offsetAddSums = cl::NDRange (2 * wgMultiple, 0);

        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrIn = nullptr;
                hPtrOut = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferIn () == nullptr)
                    hBufferIn = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferSize);

                hPtrIn = (T *) queue.enqueueMapBuffer (
                    hBufferIn, CL_FALSE, CL_MAP_WRITE, 0, bufferSize);
                queue.enqueueUnmapMemObject (hBufferIn, hPtrIn);

                if (!io)
                {
                    queue.finish ();
                    hPtrOut = nullptr;
                    break;
                }

            case Staging::O:
                if (hBufferOut () == nullptr)
                    hBufferOut = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferSize);

                hPtrOut = (T *) queue.enqueueMapBuffer (
                    hBufferOut, CL_FALSE, CL_MAP_READ, 0, bufferSize);
                queue.enqueueUnmapMemObject (hBufferOut, hPtrOut);
                queue.finish ();

                if (!io) hPtrIn = nullptr;
                break;
        }
        
        // Create device buffers
        if (dBufferIn () == nullptr)
            dBufferIn = cl::Buffer (context, CL_MEM_READ_ONLY, bufferSize);
        if (dBufferSums () == nullptr)
            dBufferSums = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSumsSize);
        if (dBufferOut () == nullptr)
            dBufferOut = cl::Buffer (context, CL_MEM_READ_WRITE, bufferSize);

        // Set kernel arguments
        if (wgXdim == 1)
        {
            kernelScan.setArg (0, dBufferIn);
            kernelScan.setArg (1, dBufferOut);
            kernelScan.setArg (2, cl::Local (2 * localScan[0] * sizeof (T)));
            kernelScan.setArg (3, dBufferSums);  // Unused
            kernelScan.setArg (4, cols / 4);
        }
        else
        {
            kernelScan.setArg (0, dBufferIn);
            kernelScan.setArg (1, dBufferOut);
            kernelScan.setArg (2, cl::Local (2 * localScan[0] * sizeof (T)));
            kernelScan.setArg (3, dBufferSums);
            kernelScan.setArg (4, cols / 4);

            kernelSumsScan.setArg (0, dBufferSums);
            kernelSumsScan.setArg (1, dBufferSums);
            kernelSumsScan.setArg (2, cl::Local (2 * localScan[0] * sizeof (T)));
            kernelSumsScan.setArg (3, dBufferSums);  // Unused
            kernelSumsScan.setArg (4, (cl_uint) (wgXdim / 4));

            kernelAddSums.setArg (0, dBufferSums);
            kernelAddSums.setArg (1, dBufferOut);
            kernelAddSums.setArg (2, cols / 4);
        }
    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    template <ScanConfig C, typename T>
    void Scan<C, T>::write (Scan::Memory mem, void *ptr, bool block, 
                            const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case Scan::Memory::D_IN:
                    if (ptr != nullptr)
                        std::copy ((T *) ptr, (T *) ptr + cols * rows, hPtrIn);
                    queue.enqueueWriteBuffer (dBufferIn, block, 0, bufferSize, hPtrIn, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    template <ScanConfig C, typename T>
    void* Scan<C, T>::read (Scan::Memory mem, bool block, 
                            const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case Scan::Memory::H_OUT:
                    queue.enqueueReadBuffer (dBufferOut, block, 0, bufferSize, hPtrOut, events, event);
                    return hPtrOut;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the last kernel execution.
     */
    template <ScanConfig C, typename T>
    void Scan<C, T>::run (const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (wgXdim == 1)
        {
            queue.enqueueNDRangeKernel (
                kernelScan, cl::NullRange, globalScan, localScan, events, event);
        }
        else
        {
            queue.enqueueNDRangeKernel (
                kernelScan, cl::NullRange, globalScan, localScan, events);

            queue.enqueueNDRangeKernel (
                kernelSumsScan, cl::NullRange, globalSumsScan, localScan);

            queue.enqueueNDRangeKernel (
                kernelAddSums, offsetAddSums, globalAddSums, localAddSums, nullptr, event);
        }
    }


    /*! \brief Template instantiation for `inclusive` scan and `int` data. */
    template class Scan<ScanConfig::INCLUSIVE, cl_int>;
    /*! \brief Template instantiation for `exclusive` scan and `int` data. */
    template class Scan<ScanConfig::EXCLUSIVE, cl_int>;


    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. Specifies the context, queue, etc, to be used.
     */
    template <RBCPermuteConfig C>
    RBCPermute<C>::RBCPermute (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0]))
    {
        switch (C)
        {
            case RBCPermuteConfig::GENERIC:
                kernel = cl::Kernel (env.getProgram (info.pgIdx), "rbcPermute");
                break;
            case RBCPermuteConfig::KINECT:
                kernel = cl::Kernel (env.getProgram (info.pgIdx), "rbcPermute_Kinect");
        }
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    template <RBCPermuteConfig C>
    cl::Memory& RBCPermute<C>::get (RBCPermute::Memory mem)
    {
        switch (mem)
        {
            case RBCPermute::Memory::H_IN_X:
                return hBufferInX;
            case RBCPermute::Memory::H_IN_ID:
                return hBufferInID;
            case RBCPermute::Memory::H_IN_RNK:
                return hBufferInRnk;
            case RBCPermute::Memory::H_IN_O:
                return hBufferInO;
            case RBCPermute::Memory::H_OUT_X_P:
                return hBufferOutXp;
            case RBCPermute::Memory::H_OUT_ID_P:
                return hBufferOutIDp;
            case RBCPermute::Memory::D_IN_X:
                return dBufferInX;
            case RBCPermute::Memory::D_IN_ID:
                return dBufferInID;
            case RBCPermute::Memory::D_IN_RNK:
                return dBufferInRnk;
            case RBCPermute::Memory::D_IN_O:
                return dBufferInO;
            case RBCPermute::Memory::D_OUT_X_P:
                return dBufferOutXp;
            case RBCPermute::Memory::D_OUT_ID_P:
                return dBufferOutIDp;
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *        
     *  \param[in] _nx number of database points.
     *  \param[in] _nr number of representative points.
     *  \param[in] _d dimensionality of the associated points.
     *  \param[in] _permID flag to indicate whether or not to also permute the ID array.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    template <RBCPermuteConfig C>
    void RBCPermute<C>::init (unsigned int _nx, unsigned int _nr, unsigned int _d, int _permID, Staging _staging)
    {
        nx = _nx; nr = _nr; d = _d; permID = _permID;
        bufferXSize = nx * d * sizeof (cl_float);
        bufferIDSize = nx * sizeof (rbc_dist_id);
        bufferOSize = nr * sizeof (cl_uint);
        bufferRnkSize = nx * sizeof (cl_uint);
        staging = _staging;

        try
        {
            if (nx == 0)
                throw "The database cannot have zero points";

            if (d % 4)
                throw "The dimensionality of the data must be a multiple of 4";
        }
        catch (const char *error)
        {
            std::cerr << "Error[RBCPermute]: " << error << std::endl;
            exit (EXIT_FAILURE);
        }

        // Set workspaces
        global = (C == RBCPermuteConfig::GENERIC) ? cl::NDRange (d >> 2, nx) : cl::NDRange (nx);

        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrInX = nullptr;
                hPtrInID = nullptr;
                hPtrInRnk = nullptr;
                hPtrInO = nullptr;
                hPtrOutXp = nullptr;
                hPtrOutIDp = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferInX () == nullptr)
                    hBufferInX = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferXSize);
                if (hBufferInID () == nullptr)
                    hBufferInID = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferIDSize);
                if (hBufferInRnk () == nullptr)
                    hBufferInRnk = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferRnkSize);
                if (hBufferInO () == nullptr)
                    hBufferInO = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferOSize);

                hPtrInX = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInX, CL_FALSE, CL_MAP_WRITE, 0, bufferXSize);
                hPtrInID = (rbc_dist_id *) queue.enqueueMapBuffer (
                    hBufferInID, CL_FALSE, CL_MAP_WRITE, 0, bufferIDSize);
                hPtrInRnk = (cl_uint *) queue.enqueueMapBuffer (
                    hBufferInRnk, CL_FALSE, CL_MAP_WRITE, 0, bufferRnkSize);
                hPtrInO = (cl_uint *) queue.enqueueMapBuffer (
                    hBufferInO, CL_FALSE, CL_MAP_WRITE, 0, bufferOSize);
                queue.enqueueUnmapMemObject (hBufferInX, hPtrInX);
                queue.enqueueUnmapMemObject (hBufferInID, hPtrInID);
                queue.enqueueUnmapMemObject (hBufferInRnk, hPtrInRnk);
                queue.enqueueUnmapMemObject (hBufferInO, hPtrInO);

                if (!io)
                {
                    queue.finish ();
                    hPtrOutXp = nullptr;
                    hPtrOutIDp = nullptr;
                    break;
                }

            case Staging::O:
                if (hBufferOutXp () == nullptr)
                    hBufferOutXp = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferXSize);

                hPtrOutXp = (cl_float *) queue.enqueueMapBuffer (
                    hBufferOutXp, CL_FALSE, CL_MAP_READ, 0, bufferXSize);
                queue.enqueueUnmapMemObject (hBufferOutXp, hPtrOutXp);

                if (permID == 1)
                {
                    if (hBufferOutIDp () == nullptr)
                        hBufferOutIDp = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferIDSize);
                    hPtrOutIDp = (rbc_dist_id *) queue.enqueueMapBuffer (
                        hBufferOutIDp, CL_FALSE, CL_MAP_READ, 0, bufferIDSize);
                    queue.enqueueUnmapMemObject (hBufferOutIDp, hPtrOutIDp);
                }
                else
                {
                    hPtrOutIDp = nullptr;
                }

                queue.finish ();

                if (!io)
                {
                    hPtrInX = nullptr;
                    hPtrInID = nullptr;
                    hPtrInRnk = nullptr;
                    hPtrInO = nullptr;
                }
                break;
        }
        
        // Create device buffers
        if (dBufferInX () == nullptr)
            dBufferInX = cl::Buffer (context, CL_MEM_READ_ONLY, bufferXSize);
        if (dBufferInID () == nullptr)
            dBufferInID = cl::Buffer (context, CL_MEM_READ_ONLY, bufferIDSize);
        if (dBufferInRnk () == nullptr)
            dBufferInRnk = cl::Buffer (context, CL_MEM_READ_ONLY, bufferRnkSize);
        if (dBufferInO () == nullptr)
            dBufferInO = cl::Buffer (context, CL_MEM_READ_ONLY, bufferOSize);
        if (dBufferOutXp () == nullptr)
            dBufferOutXp = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferXSize);
        if (dBufferOutIDp () == nullptr && permID == 1)
            dBufferOutIDp = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferIDSize);

        // Set kernel arguments
        kernel.setArg (0, dBufferInX);
        kernel.setArg (1, dBufferInID);
        kernel.setArg (2, dBufferOutXp);
        kernel.setArg (3, dBufferOutIDp);
        kernel.setArg (4, dBufferInO);
        kernel.setArg (5, dBufferInRnk);
        kernel.setArg (6, permID);

    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    template <RBCPermuteConfig C>
    void RBCPermute<C>::write (RBCPermute::Memory mem, void *ptr, bool block, 
                               const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case RBCPermute::Memory::D_IN_X:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + nx * d, hPtrInX);
                    queue.enqueueWriteBuffer (dBufferInX, block, 0, bufferXSize, hPtrInX, events, event);
                    break;
                case RBCPermute::Memory::D_IN_ID:
                    if (ptr != nullptr)
                        std::copy ((rbc_dist_id *) ptr, (rbc_dist_id *) ptr + nx, hPtrInID);
                    queue.enqueueWriteBuffer (dBufferInID, block, 0, bufferIDSize, hPtrInID, events, event);
                    break;
                case RBCPermute::Memory::D_IN_RNK:
                    if (ptr != nullptr)
                        std::copy ((uint *) ptr, (uint *) ptr + nx, hPtrInRnk);
                    queue.enqueueWriteBuffer (dBufferInRnk, block, 0, bufferRnkSize, hPtrInRnk, events, event);
                    break;
                case RBCPermute::Memory::D_IN_O:
                    if (ptr != nullptr)
                        std::copy ((uint *) ptr, (uint *) ptr + nr, hPtrInO);
                    queue.enqueueWriteBuffer (dBufferInO, block, 0, bufferOSize, hPtrInO, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    template <RBCPermuteConfig C>
    void* RBCPermute<C>::read (RBCPermute::Memory mem, bool block, 
                               const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case RBCPermute::Memory::H_OUT_X_P:
                    queue.enqueueReadBuffer (dBufferOutXp, block, 0, bufferXSize, hPtrOutXp, events, event);
                    return hPtrOutXp;
                case RBCPermute::Memory::H_OUT_ID_P:
                    if (permID == 1)
                        queue.enqueueReadBuffer (dBufferOutIDp, block, 0, bufferIDSize, hPtrOutIDp, events, event);
                    return hPtrOutIDp;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the last kernel execution.
     */
    template <RBCPermuteConfig C>
    void RBCPermute<C>::run (const std::vector<cl::Event> *events, cl::Event *event)
    {
        queue.enqueueNDRangeKernel (kernel, cl::NullRange, global, cl::NullRange, events, event);
    }


    /*! \brief Template instantiation for the case of `GENERIC` data. */
    template class RBCPermute<RBCPermuteConfig::GENERIC>;
    /*! \brief Template instantiation for the case of `KINECT` data. */
    template class RBCPermute<RBCPermuteConfig::KINECT>;


    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. Specifies the context, queue, etc, to be used.
     */
    template <KernelTypeC K, RBCPermuteConfig P>
    RBCConstruct<K, P>::RBCConstruct (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        rbcCompDists (env, info), rbcMinDists (env, info), 
        rbcScanNLists (env, info), rbcPermDB (env, info)
    {
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    template <KernelTypeC K, RBCPermuteConfig P>
    cl::Memory& RBCConstruct<K, P>::get (RBCConstruct::Memory mem)
    {
        switch (mem)
        {
            case RBCConstruct::Memory::H_IN_X:
                return hBufferInX;
            case RBCConstruct::Memory::H_IN_R:
                return hBufferInR;
            case RBCConstruct::Memory::H_OUT_ID:
                return hBufferOutID;
            case RBCConstruct::Memory::H_OUT_RNK:
                return hBufferOutRnk;
            case RBCConstruct::Memory::H_OUT_N:
                return hBufferOutN;
            case RBCConstruct::Memory::H_OUT_O:
                return hBufferOutO;
            case RBCConstruct::Memory::H_OUT_X_P:
                return hBufferOutXp;
            case RBCConstruct::Memory::H_OUT_ID_P:
                return hBufferOutIDp;
            case RBCConstruct::Memory::D_IN_X:
                return dBufferInX;
            case RBCConstruct::Memory::D_IN_R:
                return dBufferInR;
            case RBCConstruct::Memory::D_OUT_D:
                return rbcCompDists.get (RBCComputeDists<K>::Memory::D_OUT_D);
            case RBCConstruct::Memory::D_OUT_ID:
                return dBufferOutID;
            case RBCConstruct::Memory::D_OUT_RNK:
                return dBufferOutRnk;
            case RBCConstruct::Memory::D_OUT_N:
                return dBufferOutN;
            case RBCConstruct::Memory::D_OUT_O:
                return dBufferOutO;
            case RBCConstruct::Memory::D_OUT_X_P:
                return dBufferOutXp;
            case RBCConstruct::Memory::D_OUT_ID_P:
                return dBufferOutIDp;
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *        
     *  \param[in] _nx number of database points.
     *  \param[in] _nr number of representative points.
     *  \param[in] _d dimensionality of the associated points.
     *  \param[in] _a factor scaling the results of the distance calculations for the 
     *                geometric \f$ x_g \f$ and photometric \f$ x_p \f$ dimensions of 
     *                the \f$ x\epsilon\mathbb{R}^8 \f$ points. That is, \f$ \|x-x'\|_2^2= 
     *                f_g(a)\|x_g-x'_g\|_2^2+f_p(a)\|x_p-x'_p\|_2^2 \f$. This parameter is 
     *                applicable when involving the "Kinect" kernels. That is, when the 
     *                template parameter, `K`, gets the value `KINECT`, `KINECT_R`, or 
     *                `KINECT_X_R`. For more details, take a look at 
     *                `euclideanSquaredMetric8` in `kernels/rbc_kernels.cl`.
     *  \param[in] _permID flag to indicate whether or not to also permute the ID array.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    template <KernelTypeC K, RBCPermuteConfig P>
    void RBCConstruct<K, P>::init (unsigned int _nx, unsigned int _nr, unsigned int _d, 
                                   float _a, int _permID, Staging _staging)
    {
        nx = _nx; nr = _nr; d = _d; permID = _permID;
        bufferXSize = nx * d * sizeof (cl_float);
        bufferRSize = nr * d * sizeof (cl_float);
        bufferDSize = nr * nx * sizeof (cl_float);
        bufferIDSize = nx * sizeof (rbc_dist_id);
        bufferRnkSize = nx * sizeof (cl_uint);
        bufferNSize = nr * sizeof (cl_uint);
        bufferOSize = nr * sizeof (cl_uint);
        staging = _staging;

        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrInX = nullptr;
                hPtrInR = nullptr;
                hPtrOutID = nullptr;
                hPtrOutRnk = nullptr;
                hPtrOutN = nullptr;
                hPtrOutO = nullptr;
                hPtrOutXp = nullptr;
                hPtrOutIDp = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferInX () == nullptr)
                    hBufferInX = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferXSize);
                if (hBufferInR () == nullptr)
                    hBufferInR = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferRSize);

                hPtrInX = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInX, CL_FALSE, CL_MAP_WRITE, 0, bufferXSize);
                hPtrInR = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInR, CL_FALSE, CL_MAP_WRITE, 0, bufferRSize);
                queue.enqueueUnmapMemObject (hBufferInX, hPtrInX);
                queue.enqueueUnmapMemObject (hBufferInR, hPtrInR);

                if (!io)
                {
                    queue.finish ();
                    hPtrOutN = nullptr;
                    hPtrOutO = nullptr;
                    hPtrOutXp = nullptr;
                    hPtrOutIDp = nullptr;
                    break;
                }

            case Staging::O:
                if (hBufferOutID () == nullptr)
                    hBufferOutID = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferIDSize);
                if (hBufferOutRnk () == nullptr)
                    hBufferOutRnk = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferRnkSize);
                if (hBufferOutN () == nullptr)
                    hBufferOutN = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferNSize);
                if (hBufferOutO () == nullptr)
                    hBufferOutO = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferOSize);
                if (hBufferOutXp () == nullptr)
                    hBufferOutXp = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferXSize);

                hPtrOutID = (rbc_dist_id *) queue.enqueueMapBuffer (
                    hBufferOutID, CL_FALSE, CL_MAP_READ, 0, bufferIDSize);
                hPtrOutRnk = (cl_uint *) queue.enqueueMapBuffer (
                    hBufferOutRnk, CL_FALSE, CL_MAP_READ, 0, bufferRnkSize);
                hPtrOutN = (cl_uint *) queue.enqueueMapBuffer (
                    hBufferOutN, CL_FALSE, CL_MAP_READ, 0, bufferNSize);
                hPtrOutO = (cl_uint *) queue.enqueueMapBuffer (
                    hBufferOutO, CL_FALSE, CL_MAP_READ, 0, bufferOSize);
                hPtrOutXp = (cl_float *) queue.enqueueMapBuffer (
                    hBufferOutXp, CL_FALSE, CL_MAP_READ, 0, bufferXSize);
                queue.enqueueUnmapMemObject (hBufferOutID, hPtrOutID);
                queue.enqueueUnmapMemObject (hBufferOutRnk, hPtrOutRnk);
                queue.enqueueUnmapMemObject (hBufferOutN, hPtrOutN);
                queue.enqueueUnmapMemObject (hBufferOutO, hPtrOutO);
                queue.enqueueUnmapMemObject (hBufferOutXp, hPtrOutXp);

                if (permID == 1)
                {
                    if (hBufferOutIDp () == nullptr)
                        hBufferOutIDp = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferIDSize);
                    hPtrOutIDp = (rbc_dist_id *) queue.enqueueMapBuffer (
                        hBufferOutIDp, CL_FALSE, CL_MAP_READ, 0, bufferIDSize);
                    queue.enqueueUnmapMemObject (hBufferOutIDp, hPtrOutIDp);
                }
                else
                {
                    hPtrOutIDp = nullptr;
                }

                queue.finish ();

                if (!io)
                {
                    hPtrInX = nullptr;
                    hPtrInR = nullptr;
                }
                break;
        }
        
        // Create device buffers
        if (dBufferInX () == nullptr)
            dBufferInX = cl::Buffer (context, CL_MEM_READ_ONLY, bufferXSize);
        if (dBufferInR () == nullptr)
            dBufferInR = cl::Buffer (context, CL_MEM_READ_ONLY, bufferRSize);
        if (dBufferOutID () == nullptr)
            dBufferOutID = cl::Buffer (context, CL_MEM_READ_WRITE, bufferIDSize);
        if (dBufferOutRnk () == nullptr)
            dBufferOutRnk = cl::Buffer (context, CL_MEM_READ_WRITE, bufferRnkSize);
        if (dBufferOutN () == nullptr)
            dBufferOutN = cl::Buffer (context, CL_MEM_READ_WRITE, bufferNSize);
        if (dBufferOutO () == nullptr)
            dBufferOutO = cl::Buffer (context, CL_MEM_READ_WRITE, bufferOSize);
        if (dBufferOutXp () == nullptr)
            dBufferOutXp = cl::Buffer (context, CL_MEM_READ_WRITE, bufferXSize);
        if (dBufferOutIDp () == nullptr && permID == 1)
            dBufferOutIDp = cl::Buffer (context, CL_MEM_READ_WRITE, bufferIDSize);


        rbcCompDists.get (RBCComputeDists<K>::Memory::D_IN_X) = dBufferInX;
        rbcCompDists.get (RBCComputeDists<K>::Memory::D_IN_R) = dBufferInR;
        rbcCompDists.get (RBCComputeDists<K>::Memory::D_OUT_D) = 
            cl::Buffer (context, CL_MEM_READ_WRITE, bufferDSize);
        rbcCompDists.init (nx, nr, d, _a, Staging::NONE);

        rbcMinDists.get (RBCMin::Memory::D_IN_D) = 
            rbcCompDists.get (RBCComputeDists<K>::Memory::D_OUT_D);
        rbcMinDists.get (RBCMin::Memory::D_OUT_ID) = dBufferOutID;
        rbcMinDists.get (RBCMin::Memory::D_OUT_N) = dBufferOutN;
        rbcMinDists.get (RBCMin::Memory::D_OUT_RNK) = dBufferOutRnk;
        rbcMinDists.init (nr, nx, 1, Staging::NONE);

        rbcScanNLists.get (Scan<ScanConfig::EXCLUSIVE>::Memory::D_IN) = dBufferOutN;
        rbcScanNLists.get (Scan<ScanConfig::EXCLUSIVE>::Memory::D_OUT) = dBufferOutO;
        rbcScanNLists.init (nr, 1, Staging::NONE);

        rbcPermDB.get (RBCPermute<P>::Memory::D_IN_X) = dBufferInX;
        rbcPermDB.get (RBCPermute<P>::Memory::D_IN_ID) = dBufferOutID;
        rbcPermDB.get (RBCPermute<P>::Memory::D_IN_RNK) = dBufferOutRnk;
        rbcPermDB.get (RBCPermute<P>::Memory::D_IN_O) = dBufferOutO;
        rbcPermDB.get (RBCPermute<P>::Memory::D_OUT_X_P) = dBufferOutXp;
        rbcPermDB.get (RBCPermute<P>::Memory::D_OUT_ID_P) = dBufferOutIDp;
        rbcPermDB.init (nx, nr, d, permID, Staging::NONE);
    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    template <KernelTypeC K, RBCPermuteConfig P>
    void RBCConstruct<K, P>::write (RBCConstruct::Memory mem, void *ptr, bool block, 
                                    const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case RBCConstruct::Memory::D_IN_X:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + nx * d, hPtrInX);
                    queue.enqueueWriteBuffer (dBufferInX, block, 0, bufferXSize, hPtrInX, events, event);
                    break;
                case RBCConstruct::Memory::D_IN_R:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + nr * d, hPtrInR);
                    queue.enqueueWriteBuffer (dBufferInR, block, 0, bufferRSize, hPtrInR, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    template <KernelTypeC K, RBCPermuteConfig P>
    void* RBCConstruct<K, P>::read (RBCConstruct::Memory mem, bool block, 
                                    const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case RBCConstruct::Memory::H_OUT_ID:
                    queue.enqueueReadBuffer (dBufferOutID, block, 0, bufferIDSize, hPtrOutID, events, event);
                    return hPtrOutID;
                case RBCConstruct::Memory::H_OUT_RNK:
                    queue.enqueueReadBuffer (dBufferOutRnk, block, 0, bufferRnkSize, hPtrOutRnk, events, event);
                    return hPtrOutRnk;
                case RBCConstruct::Memory::H_OUT_N:
                    queue.enqueueReadBuffer (dBufferOutN, block, 0, bufferNSize, hPtrOutN, events, event);
                    return hPtrOutN;
                case RBCConstruct::Memory::H_OUT_O:
                    queue.enqueueReadBuffer (dBufferOutO, block, 0, bufferOSize, hPtrOutO, events, event);
                    return hPtrOutO;
                case RBCConstruct::Memory::H_OUT_X_P:
                    queue.enqueueReadBuffer (dBufferOutXp, block, 0, bufferXSize, hPtrOutXp, events, event);
                    return hPtrOutXp;
                case RBCConstruct::Memory::H_OUT_ID_P:
                    if (permID == 1)
                        queue.enqueueReadBuffer (dBufferOutIDp, block, 0, bufferIDSize, hPtrOutIDp, events, event);
                    return hPtrOutIDp;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the last kernel execution.
     */
    template <KernelTypeC K, RBCPermuteConfig P>
    void RBCConstruct<K, P>::run (const std::vector<cl::Event> *events, cl::Event *event)
    {
        rbcCompDists.run (events);
        rbcMinDists.run ();
        rbcScanNLists.run ();
        rbcPermDB.run (nullptr, event);
    }


    /*! \note If the template parameter, `K`, has the value `SHARED_NONE`, `SHARED_R`, or
     *        `SHARED_X_R`, the parameter \f$ \alpha \f$ is not applicable and the 
     *        value returned is \f$ \infty \f$.
     *  
     *  \return The scaling factor \f$ \alpha \f$.
     */
    template <KernelTypeC K, RBCPermuteConfig P>
    float RBCConstruct<K, P>::getAlpha ()
    {
        return rbcCompDists.getAlpha ();
    }


    /*! \details Updates the kernel argument for the scaling factor \f$ \alpha \f$.
     *  \note If the template parameter, `K`, has the value `SHARED_NONE`, `SHARED_R`, 
     *        or `SHARED_X_R`, the parameter \f$ \alpha \f$ is not applicable and the 
     *        function has no effect.
     *
     *  \param[in] _a scaling factor \f$ \alpha \f$.
     */
    template <KernelTypeC K, RBCPermuteConfig P>
    void RBCConstruct<K, P>::setAlpha (float _a)
    {
        rbcCompDists.setAlpha (_a);
    }


    /*! \brief Template instantiation for the case of the `rbcComputeDists_SharedNone` 
     *         and `rbcPermute` kernels. */
    template class RBCConstruct<KernelTypeC::SHARED_NONE, RBCPermuteConfig::GENERIC>;
    /*! \brief Template instantiation for the case of the `rbcComputeDists_SharedR` 
     *         and `rbcPermute` kernels. */
    template class RBCConstruct<KernelTypeC::SHARED_R, RBCPermuteConfig::GENERIC>;
    /*! \brief Template instantiation for the case of the `rbcComputeDists_SharedXR` 
     *         and `rbcPermute` kernels. */
    template class RBCConstruct<KernelTypeC::SHARED_X_R, RBCPermuteConfig::GENERIC>;
    /*! \brief Template instantiation for the case of the `rbcComputeDists_Kinect_R` 
     *         and `rbcPermute` kernels. */
    template class RBCConstruct<KernelTypeC::KINECT_R, RBCPermuteConfig::GENERIC>;
    /*! \brief Template instantiation for the case of the `rbcComputeDists_Kinect_XR` 
     *         and `rbcPermute` kernels. */
    template class RBCConstruct<KernelTypeC::KINECT_X_R, RBCPermuteConfig::GENERIC>;
    /*! \brief Template instantiation for the case of the `rbcComputeDists_SharedNone` 
     *         and `rbcPermute_Kinect` kernels. */
    template class RBCConstruct<KernelTypeC::SHARED_NONE, RBCPermuteConfig::KINECT>;
    /*! \brief Template instantiation for the case of the `rbcComputeDists_Kinect_R` 
     *         and `rbcPermute_Kinect` kernels. */
    template class RBCConstruct<KernelTypeC::KINECT_R, RBCPermuteConfig::KINECT>;
    /*! \brief Template instantiation for the case of the `rbcComputeDists_Kinect_XR` 
     *         and `rbcPermute_Kinect` kernels. */
    template class RBCConstruct<KernelTypeC::KINECT_X_R, RBCPermuteConfig::KINECT>;


    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. Specifies the context, queue, etc, to be used.
     */
    template <KernelTypeC K, RBCPermuteConfig P>
    RBCSearch<K, P, KernelTypeS::GENERIC>::RBCSearch (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        rbcCompQXDistsKernel (env.getProgram (info.pgIdx), "rbcComputeQXDists"), 
        nnidMinsKernel (env.getProgram (info.pgIdx), "rbcMinDists"), 
        nnidGroupMinsKernel (env.getProgram (info.pgIdx), "rbcGroupMinDists"), 
        rbcNNKernel (env.getProgram (info.pgIdx), "rbcGetNNs"), 
        rbcCompRIDs (env, info), compMaxN (env, info)
    {
        // try
        // {
        //     switch (K)
        //     {
        //         case KernelTypeC::KINECT:
        //             throw "The K = KernelTypeC::KINECT instantiation is not applicable in the "
        //                   "current configuration, S = KernelTypeS::GENERIC";
        //             break;
        //         case KernelTypeC::KINECT_R:
        //             throw "The K = KernelTypeC::KINECT_R instantiation is not applicable in the "
        //                   "current configuration, S = KernelTypeS::GENERIC";
        //             break;
        //         case KernelTypeC::KINECT_X_R:
        //             throw "The K = KernelTypeC::KINECT_X_R instantiation is not applicable in the "
        //                   "current configuration, S = KernelTypeS::GENERIC";
        //             break;
        //     }
        // }
        // catch (const char *error)
        // {
        //     std::cerr << "Error[RBCSearch]: " << error << std::endl;
        //     exit (EXIT_FAILURE);
        // }

        wgMultiple = rbcCompQXDistsKernel.getWorkGroupInfo
            <CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE> (env.devices[info.pIdx][info.dIdx]);
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    template <KernelTypeC K, RBCPermuteConfig P>
    cl::Memory& RBCSearch<K, P, KernelTypeS::GENERIC>::get (RBCSearch::Memory mem)
    {
        switch (mem)
        {
            case RBCSearch::Memory::H_IN_Q:
                return hBufferInQ;
            case RBCSearch::Memory::H_IN_R:
                return hBufferInR;
            case RBCSearch::Memory::H_IN_X_P:
                return hBufferInXp;
            case RBCSearch::Memory::H_IN_O:
                return hBufferInO;
            case RBCSearch::Memory::H_IN_N:
                return hBufferInN;
            case RBCSearch::Memory::H_OUT_R_ID:
                return hBufferOutRID;
            case RBCSearch::Memory::H_OUT_Q_P:
                return hBufferOutQp;
            case RBCSearch::Memory::H_OUT_NN_ID:
                return hBufferOutNNID;
            case RBCSearch::Memory::H_OUT_NN:
                return hBufferOutNN;
            case RBCSearch::Memory::D_IN_Q:
                return dBufferInQ;
            case RBCSearch::Memory::D_IN_R:
                return dBufferInR;
            case RBCSearch::Memory::D_IN_X_P:
                return dBufferInXp;
            case RBCSearch::Memory::D_IN_O:
                return dBufferInO;
            case RBCSearch::Memory::D_IN_N:
                return dBufferInN;
            case RBCSearch::Memory::D_OUT_R_ID:
                return dBufferOutRID;
            case RBCSearch::Memory::D_OUT_Q_P:
                return dBufferOutQp;
            case RBCSearch::Memory::D_OUT_NN_ID:
                return dBufferOutNNID;
            case RBCSearch::Memory::D_OUT_NN:
                return dBufferOutNN;
            case RBCSearch::Memory::D_QR_D:
                return rbcCompRIDs.get (RBCConstruct<K, P>::Memory::D_OUT_D);
            case RBCSearch::Memory::D_QX_D:
                return dBufferQXD;
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *        
     *  \param[in] _nq number of query points.
     *  \param[in] _nr number of representative points.
     *  \param[in] _nx number of database points.
     *  \param[in] _d dimensionality of the associated points.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    template <KernelTypeC K, RBCPermuteConfig P>
    void RBCSearch<K, P, KernelTypeS::GENERIC>::init (
        unsigned int _nq, unsigned int _nr, unsigned int _nx, unsigned int _d, Staging _staging)
    {
        nq = _nq; nr = _nr; nx = _nx; d = _d;
        bufferQSize = nq * d * sizeof (cl_float);
        bufferRSize = nr * d * sizeof (cl_float);
        bufferXSize = nx * d * sizeof (cl_float);
        bufferOSize = nr * sizeof (cl_uint);
        bufferNSize = nr * sizeof (cl_uint);
        bufferRIDSize = nq * sizeof (rbc_dist_id);
        bufferNNIDSize = nq * sizeof (rbc_dist_id);
        bufferNNSize = nq * d * sizeof (cl_float);
        wgXdim = 0;
        staging = _staging;

        cl::Device &device = env.devices[info.pIdx][info.dIdx];
        size_t lXYdim, maxLocalSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE> ();
        for (lXYdim = maxLocalSize; lXYdim > (size_t) std::sqrt (maxLocalSize); lXYdim >>= 1) ;

        try
        {
            if (nq == 0 || nr == 0 || nx == 0)
                throw "The number of points in Q, R or X cannot be zero";
        }
        catch (const char *error)
        {
            std::cerr << "Error[RBCSearch]: " << error << std::endl;
            exit (EXIT_FAILURE);
        }

        // Set workspaces
        globalGNNID = cl::NDRange (wgMultiple, nq);
        globalNN = cl::NDRange (d >> 2, nq);
        localQXD = cl::NDRange (lXYdim, lXYdim);
        local = cl::NDRange (wgMultiple, 1);

        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrInQ = nullptr;
                hPtrInR = nullptr;
                hPtrInXp = nullptr;
                hPtrInO = nullptr;
                hPtrInN = nullptr;
                hPtrOutRID = nullptr;
                hPtrOutQp = nullptr;
                hPtrOutNNID = nullptr;
                hPtrOutNN = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferInQ () == nullptr)
                    hBufferInQ = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferQSize);
                if (hBufferInR () == nullptr)
                    hBufferInR = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferRSize);
                if (hBufferInXp () == nullptr)
                    hBufferInXp = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferXSize);
                if (hBufferInO () == nullptr)
                    hBufferInO = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferOSize);
                if (hBufferInN () == nullptr)
                    hBufferInN = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferNSize);

                hPtrInQ = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInQ, CL_FALSE, CL_MAP_WRITE, 0, bufferQSize);
                hPtrInR = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInR, CL_FALSE, CL_MAP_WRITE, 0, bufferRSize);
                hPtrInXp = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInXp, CL_FALSE, CL_MAP_WRITE, 0, bufferXSize);
                hPtrInO = (cl_uint *) queue.enqueueMapBuffer (
                    hBufferInO, CL_FALSE, CL_MAP_WRITE, 0, bufferOSize);
                hPtrInN = (cl_uint *) queue.enqueueMapBuffer (
                    hBufferInN, CL_FALSE, CL_MAP_WRITE, 0, bufferNSize);
                queue.enqueueUnmapMemObject (hBufferInQ, hPtrInQ);
                queue.enqueueUnmapMemObject (hBufferInR, hPtrInR);
                queue.enqueueUnmapMemObject (hBufferInXp, hPtrInXp);
                queue.enqueueUnmapMemObject (hBufferInO, hPtrInO);
                queue.enqueueUnmapMemObject (hBufferInN, hPtrInN);

                if (!io)
                {
                    queue.finish ();
                    hPtrOutRID = nullptr;
                    hPtrOutQp = nullptr;
                    hPtrOutNNID = nullptr;
                    hPtrOutNN = nullptr;
                    break;
                }

            case Staging::O:
                if (hBufferOutRID () == nullptr)
                    hBufferOutRID = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferRIDSize);
                if (hBufferOutQp () == nullptr)
                    hBufferOutQp = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferQSize);
                if (hBufferOutNNID () == nullptr)
                    hBufferOutNNID = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferNNIDSize);
                if (hBufferOutNN () == nullptr)
                    hBufferOutNN = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferNNSize);

                hPtrOutRID = (rbc_dist_id *) queue.enqueueMapBuffer (
                    hBufferOutRID, CL_FALSE, CL_MAP_READ, 0, bufferRIDSize);
                hPtrOutQp = (cl_float *) queue.enqueueMapBuffer (
                    hBufferOutQp, CL_FALSE, CL_MAP_READ, 0, bufferQSize);
                hPtrOutNNID = (rbc_dist_id *) queue.enqueueMapBuffer (
                    hBufferOutNNID, CL_FALSE, CL_MAP_READ, 0, bufferNNIDSize);
                hPtrOutNN = (cl_float *) queue.enqueueMapBuffer (
                    hBufferOutNN, CL_FALSE, CL_MAP_READ, 0, bufferNNSize);
                queue.enqueueUnmapMemObject (hBufferOutRID, hPtrOutRID);
                queue.enqueueUnmapMemObject (hBufferOutQp, hPtrOutQp);
                queue.enqueueUnmapMemObject (hBufferOutNNID, hPtrOutNNID);
                queue.enqueueUnmapMemObject (hBufferOutNN, hPtrOutNN);
                queue.finish ();

                if (!io)
                {
                    hPtrInQ = nullptr;
                    hPtrInR = nullptr;
                    hPtrInXp = nullptr;
                    hPtrInO = nullptr;
                    hPtrInN = nullptr;
                }
                break;
        }
        
        // Create device buffers
        if (dBufferInQ () == nullptr)
            dBufferInQ = cl::Buffer (context, CL_MEM_READ_ONLY, bufferQSize);
        if (dBufferInR () == nullptr)
            dBufferInR = cl::Buffer (context, CL_MEM_READ_ONLY, bufferRSize);
        if (dBufferInXp () == nullptr)
            dBufferInXp = cl::Buffer (context, CL_MEM_READ_ONLY, bufferXSize);
        if (dBufferInO () == nullptr)
            dBufferInO = cl::Buffer (context, CL_MEM_READ_ONLY, bufferOSize);
        if (dBufferInN () == nullptr)
            dBufferInN = cl::Buffer (context, CL_MEM_READ_ONLY, bufferNSize);
        if (dBufferOutRID () == nullptr)
            dBufferOutRID = cl::Buffer (context, CL_MEM_READ_WRITE, bufferRIDSize);
        if (dBufferOutQp () == nullptr)
            dBufferOutQp = cl::Buffer (context, CL_MEM_READ_WRITE, bufferQSize);
        if (dBufferOutNNID () == nullptr)
            dBufferOutNNID = cl::Buffer (context, CL_MEM_READ_WRITE, bufferNNIDSize);
        if (dBufferOutNN () == nullptr)
            dBufferOutNN = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferNNSize);

        rbcCompRIDs.get (RBCConstruct<K, P>::Memory::D_IN_X) = dBufferInQ;
        rbcCompRIDs.get (RBCConstruct<K, P>::Memory::D_IN_R) = dBufferInR;
        rbcCompRIDs.get (RBCConstruct<K, P>::Memory::D_OUT_X_P) = dBufferOutQp;
        rbcCompRIDs.get (RBCConstruct<K, P>::Memory::D_OUT_ID_P) = dBufferOutRID;
        rbcCompRIDs.init (nq, nr, d, 1.f, 1, Staging::NONE);

        compMaxN.get (Reduce<ReduceConfig::MAX, cl_uint>::Memory::D_IN) = dBufferInN;
        compMaxN.init (nr, 1, Staging::O);

        rbcCompQXDistsKernel.setArg (0, dBufferOutQp);
        rbcCompQXDistsKernel.setArg (1, dBufferInXp);
        rbcCompQXDistsKernel.setArg (3, dBufferInO);
        rbcCompQXDistsKernel.setArg (4, dBufferInN);
        rbcCompQXDistsKernel.setArg (5, dBufferOutRID);
        rbcCompQXDistsKernel.setArg (6, d);

        nnidMinsKernel.setArg (2, dBufferOutNNID);  // Unused
        nnidMinsKernel.setArg (3, dBufferOutNNID);  // Unused
        nnidMinsKernel.setArg (4, cl::Local (2 * local[0] * sizeof (rbc_dist_id)));
        nnidMinsKernel.setArg (6, 0);

        nnidGroupMinsKernel.setArg (1, dBufferOutNNID);
        nnidGroupMinsKernel.setArg (2, dBufferOutNNID);  // Unused
        nnidGroupMinsKernel.setArg (3, dBufferOutNNID);  // Unused
        nnidGroupMinsKernel.setArg (4, cl::Local (2 * local[0] * sizeof (rbc_dist_id)));
        nnidGroupMinsKernel.setArg (6, 0);

        rbcNNKernel.setArg (0, dBufferInXp);
        rbcNNKernel.setArg (1, dBufferOutNN);
        rbcNNKernel.setArg (2, dBufferInO);
        rbcNNKernel.setArg (3, dBufferOutRID);
        rbcNNKernel.setArg (4, dBufferOutNNID);
    }


    /*! \details Computes the maximum representative list cardinality, and sets the 
     *           workspaces and arguments for the execution of the associated kernels.
     */
    template <KernelTypeC K, RBCPermuteConfig P>
    void RBCSearch<K, P, KernelTypeS::GENERIC>::setExecParams (const std::vector<cl::Event> *events)
    {
        // Get maximum representative list cardinality
        // It is assumed the data are already on the device
        compMaxN.run (events);
        max_n = *((cl_uint *) compMaxN.read ());
        max_n = std::max (max_n, 4u);
        // if (max_n % 4) max_n += 4 - (max_n % 4);
        // The x dimension of the global workspace for rbcComputeQXDists not only has to be a 
        // multiple of 4, but it also has to be a multiple of the x dimension of the local workspace
        if ((max_n / 4) % localQXD[0]) max_n += 4 * (localQXD[0] - ((max_n / 4) % localQXD[0]));

        // Establish the number of work-groups per row
        wgXdim = std::ceil (max_n / (float) (8 * wgMultiple));

        bufferQXDSize = nq * max_n * sizeof (cl_float);
        bufferGNNIDSize = wgXdim * (nq * sizeof (rbc_dist_id));

        try
        {
            if (max_n == 0)
                throw "The array QXD cannot have zero columns";

            // (8 * wgMultiple) elements per work-group
            // (2 * wgMultiple) work-groups maximum
            if (max_n > 16 * wgMultiple * wgMultiple)
            {
                std::ostringstream ss;
                ss << "The current configuration of RBCSearch supports arrays ";
                ss << "of up to " << 16 * wgMultiple * wgMultiple << " list cardinalities";
                throw ss.str ().c_str ();
            }
        }
        catch (const char *error)
        {
            std::cerr << "Error[RBCSearch]: " << error << std::endl;
            exit (EXIT_FAILURE);
        }

        // Set workspaces
        globalQXD = cl::NDRange (max_n / 4, nq / 4);
        globalNNID = cl::NDRange (wgXdim * wgMultiple, nq);

        // Create device buffers
        dBufferQXD = cl::Buffer (context, CL_MEM_READ_WRITE, bufferQXDSize);
        if (wgXdim == 1)
            dBufferOutGNNID = cl::Buffer ();
        else
            dBufferOutGNNID = cl::Buffer (context, CL_MEM_READ_WRITE, bufferGNNIDSize);

        // Set kernel arguments
        rbcCompQXDistsKernel.setArg (2, dBufferQXD);

        nnidMinsKernel.setArg (0, dBufferQXD);
        nnidMinsKernel.setArg (5, max_n / 4);

        if (wgXdim == 1)
            nnidMinsKernel.setArg (1, dBufferOutNNID);
        else
            nnidMinsKernel.setArg (1, dBufferOutGNNID);

        nnidGroupMinsKernel.setArg (0, dBufferOutGNNID);
        nnidGroupMinsKernel.setArg (5, (cl_uint) wgXdim);
    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    template <KernelTypeC K, RBCPermuteConfig P>
    void RBCSearch<K, P, KernelTypeS::GENERIC>::write (
        RBCSearch::Memory mem, void *ptr, bool block, const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case RBCSearch::Memory::D_IN_Q:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + nq * d, hPtrInQ);
                    queue.enqueueWriteBuffer (dBufferInQ, block, 0, bufferQSize, hPtrInQ, events, event);
                    break;
                case RBCSearch::Memory::D_IN_R:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + nr * d, hPtrInR);
                    queue.enqueueWriteBuffer (dBufferInR, block, 0, bufferRSize, hPtrInR, events, event);
                    break;
                case RBCSearch::Memory::D_IN_X_P:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + nx * d, hPtrInXp);
                    queue.enqueueWriteBuffer (dBufferInXp, block, 0, bufferXSize, hPtrInXp, events, event);
                    break;
                case RBCSearch::Memory::D_IN_O:
                    if (ptr != nullptr)
                        std::copy ((cl_uint *) ptr, (cl_uint *) ptr + nr, hPtrInO);
                    queue.enqueueWriteBuffer (dBufferInO, block, 0, bufferOSize, hPtrInO, events, event);
                    break;
                case RBCSearch::Memory::D_IN_N:
                    if (ptr != nullptr)
                        std::copy ((cl_uint *) ptr, (cl_uint *) ptr + nr, hPtrInN);
                    queue.enqueueWriteBuffer (dBufferInN, block, 0, bufferNSize, hPtrInN, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    template <KernelTypeC K, RBCPermuteConfig P>
    void* RBCSearch<K, P, KernelTypeS::GENERIC>::read (
        RBCSearch::Memory mem, bool block, const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case RBCSearch::Memory::H_OUT_R_ID:
                    queue.enqueueReadBuffer (dBufferOutRID, block, 0, bufferRIDSize, hPtrOutRID, events, event);
                    return hPtrOutRID;
                case RBCSearch::Memory::H_OUT_Q_P:
                    queue.enqueueReadBuffer (dBufferOutQp, block, 0, bufferQSize, hPtrOutQp, events, event);
                    return hPtrOutQp;
                case RBCSearch::Memory::H_OUT_NN_ID:
                    queue.enqueueReadBuffer (dBufferOutNNID, block, 0, bufferNNIDSize, hPtrOutNNID, events, event);
                    return hPtrOutNNID;
                case RBCSearch::Memory::H_OUT_NN:
                    queue.enqueueReadBuffer (dBufferOutNN, block, 0, bufferNNSize, hPtrOutNN, events, event);
                    return hPtrOutNN;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the last kernel execution.
     *  \param[in] config configuration flag. If true, it runs the necessary kernel, 
     *                    and initializes the remaining parameters and objects.
     */
    template <KernelTypeC K, RBCPermuteConfig P>
    void RBCSearch<K, P, KernelTypeS::GENERIC>::run (
        const std::vector<cl::Event> *events, cl::Event *event, bool config)
    {
        if (config) setExecParams (events);

        // Compute nearest representatives
        rbcCompRIDs.run (events);

        // Compute distances from the points in the representative lists
        queue.enqueueNDRangeKernel (rbcCompQXDistsKernel, cl::NullRange, globalQXD, localQXD);

        // Compute NN ids
        queue.enqueueNDRangeKernel (nnidMinsKernel, cl::NullRange, globalNNID, local);
        if (wgXdim > 1)
            queue.enqueueNDRangeKernel (nnidGroupMinsKernel, cl::NullRange, globalGNNID, local);
        
        // Collect NNs
        queue.enqueueNDRangeKernel (rbcNNKernel, cl::NullRange, globalNN, cl::NullRange, nullptr, event);
    }


    /*! \param[in] _env opencl environment.
     *  \param[in] _info opencl configuration. Specifies the context, queue, etc, to be used.
     */
    template <KernelTypeC K, RBCPermuteConfig P>
    RBCSearch<K, P, KernelTypeS::KINECT>::RBCSearch (clutils::CLEnv &_env, clutils::CLEnvInfo<1> _info) : 
        env (_env), info (_info), 
        context (env.getContext (info.pIdx)), 
        queue (env.getQueue (info.ctxIdx, info.qIdx[0])), 
        rbcCompQXDistsKernel (env.getProgram (info.pgIdx), "rbcComputeQXDists_Kinect"), 
        nnidMinsKernel (env.getProgram (info.pgIdx), "rbcMinDists"), 
        nnidGroupMinsKernel (env.getProgram (info.pgIdx), "rbcGroupMinDists"), 
        rbcNNKernel (env.getProgram (info.pgIdx), "rbcGetNNs"), 
        rbcCompRIDs (env, info), compMaxN (env, info), d (8)
    {
        // try
        // {
        //     switch (K)
        //     {
        //         case KernelTypeC::SHARED_NONE:
        //             throw "The K = KernelTypeC::SHARED_NONE instantiation is not applicable in the "
        //                   "current configuration, S = KernelTypeS::KINECT";
        //             break;
        //         case KernelTypeC::SHARED_R:
        //             throw "The K = KernelTypeC::SHARED_R instantiation is not applicable in the "
        //                   "current configuration, S = KernelTypeS::KINECT";
        //             break;
        //         case KernelTypeC::SHARED_X_R:
        //             throw "The K = KernelTypeC::SHARED_X_R instantiation is not applicable in the "
        //                   "current configuration, S = KernelTypeS::KINECT";
        //             break;
        //     }
        // }
        // catch (const char *error)
        // {
        //     std::cerr << "Error[RBCSearch]: " << error << std::endl;
        //     exit (EXIT_FAILURE);
        // }

        wgMultiple = rbcCompQXDistsKernel.getWorkGroupInfo
            <CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE> (env.devices[info.pIdx][info.dIdx]);
    }


    /*! \details This interface exists to allow CL memory sharing between different kernels.
     *
     *  \param[in] mem enumeration value specifying the requested memory object.
     *  \return A reference to the requested memory object.
     */
    template <KernelTypeC K, RBCPermuteConfig P>
    cl::Memory& RBCSearch<K, P, KernelTypeS::KINECT>::get (RBCSearch::Memory mem)
    {
        switch (mem)
        {
            case RBCSearch::Memory::H_IN_Q:
                return hBufferInQ;
            case RBCSearch::Memory::H_IN_R:
                return hBufferInR;
            case RBCSearch::Memory::H_IN_X_P:
                return hBufferInXp;
            case RBCSearch::Memory::H_IN_O:
                return hBufferInO;
            case RBCSearch::Memory::H_IN_N:
                return hBufferInN;
            case RBCSearch::Memory::H_OUT_R_ID:
                return hBufferOutRID;
            case RBCSearch::Memory::H_OUT_Q_P:
                return hBufferOutQp;
            case RBCSearch::Memory::H_OUT_NN_ID:
                return hBufferOutNNID;
            case RBCSearch::Memory::H_OUT_NN:
                return hBufferOutNN;
            case RBCSearch::Memory::D_IN_Q:
                return dBufferInQ;
            case RBCSearch::Memory::D_IN_R:
                return dBufferInR;
            case RBCSearch::Memory::D_IN_X_P:
                return dBufferInXp;
            case RBCSearch::Memory::D_IN_O:
                return dBufferInO;
            case RBCSearch::Memory::D_IN_N:
                return dBufferInN;
            case RBCSearch::Memory::D_OUT_R_ID:
                return dBufferOutRID;
            case RBCSearch::Memory::D_OUT_Q_P:
                return dBufferOutQp;
            case RBCSearch::Memory::D_OUT_NN_ID:
                return dBufferOutNNID;
            case RBCSearch::Memory::D_OUT_NN:
                return dBufferOutNN;
            case RBCSearch::Memory::D_QR_D:
                return rbcCompRIDs.get (RBCConstruct<K, P>::Memory::D_OUT_D);
            case RBCSearch::Memory::D_QX_D:
                return dBufferQXD;
        }
    }


    /*! \details Sets up memory objects as necessary, and defines the kernel workspaces.
     *  \note If you have assigned a memory object to one member variable of the class 
     *        before the call to `init`, then that memory will be maintained. Otherwise, 
     *        a new memory object will be created.
     *  \attention The \f$ \alpha \f$ parameter **cannot** be \f$0\f$. To overcome this restriction, 
     *             take a look at `euclideanSquaredMetric8` in `kernels/rbc_kernels.cl`.
     *        
     *  \param[in] _nq number of query points.
     *  \param[in] _nr number of representative points.
     *  \param[in] _nx number of database points.
     *  \param[in] _a factor scaling the results of the distance calculations for the 
     *                geometric \f$ x_g \f$ and photometric \f$ x_p \f$ dimensions of 
     *                the \f$ x\epsilon\mathbb{R}^8 \f$ points. That is, \f$ \|x-x'\|_2^2= 
     *                f_g(a)\|x_g-x'_g\|_2^2+f_p(a)\|x_p-x'_p\|_2^2 \f$. This parameter is 
     *                applicable when involving the "Kinect" kernels. That is, when the 
     *                template parameter, `K`, gets the value `KINECT`, `KINECT_R`, or 
     *                `KINECT_X_R`, and the template parameter, `S`, gets the value 
     *                `KINECT`. For more details, take a look at 
     *                `euclideanSquaredMetric8` in `kernels/rbc_kernels.cl`.
     *  \param[in] _staging flag to indicate whether or not to instantiate the staging buffers.
     */
    template <KernelTypeC K, RBCPermuteConfig P>
    void RBCSearch<K, P, KernelTypeS::KINECT>::init (
        unsigned int _nq, unsigned int _nr, unsigned int _nx, float _a, Staging _staging)
    {
        nq = _nq; nr = _nr; nx = _nx; a = _a;
        bufferQSize = nq * d * sizeof (cl_float);
        bufferRSize = nr * d * sizeof (cl_float);
        bufferXSize = nx * d * sizeof (cl_float);
        bufferOSize = nr * sizeof (cl_uint);
        bufferNSize = nr * sizeof (cl_uint);
        bufferRIDSize = nq * sizeof (rbc_dist_id);
        bufferNNIDSize = nq * sizeof (rbc_dist_id);
        bufferNNSize = nq * d * sizeof (cl_float);
        wgXdim = 0;
        staging = _staging;

        cl::Device &device = env.devices[info.pIdx][info.dIdx];
        size_t lXYdim, maxLocalSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE> ();
        for (lXYdim = maxLocalSize; lXYdim > (size_t) std::sqrt (maxLocalSize); lXYdim >>= 1) ;

        try
        {
            if (nq == 0 || nr == 0 || nx == 0)
                throw "The number of points in Q, R or X cannot be zero";
        }
        catch (const char *error)
        {
            std::cerr << "Error[RBCSearch]: " << error << std::endl;
            exit (EXIT_FAILURE);
        }

        // Set workspaces
        globalGNNID = cl::NDRange (wgMultiple, nq);
        globalNN = cl::NDRange (d >> 2, nq);
        localQXD = cl::NDRange (lXYdim, lXYdim);
        local = cl::NDRange (wgMultiple, 1);

        // Create staging buffers
        bool io = false;
        switch (staging)
        {
            case Staging::NONE:
                hPtrInQ = nullptr;
                hPtrInR = nullptr;
                hPtrInXp = nullptr;
                hPtrInO = nullptr;
                hPtrInN = nullptr;
                hPtrOutRID = nullptr;
                hPtrOutQp = nullptr;
                hPtrOutNNID = nullptr;
                hPtrOutNN = nullptr;
                break;

            case Staging::IO:
                io = true;

            case Staging::I:
                if (hBufferInQ () == nullptr)
                    hBufferInQ = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferQSize);
                if (hBufferInR () == nullptr)
                    hBufferInR = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferRSize);
                if (hBufferInXp () == nullptr)
                    hBufferInXp = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferXSize);
                if (hBufferInO () == nullptr)
                    hBufferInO = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferOSize);
                if (hBufferInN () == nullptr)
                    hBufferInN = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferNSize);

                hPtrInQ = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInQ, CL_FALSE, CL_MAP_WRITE, 0, bufferQSize);
                hPtrInR = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInR, CL_FALSE, CL_MAP_WRITE, 0, bufferRSize);
                hPtrInXp = (cl_float *) queue.enqueueMapBuffer (
                    hBufferInXp, CL_FALSE, CL_MAP_WRITE, 0, bufferXSize);
                hPtrInO = (cl_uint *) queue.enqueueMapBuffer (
                    hBufferInO, CL_FALSE, CL_MAP_WRITE, 0, bufferOSize);
                hPtrInN = (cl_uint *) queue.enqueueMapBuffer (
                    hBufferInN, CL_FALSE, CL_MAP_WRITE, 0, bufferNSize);
                queue.enqueueUnmapMemObject (hBufferInQ, hPtrInQ);
                queue.enqueueUnmapMemObject (hBufferInR, hPtrInR);
                queue.enqueueUnmapMemObject (hBufferInXp, hPtrInXp);
                queue.enqueueUnmapMemObject (hBufferInO, hPtrInO);
                queue.enqueueUnmapMemObject (hBufferInN, hPtrInN);

                if (!io)
                {
                    queue.finish ();
                    hPtrOutRID = nullptr;
                    hPtrOutQp = nullptr;
                    hPtrOutNNID = nullptr;
                    hPtrOutNN = nullptr;
                    break;
                }

            case Staging::O:
                if (hBufferOutRID () == nullptr)
                    hBufferOutRID = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferRIDSize);
                if (hBufferOutQp () == nullptr)
                    hBufferOutQp = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferQSize);
                if (hBufferOutNNID () == nullptr)
                    hBufferOutNNID = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferNNIDSize);
                if (hBufferOutNN () == nullptr)
                    hBufferOutNN = cl::Buffer (context, CL_MEM_ALLOC_HOST_PTR, bufferNNSize);

                hPtrOutRID = (rbc_dist_id *) queue.enqueueMapBuffer (
                    hBufferOutRID, CL_FALSE, CL_MAP_READ, 0, bufferRIDSize);
                hPtrOutQp = (cl_float *) queue.enqueueMapBuffer (
                    hBufferOutQp, CL_FALSE, CL_MAP_READ, 0, bufferQSize);
                hPtrOutNNID = (rbc_dist_id *) queue.enqueueMapBuffer (
                    hBufferOutNNID, CL_FALSE, CL_MAP_READ, 0, bufferNNIDSize);
                hPtrOutNN = (cl_float *) queue.enqueueMapBuffer (
                    hBufferOutNN, CL_FALSE, CL_MAP_READ, 0, bufferNNSize);
                queue.enqueueUnmapMemObject (hBufferOutRID, hPtrOutRID);
                queue.enqueueUnmapMemObject (hBufferOutQp, hPtrOutQp);
                queue.enqueueUnmapMemObject (hBufferOutNNID, hPtrOutNNID);
                queue.enqueueUnmapMemObject (hBufferOutNN, hPtrOutNN);
                queue.finish ();

                if (!io)
                {
                    hPtrInQ = nullptr;
                    hPtrInR = nullptr;
                    hPtrInXp = nullptr;
                    hPtrInO = nullptr;
                    hPtrInN = nullptr;
                }
                break;
        }
        
        // Create device buffers
        if (dBufferInQ () == nullptr)
            dBufferInQ = cl::Buffer (context, CL_MEM_READ_ONLY, bufferQSize);
        if (dBufferInR () == nullptr)
            dBufferInR = cl::Buffer (context, CL_MEM_READ_ONLY, bufferRSize);
        if (dBufferInXp () == nullptr)
            dBufferInXp = cl::Buffer (context, CL_MEM_READ_ONLY, bufferXSize);
        if (dBufferInO () == nullptr)
            dBufferInO = cl::Buffer (context, CL_MEM_READ_ONLY, bufferOSize);
        if (dBufferInN () == nullptr)
            dBufferInN = cl::Buffer (context, CL_MEM_READ_ONLY, bufferNSize);
        if (dBufferOutRID () == nullptr)
            dBufferOutRID = cl::Buffer (context, CL_MEM_READ_WRITE, bufferRIDSize);
        if (dBufferOutQp () == nullptr)
            dBufferOutQp = cl::Buffer (context, CL_MEM_READ_WRITE, bufferQSize);
        if (dBufferOutNNID () == nullptr)
            dBufferOutNNID = cl::Buffer (context, CL_MEM_READ_WRITE, bufferNNIDSize);
        if (dBufferOutNN () == nullptr)
            dBufferOutNN = cl::Buffer (context, CL_MEM_WRITE_ONLY, bufferNNSize);

        rbcCompRIDs.get (RBCConstruct<K, P>::Memory::D_IN_X) = dBufferInQ;
        rbcCompRIDs.get (RBCConstruct<K, P>::Memory::D_IN_R) = dBufferInR;
        rbcCompRIDs.get (RBCConstruct<K, P>::Memory::D_OUT_X_P) = dBufferOutQp;
        rbcCompRIDs.get (RBCConstruct<K, P>::Memory::D_OUT_ID_P) = dBufferOutRID;
        rbcCompRIDs.init (nq, nr, d, a, 1, Staging::NONE);

        compMaxN.get (Reduce<ReduceConfig::MAX, cl_uint>::Memory::D_IN) = dBufferInN;
        compMaxN.init (nr, 1, Staging::O);

        rbcCompQXDistsKernel.setArg (0, dBufferOutQp);
        rbcCompQXDistsKernel.setArg (1, dBufferInXp);
        rbcCompQXDistsKernel.setArg (3, dBufferInO);
        rbcCompQXDistsKernel.setArg (4, dBufferInN);
        rbcCompQXDistsKernel.setArg (5, dBufferOutRID);
        rbcCompQXDistsKernel.setArg (6, a);

        nnidMinsKernel.setArg (2, dBufferOutNNID);  // Unused
        nnidMinsKernel.setArg (3, dBufferOutNNID);  // Unused
        nnidMinsKernel.setArg (4, cl::Local (2 * local[0] * sizeof (rbc_dist_id)));
        nnidMinsKernel.setArg (6, 0);

        nnidGroupMinsKernel.setArg (1, dBufferOutNNID);
        nnidGroupMinsKernel.setArg (2, dBufferOutNNID);  // Unused
        nnidGroupMinsKernel.setArg (3, dBufferOutNNID);  // Unused
        nnidGroupMinsKernel.setArg (4, cl::Local (2 * local[0] * sizeof (rbc_dist_id)));
        nnidGroupMinsKernel.setArg (6, 0);

        rbcNNKernel.setArg (0, dBufferInXp);
        rbcNNKernel.setArg (1, dBufferOutNN);
        rbcNNKernel.setArg (2, dBufferInO);
        rbcNNKernel.setArg (3, dBufferOutRID);
        rbcNNKernel.setArg (4, dBufferOutNNID);
    }


    /*! \details Computes the maximum representative list cardinality, and sets the 
     *           workspaces and arguments for the execution of the associated kernels.
     */
    template <KernelTypeC K, RBCPermuteConfig P>
    void RBCSearch<K, P, KernelTypeS::KINECT>::setExecParams (const std::vector<cl::Event> *events)
    {
        // Get maximum representative list cardinality
        // It is assumed the data are already on the device
        compMaxN.run (events);
        max_n = *((cl_uint *) compMaxN.read ());
        max_n = std::max (max_n, 4u);
        // if (max_n % 4) max_n += 4 - (max_n % 4);
        // The x dimension of the global workspace for rbcComputeQXDists_Kinect not only has to be a 
        // multiple of 4, but it also has to be a multiple of the x dimension of the local workspace
        if ((max_n / 4) % localQXD[0]) max_n += 4 * (localQXD[0] - ((max_n / 4) % localQXD[0]));

        // Establish the number of work-groups per row
        wgXdim = std::ceil (max_n / (float) (8 * wgMultiple));

        bufferQXDSize = nq * max_n * sizeof (cl_float);
        bufferGNNIDSize = wgXdim * (nq * sizeof (rbc_dist_id));

        try
        {
            if (max_n == 0)
                throw "The array QXD cannot have zero columns";

            // (8 * wgMultiple) elements per work-group
            // (2 * wgMultiple) work-groups maximum
            if (max_n > 16 * wgMultiple * wgMultiple)
            {
                std::ostringstream ss;
                ss << "The current configuration of RBCSearch supports arrays ";
                ss << "of up to " << 16 * wgMultiple * wgMultiple << " list cardinalities";
                throw ss.str ().c_str ();
            }
        }
        catch (const char *error)
        {
            std::cerr << "Error[RBCSearch]: " << error << std::endl;
            exit (EXIT_FAILURE);
        }

        // Set workspaces
        globalQXD = cl::NDRange (max_n / 4, nq / 4);
        globalNNID = cl::NDRange (wgXdim * wgMultiple, nq);

        // Create device buffers
        dBufferQXD = cl::Buffer (context, CL_MEM_READ_WRITE, bufferQXDSize);
        if (wgXdim == 1)
            dBufferOutGNNID = cl::Buffer ();
        else
            dBufferOutGNNID = cl::Buffer (context, CL_MEM_READ_WRITE, bufferGNNIDSize);

        // Set kernel arguments
        rbcCompQXDistsKernel.setArg (2, dBufferQXD);

        nnidMinsKernel.setArg (0, dBufferQXD);
        nnidMinsKernel.setArg (5, max_n / 4);

        if (wgXdim == 1)
            nnidMinsKernel.setArg (1, dBufferOutNNID);
        else
            nnidMinsKernel.setArg (1, dBufferOutGNNID);

        nnidGroupMinsKernel.setArg (0, dBufferOutGNNID);
        nnidGroupMinsKernel.setArg (5, (cl_uint) wgXdim);
    }


    /*! \details The transfer happens from a staging buffer on the host to the 
     *           associated (specified) device buffer.
     *  
     *  \param[in] mem enumeration value specifying an input device buffer.
     *  \param[in] ptr a pointer to an array holding input data. If not NULL, the 
     *                 data from `ptr` will be copied to the associated staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the write operation to the device buffer.
     */
    template <KernelTypeC K, RBCPermuteConfig P>
    void RBCSearch<K, P, KernelTypeS::KINECT>::write (
        RBCSearch::Memory mem, void *ptr, bool block, const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::I || staging == Staging::IO)
        {
            switch (mem)
            {
                case RBCSearch::Memory::D_IN_Q:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + nq * d, hPtrInQ);
                    queue.enqueueWriteBuffer (dBufferInQ, block, 0, bufferQSize, hPtrInQ, events, event);
                    break;
                case RBCSearch::Memory::D_IN_R:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + nr * d, hPtrInR);
                    queue.enqueueWriteBuffer (dBufferInR, block, 0, bufferRSize, hPtrInR, events, event);
                    break;
                case RBCSearch::Memory::D_IN_X_P:
                    if (ptr != nullptr)
                        std::copy ((cl_float *) ptr, (cl_float *) ptr + nx * d, hPtrInXp);
                    queue.enqueueWriteBuffer (dBufferInXp, block, 0, bufferXSize, hPtrInXp, events, event);
                    break;
                case RBCSearch::Memory::D_IN_O:
                    if (ptr != nullptr)
                        std::copy ((cl_uint *) ptr, (cl_uint *) ptr + nr, hPtrInO);
                    queue.enqueueWriteBuffer (dBufferInO, block, 0, bufferOSize, hPtrInO, events, event);
                    break;
                case RBCSearch::Memory::D_IN_N:
                    if (ptr != nullptr)
                        std::copy ((cl_uint *) ptr, (cl_uint *) ptr + nr, hPtrInN);
                    queue.enqueueWriteBuffer (dBufferInN, block, 0, bufferNSize, hPtrInN, events, event);
                    break;
                default:
                    break;
            }
        }
    }


    /*! \details The transfer happens from a device buffer to the associated 
     *           (specified) staging buffer on the host.
     *  
     *  \param[in] mem enumeration value specifying an output staging buffer.
     *  \param[in] block a flag to indicate whether to perform a blocking 
     *                   or a non-blocking operation.
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the read operation to the staging buffer.
     */
    template <KernelTypeC K, RBCPermuteConfig P>
    void* RBCSearch<K, P, KernelTypeS::KINECT>::read (
        RBCSearch::Memory mem, bool block, const std::vector<cl::Event> *events, cl::Event *event)
    {
        if (staging == Staging::O || staging == Staging::IO)
        {
            switch (mem)
            {
                case RBCSearch::Memory::H_OUT_R_ID:
                    queue.enqueueReadBuffer (dBufferOutRID, block, 0, bufferRIDSize, hPtrOutRID, events, event);
                    return hPtrOutRID;
                case RBCSearch::Memory::H_OUT_Q_P:
                    queue.enqueueReadBuffer (dBufferOutQp, block, 0, bufferQSize, hPtrOutQp, events, event);
                    return hPtrOutQp;
                case RBCSearch::Memory::H_OUT_NN_ID:
                    queue.enqueueReadBuffer (dBufferOutNNID, block, 0, bufferNNIDSize, hPtrOutNNID, events, event);
                    return hPtrOutNNID;
                case RBCSearch::Memory::H_OUT_NN:
                    queue.enqueueReadBuffer (dBufferOutNN, block, 0, bufferNNSize, hPtrOutNN, events, event);
                    return hPtrOutNN;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    }


    /*! \details The function call is non-blocking.
     *
     *  \param[in] events a wait-list of events.
     *  \param[out] event event associated with the last kernel execution.
     *  \param[in] config configuration flag. If true, it runs the necessary kernel, 
     *                    and initializes the remaining parameters and objects.
     */
    template <KernelTypeC K, RBCPermuteConfig P>
    void RBCSearch<K, P, KernelTypeS::KINECT>::run (
        const std::vector<cl::Event> *events, cl::Event *event, bool config)
    {
        if (config) setExecParams (events);

        // Compute nearest representatives
        rbcCompRIDs.run (events);

        // Compute distances from the points in the representative lists
        queue.enqueueNDRangeKernel (rbcCompQXDistsKernel, cl::NullRange, globalQXD, localQXD);

        // Compute NN ids
        queue.enqueueNDRangeKernel (nnidMinsKernel, cl::NullRange, globalNNID, local);
        if (wgXdim > 1)
            queue.enqueueNDRangeKernel (nnidGroupMinsKernel, cl::NullRange, globalGNNID, local);
        
        // Collect NNs
        queue.enqueueNDRangeKernel (rbcNNKernel, cl::NullRange, globalNN, cl::NullRange, nullptr, event);
    }


    /*! \note If the template parameter, `K`, has the value `SHARED_NONE`, `SHARED_R`, or
     *        `SHARED_X_R`, the parameter \f$ \alpha \f$ is not applicable and the 
     *        value returned is \f$ \infty \f$.
     *  
     *  \return The scaling factor \f$ \alpha \f$.
     */
    template <KernelTypeC K, RBCPermuteConfig P>
    float RBCSearch<K, P, KernelTypeS::KINECT>::getAlpha ()
    {
        return a;
    }


    /*! \details Updates the kernel arguments for the scaling factor \f$ \alpha \f$.
     *  \note If the template parameter, `K`, has the value `SHARED_NONE`, `SHARED_R`, 
     *        or `SHARED_X_R`, the parameter \f$ \alpha \f$ is not applicable and the 
     *        function has no effect.
     *
     *  \param[in] _a scaling factor \f$ \alpha \f$.
     */
    template <KernelTypeC K, RBCPermuteConfig P>
    void RBCSearch<K, P, KernelTypeS::KINECT>::setAlpha (float _a)
    {
        a = _a;
        rbcCompRIDs.setAlpha (a);
        rbcCompQXDistsKernel.setArg (6, a);
    }


    /*! \brief Template instantiation for the case of the `rbcComputeDists_SharedNone`, 
     *         `rbcPermute` and `rbcComputeQXDists` kernels. */
    template class RBCSearch<KernelTypeC::SHARED_NONE, RBCPermuteConfig::GENERIC, KernelTypeS::GENERIC>;
    /*! \brief Template instantiation for the case of the `rbcComputeDists_SharedR`, 
     *         `rbcPermute` and `rbcComputeQXDists` kernels. */
    template class RBCSearch<KernelTypeC::SHARED_R, RBCPermuteConfig::GENERIC, KernelTypeS::GENERIC>;
    /*! \brief Template instantiation for the case of the `rbcComputeDists_SharedXR`, 
     *         `rbcPermute` and `rbcComputeQXDists` kernels. */
    template class RBCSearch<KernelTypeC::SHARED_X_R, RBCPermuteConfig::GENERIC, KernelTypeS::GENERIC>;
    /*! \brief Template instantiation for the case of the `rbcComputeDists_Kinect`, 
     *         `rbcPermute` and `rbcComputeQXDists_Kinect` kernels. */
    template class RBCSearch<KernelTypeC::KINECT, RBCPermuteConfig::GENERIC, KernelTypeS::KINECT>;
    /*! \brief Template instantiation for the case of the `rbcComputeDists_Kinect_R`, 
     *         `rbcPermute` and `rbcComputeQXDists_Kinect` kernels. */
    template class RBCSearch<KernelTypeC::KINECT_R, RBCPermuteConfig::GENERIC, KernelTypeS::KINECT>;
    /*! \brief Template instantiation for the case of the `rbcComputeDists_Kinect_XR`, 
     *         `rbcPermute` and `rbcComputeQXDists_Kinect` kernels. */
    template class RBCSearch<KernelTypeC::KINECT_X_R, RBCPermuteConfig::GENERIC, KernelTypeS::KINECT>;
    /*! \brief Template instantiation for the case of the `rbcComputeDists_Kinect`, 
     *         `rbcPermute_Kinect` and `rbcComputeQXDists_Kinect` kernels. */
    template class RBCSearch<KernelTypeC::KINECT, RBCPermuteConfig::KINECT, KernelTypeS::KINECT>;
    /*! \brief Template instantiation for the case of the `rbcComputeDists_Kinect_R`, 
     *         `rbcPermute_Kinect` and `rbcComputeQXDists_Kinect` kernels. */
    template class RBCSearch<KernelTypeC::KINECT_R, RBCPermuteConfig::KINECT, KernelTypeS::KINECT>;
    /*! \brief Template instantiation for the case of the `rbcComputeDists_Kinect_XR`, 
     *         `rbcPermute_Kinect` and `rbcComputeQXDists_Kinect` kernels. */
    template class RBCSearch<KernelTypeC::KINECT_X_R, RBCPermuteConfig::KINECT, KernelTypeS::KINECT>;

}
}
