/*! \file kinect_frame_grabber.cpp
 *  \brief Grabs and stores in binary files an RGB and a Depth Kinect frame.
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
#include <fstream>
#include <vector>
#include <algorithm>
#include <mutex>

#include <GL/glew.h>
#if defined(__APPLE__) || defined(__MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <libfreenect.hpp>


// Window parameters
const int gl_win_width = 640;
const int gl_win_height = 480;
int glWinId;

// Model parameters
int mouseX = -1, mouseY = -1;
float angleX = 0.f, angleY = 0.f;
float zoom = 1.f;
bool frame_taken = false;
int frame_id = 0;

// GL texture IDs
GLuint glRGBBuf, glDepthBuf;

// Freenect
class MyFreenectDevice;
Freenect::Freenect freenect;
MyFreenectDevice *device;
double freenectAngle = 0.0;


/*! \brief A class hierarchy for manipulating a mutex. */
class Mutex
{
public:
    void lock () { freenectMutex.lock (); }
    void unlock () { freenectMutex.unlock (); }

    /*! \brief A class that automates the manipulation of 
     *         the outer class instance's mutex.
     *  \details Mutex's mutex is locked with the creation of a 
     *           ScopedLock instance and unlocked with the 
     *           destruction of the ScopedLock instance.
     */
    class ScopedLock
    {
    public:
        ScopedLock (Mutex &mtx) : mMutex (mtx) { mMutex.lock (); }
        ~ScopedLock () { mMutex.unlock (); }

    private:
        Mutex &mMutex;

    };

private:
    /*! A mutex for safely accessing a buffer updated by the freenect thread. */
    std::mutex freenectMutex;

};


/*! \brief A class that extends Freenect::FreenectDevice by defining 
 *         the VideoCallback function so we can be getting updates 
 *         with the latest RGB frame.
 */
class MyFreenectDevice : public Freenect::FreenectDevice
{
public:
    /*! \note The creation of the device is done through the Freenect class.
     *
     *  \param[in] ctx context to open device through (handled by the library).
     *  \param[in] idx index of the device on the bus.
     */
    MyFreenectDevice (freenect_context *ctx, int idx) : 
        Freenect::FreenectDevice (ctx, idx), 
        rgbBuffer (freenect_find_video_mode (
            FREENECT_RESOLUTION_MEDIUM, FREENECT_VIDEO_RGB).bytes), 
        depthBuffer (freenect_find_depth_mode (
            FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_REGISTERED).bytes / 2), 
        newRGBFrame (false), newDepthFrame (false)
    {
        // setVideoFormat (FREENECT_VIDEO_YUV_RGB, FREENECT_RESOLUTION_MEDIUM);
        setDepthFormat (FREENECT_DEPTH_REGISTERED);
    }


    /*! \brief Delivers the latest RGB frame.
     *  \note Do not call directly, it's only used by the library.
     *  
     *  \param[in] rgb an array holding the rgb frame.
     *  \param[in] timestamp a time stamp.
     */
    void VideoCallback (void *rgb, uint32_t timestamp)
    {
        Mutex::ScopedLock lock (rgbMutex);
        
        std::copy ((uint8_t *) rgb, (uint8_t *) rgb + getVideoBufferSize (), rgbBuffer.data ());
        newRGBFrame = true;
    }


    /*! \brief Delivers the latest Depth frame.
     *  \note Do not call directly, it's only used by the library.
     *  
     *  \param[in] depth an array holding the depth frame.
     *  \param[in] timestamp a time stamp.
     */
    void DepthCallback (void *depth, uint32_t timestamp)
    {
        Mutex::ScopedLock lock (depthMutex);
        
        std::copy ((uint16_t *) depth, (uint16_t *) depth + getDepthBufferSize () / 2, depthBuffer.data ());
        newDepthFrame = true;
    }


    /*! \brief Retrieves the most recently received RGB and Depth frames.
     *
     *  \return A flag to indicate whether new frames were present.
     */
    bool updateFrames (std::vector<uint8_t> &rgb, std::vector<uint16_t> &depth)
    {
        Mutex::ScopedLock lockRGB (rgbMutex);
        Mutex::ScopedLock lockDepth (depthMutex);
        
        if (!newRGBFrame || !newDepthFrame)
            return false;

        rgb.swap (rgbBuffer);
        depth.swap (depthBuffer);

        newRGBFrame = false;
        newDepthFrame = false;

        return true;
    }

private:
    Mutex rgbMutex, depthMutex;
    std::vector<uint8_t> rgbBuffer;
    std::vector<uint16_t> depthBuffer;
    bool newRGBFrame, newDepthFrame;

};


/*! \brief Stores the given Kinect frames in binary files. */
void saveBinary (std::vector<uint8_t> &rgb, std::vector<uint16_t> &depth)
{
    std::vector<float> frgba (4 * 640 * 480);
    std::vector<float> fpc4d (4 * 640 * 480);

    for (int p = 0; p < 480*640; ++p)
    {
        frgba[4 * p + 0] = rgb[3 * p]     / 255.f;
        frgba[4 * p + 1] = rgb[3 * p + 1] / 255.f;
        frgba[4 * p + 2] = rgb[3 * p + 2] / 255.f;
        frgba[4 * p + 3] = 1.f;
    }

    for (int y = 0; y < 480; ++y)
    {
        for (int x = 0; x < 640; ++x)
        {
            int p = y * 640 + x;
            float d = (float) depth[p];

            fpc4d[4 * p + 0] = (x - (640 - 1) / 2.f) * d / 595.f;
            fpc4d[4 * p + 1] = (y - (480 - 1) / 2.f) * d / 595.f;
            fpc4d[4 * p + 2] = d;
            fpc4d[4 * p + 3] = 1.f;
        }
    }

    std::ofstream f ("../data/kg_rgba.bin", std::ios::binary);
    f.write ((char *) frgba.data (), 4 * 480 * 640 * sizeof (float));
    f.close ();
    std::cout << "Frame saved in ../data/kg_rgba.bin" << std::endl;

    f.open ("../data/kg_pc4d.bin", std::ios::binary);
    f.write ((char *) fpc4d.data (), 4 * 480 * 640 * sizeof (float));
    f.close ();
    std::cout << "Frame saved in ../data/kg_pc4d.bin" << std::endl << std::endl;
}


/*! \brief Display callback for the window. */
void drawGLScene ()
{
    static std::vector<uint8_t> rgb (3 * 640 * 480);
    static std::vector<uint16_t> depth (640 * 480);

    if (device->updateFrames (rgb, depth))
        frame_id++;

    if (!frame_taken && frame_id == 10)
    {
        saveBinary (rgb, depth);
        frame_taken = true;
    }

    glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glPointSize (1.0f);

    glBegin (GL_POINTS);

    for (int i = 0; i < 480 * 640; ++i)
    {   
        glColor3ub (rgb[3 * i + 0], rgb[3 * i + 1], rgb[3 * i + 2]);

        // Convert from image plane coordinates to world coordinates
        glVertex3f ( (i % 640 - (640 - 1) / 2.f) * depth[i] / 595.f,  // X = (x - cx) * d / fx
                     (i / 640 - (480 - 1) / 2.f) * depth[i] / 595.f,  // Y = (y - cy) * d / fy
                     depth[i] );                                      // Z = d
    }

    glEnd ();
    
    // Draw the world coordinate frame
    glLineWidth (2.0f);
    glBegin (GL_LINES);
    glColor3ub (255, 0, 0);
    glVertex3f (  0, 0, 0);
    glVertex3f ( 50, 0, 0);

    glColor3ub (0, 255, 0);
    glVertex3f (0,   0, 0);
    glVertex3f (0,  50, 0);

    glColor3ub (0, 0, 255);
    glVertex3f (0, 0,   0);
    glVertex3f (0, 0,  50);
    glEnd ();

    // Position the camera
    glMatrixMode (GL_MODELVIEW);
    glLoadIdentity ();
    glScalef (zoom, zoom, 1);
    gluLookAt (-7 * angleX, -7 * angleY, -1000.0,
                       0.0,         0.0,  2000.0,
                       0.0,        -1.0,     0.0 );

    glutSwapBuffers ();
}


/*! \brief Idle callback for the window. */
void idleGLScene ()
{
    glutPostRedisplay ();
}


/*! \brief Reshape callback for the window. */
void resizeGLScene (int width, int height)
{
    glViewport (0, 0, width, height);
    glMatrixMode (GL_PROJECTION);
    glLoadIdentity ();
    gluPerspective (50.0, width / (float) height, 900.0, 11000.0);
    glMatrixMode (GL_MODELVIEW);
}


/*! \brief Keyboard callback for the window. */
void keyPressed (unsigned char key, int x, int y)
{
    switch (key)
    {
        case 0x1B:  // ESC
        case  'Q':
        case  'q':
            glutDestroyWindow (glWinId);
            break;
        case  'W':
        case  'w':
            if (++freenectAngle > 30)
                freenectAngle = 30;
            device->setTiltDegrees (freenectAngle);
            break;
        case  'S':
        case  's':
            if (--freenectAngle < -30)
                freenectAngle = -30;
            device->setTiltDegrees (freenectAngle);
            break;
        case  'R':
        case  'r':
            freenectAngle = 0;
            device->setTiltDegrees (freenectAngle);
            break;
    }
}


/*! \brief Mouse callback for the window. */
void mouseMoved (int x, int y)
{
    if (mouseX >= 0 && mouseY >= 0)
    {
        angleX += x - mouseX;
        angleY += y - mouseY;
    }

    mouseX = x;
    mouseY = y;
}


/*! \brief Mouse button callback for the window. */
void mouseButtonPressed (int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        switch (button)
        {
            case GLUT_LEFT_BUTTON:
                mouseX = x;
                mouseY = y;
                break;
            case 3:  // Scroll Up
                zoom *= 1.2f;
                break;
            case 4:  // Scroll Down
                zoom /= 1.2f;
                break;
        }
    }
    else if (state == GLUT_UP && button == GLUT_LEFT_BUTTON)
    {
        mouseX = -1;
        mouseY = -1;
    }
}


/*! \brief Initializes GLUT. */
void initGL (int argc, char **argv)
{
    glutInit (&argc, argv);
    glutInitDisplayMode (GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA);
    glutInitWindowSize (gl_win_width, gl_win_height);
    glutInitWindowPosition ((glutGet (GLUT_SCREEN_WIDTH) - gl_win_width) / 2,
                            (glutGet (GLUT_SCREEN_HEIGHT) - gl_win_height) / 2 - 70);
    glWinId = glutCreateWindow ("Kinect Frame Grabber");

    glutDisplayFunc (&drawGLScene);
    glutIdleFunc (&idleGLScene);
    glutReshapeFunc (&resizeGLScene);
    glutKeyboardFunc (&keyPressed);
    glutMotionFunc (&mouseMoved);
    glutMouseFunc (&mouseButtonPressed);

    glewInit ();

    glClearColor (0.65f, 0.65f, 0.65f, 1.f);
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable (GL_ALPHA_TEST);
    glAlphaFunc (GL_GREATER, 0.f);
    glEnable (GL_DEPTH_TEST);
    glShadeModel (GL_SMOOTH);
}


// Displays the available controls 
void printInfo ()
{
    std::cout << "\nAvailable Controls:\n";
    std::cout << "===================\n";
    std::cout << " Rotate                     :  Mouse Left Button\n";
    std::cout << " Zoom                       :  Mouse Wheel\n";
    std::cout << " Kinect Tilt Angle  [-/r/+] :  S/R/W\n";
    std::cout << " Quit                       :  Q or Esc\n\n";
}


int main (int argc, char **argv)
{
    try
    {
        printInfo ();

        device = &freenect.createDevice<MyFreenectDevice> (0);
        device->startVideo ();
        device->startDepth ();

        initGL (argc, argv);

        glutMainLoop ();

        device->stopVideo ();
        device->stopDepth ();

        return 0;
    }
    catch (const std::runtime_error &error)
    {
        std::cerr << "Kinect: " << error.what () << std::endl;
    }
    exit (EXIT_FAILURE);
}
