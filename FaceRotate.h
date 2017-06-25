#ifdef FACE_ROTATE_H_H
#define FACE_ROTATE_H_H

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include<dlib/opencv/cv_image.h>
#include <dlib/opencv.h>

using namespace dlib;

frontal_face_detector detector = get_frontal_face_detector();
shape_predictor sp;//Already get

#endif

