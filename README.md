# cv-open

## Computer Vision Code for OpenCV

As much as possible, the code in this repository should be OpenCV-specific without requiring the assistance of models and conveniences. The code in this repos should use just what OpenCV offers. When it comes time to start designing the alpha product, this repos will provide a body of best practices of OpenCV. Or at least learn ways to peform various actions/functions/activities.

On examination of OpenCV tutorials and documentation, the product has a huge number of highly detailed functions and features. The majority of these features are perhaps too low-level and specialized to warrant a deep dive. OpenCV is happy using DNNs and pretrained models and using these models is probably the easiest and quickest way forward.

If you really want a deeper dive on OpenCV, contours is probably the most useful feature to study because it seems to be the foundation for so many other capabilities. It converts color images into b&w images and then can determine the contours ("outlines" or "shapes") inside an image. Then, based on the contours it found, it can calculate the center point, the area, the perimeter, etc. And those contours also determine the bounding box dimensions, best-fit circle & ellipse, etc., which shows up in obj det and segmentation, etc.

## Goals

* Explore OpenCV to learn what it can do on its own
* Develop a body of commonly used actions that can be used in larger projects
* Device actions
  * Open, read, and close a file
  * Show the contents of a file (video or still)
  * Find your camera port(s), start a camera, stop a camera
* Demonstrate some basics, such as:
  * Draw a circle
  * Draw lines
  * Find contours in images and draw the contours on the image
  * Track objects using different Tracker classes (no dnn or pre-trained models)
  * Show some camera filters, such as Blur, Canny, or GoodFeaturesToTrack (find corners)

* Avoid kludging together a bunch of  "stuff" by knowing OpenCV just well enough to make resilliant code.
* Learn about the strengths, weaknesses and conventions of OpenCV.

## Useful Images

A large variety of images that show up often in OpenCV courses & classes & and examples are located at

`https://github.com/opencv/opencv/tree/4.x/samples/data`

## Useful Tutorials

A significant number of tutorials on all things OpenCV are available starting at:

    `https://docs.opencv.org/4.11.0/d9/df8/tutorial_root.html`
    (This course sometimes uses out of date formats or commands, even when 4.11 version selected).

Instructor-led videos and Jupyter Notebooks on Google Colab are available at:

    courses.opencv.org/dashboard
    (You will need to create a free signin. For pay courses in greater detail are also available)