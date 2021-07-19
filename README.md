# AR_markerdetect

## Environment
- Windows10
- Visual Studio 2019 16.0, C++
- OpenCV 4.5.2 , OpenCV-contrib
- ArUco(opencv-contrib) - 6x6-250
- OpenGL 4.3, GLM, freeGLuT, GLSL
- Cam : app(droidCam) + mobile phone

## Reference
- aruco : https://docs.opencv.org/4.5.2/d5/dae/tutorial_aruco_detection.html 
- aruco : https://docs.opencv.org/4.5.2/da/d13/tutorial_aruco_calibration.html
- AR : https://github.com/ramkalath/Augmented_Reality_Tutorials
- GLSL : https://learnopengl.com/Lighting/Multiple-lights
- OpenGL : http://www.opengl-tutorial.org/
- free model obj : https://www.cgtrader.com/free-3d-models/space/spaceship/star-sparrow-modular-spaceship
## Task
### calibration
- init_calibrate.cpp --> using chessboard(charuco)and sample code, camera intrinsic information
- ![image](https://user-images.githubusercontent.com/49244613/126229775-220e0c79-608a-45e6-bd59-2c2496fd79fb.png)
### detect marker
- using sample code --> camera pose estimation ( rvec, tvec)
- ![image](https://user-images.githubusercontent.com/49244613/126230012-361776fd-ab28-46f8-843c-53be8d51ae65.png)

### model rendering
- openGL, GLSL : simple 3d model rendering, shading, texture
- ![ezgif com-gif-maker (3)](https://user-images.githubusercontent.com/49244613/126230961-df96eb7e-170d-49ce-bb26-a53efc3948f6.gif)
