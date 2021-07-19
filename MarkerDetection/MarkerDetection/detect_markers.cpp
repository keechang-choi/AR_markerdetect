/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.
						  License Agreement
			   For Open Source Computer Vision Library
					   (3-clause BSD License)
Copyright (C) 2013, OpenCV Foundation, all rights reserved.
Third party copyrights are property of their respective owners.
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
  * Redistributions of source code must retain the above copyright notice,
	this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright notice,
	this list of conditions and the following disclaimer in the documentation
	and/or other materials provided with the distribution.
  * Neither the names of the copyright holders nor the names of the contributors
	may be used to endorse or promote products derived from this software
	without specific prior written permission.
This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/
//sample code from opencv-contrib

#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <glm/glm.hpp> 
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "objloader.hpp"
#include "shader.hpp"
#include "texture.hpp"

using namespace std;
using namespace cv;



namespace {
	const char* about = "Basic marker detection";
	const char* keys =
		"{d        |       | dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,"
		"DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, "
		"DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
		"DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16,"
		"DICT_APRILTAG_16h5=17, DICT_APRILTAG_25h9=18, DICT_APRILTAG_36h10=19, DICT_APRILTAG_36h11=20}"
		"{v        |       | Input from video file, if ommited, input comes from camera }"
		"{ci       | 0     | Camera id if input doesnt come from video (-v) }"
		"{c        |       | Camera intrinsic parameters. Needed for camera pose }"
		"{l        | 0.1   | Marker side length (in meters). Needed for correct scale in camera pose }"
		"{dp       |       | File of marker detector parameters }"
		"{r        |       | show rejected candidates too }"
		"{refine   |       | Corner refinement: CORNER_REFINE_NONE=0, CORNER_REFINE_SUBPIX=1,"
		"CORNER_REFINE_CONTOUR=2, CORNER_REFINE_APRILTAG=3}";
}

static bool readCameraParameters(string filename, Mat& camMatrix, Mat& distCoeffs);
static bool readDetectorParameters(string filename, Ptr<aruco::DetectorParameters>& params);
int init_cam(int argc, char* argv[]);
int detect_markers();
void init_Proj();
void init_bg();
void init_model1();
void init_light();
void init();
void draw_bg();
void draw_model1();
void draw_light();
void display(void);
void reshape(int w, int h);
void timer1(int value);


/**
 */
static bool readCameraParameters(string filename, Mat& camMatrix, Mat& distCoeffs) {
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened())
		return false;
	fs["camera_matrix"] >> camMatrix;
	fs["distortion_coefficients"] >> distCoeffs;
	return true;
}



/**
 */
static bool readDetectorParameters(string filename, Ptr<aruco::DetectorParameters>& params) {
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened())
		return false;
	fs["adaptiveThreshWinSizeMin"] >> params->adaptiveThreshWinSizeMin;
	fs["adaptiveThreshWinSizeMax"] >> params->adaptiveThreshWinSizeMax;
	fs["adaptiveThreshWinSizeStep"] >> params->adaptiveThreshWinSizeStep;
	fs["adaptiveThreshConstant"] >> params->adaptiveThreshConstant;
	fs["minMarkerPerimeterRate"] >> params->minMarkerPerimeterRate;
	fs["maxMarkerPerimeterRate"] >> params->maxMarkerPerimeterRate;
	fs["polygonalApproxAccuracyRate"] >> params->polygonalApproxAccuracyRate;
	fs["minCornerDistanceRate"] >> params->minCornerDistanceRate;
	fs["minDistanceToBorder"] >> params->minDistanceToBorder;
	fs["minMarkerDistanceRate"] >> params->minMarkerDistanceRate;
	fs["cornerRefinementMethod"] >> params->cornerRefinementMethod;
	fs["cornerRefinementWinSize"] >> params->cornerRefinementWinSize;
	fs["cornerRefinementMaxIterations"] >> params->cornerRefinementMaxIterations;
	fs["cornerRefinementMinAccuracy"] >> params->cornerRefinementMinAccuracy;
	fs["markerBorderBits"] >> params->markerBorderBits;
	fs["perspectiveRemovePixelPerCell"] >> params->perspectiveRemovePixelPerCell;
	fs["perspectiveRemoveIgnoredMarginPerCell"] >> params->perspectiveRemoveIgnoredMarginPerCell;
	fs["maxErroneousBitsInBorderRate"] >> params->maxErroneousBitsInBorderRate;
	fs["minOtsuStdDev"] >> params->minOtsuStdDev;
	fs["errorCorrectionRate"] >> params->errorCorrectionRate;
	return true;
}



/**
 */
int dictionaryId;
bool showRejected;
bool estimatePose;
float markerLength;
Ptr<aruco::DetectorParameters> detectorParams;
int camId;
String video;
Ptr<aruco::Dictionary> dictionary;
Mat camMatrix, distCoeffs;
VideoCapture inputVideo;
int waitTime;
double totalTime = 0;
int totalIterations = 0;
Mat frame,frameCopy;




//rendering
//openGL 4.3.0

class RenderOBJ {
public:
	vector<glm::vec3> vertices;
	vector<glm::vec2> uvs;
	vector<glm::vec3> normals;
	GLuint VBO_vert,VBO_uv,VBO_norm, VAO;
	GLuint texture;

	RenderOBJ() {

	}
	bool load_model(const char* path) {
		bool res;
		res = loadOBJ(path, vertices, uvs, normals);
		return res;
	}
};


GLuint programID_bg;
GLuint programID_model1;
GLuint programID_light;

float near_z, far_z, fx, fy, cx, cy;
glm::mat4 Proj;

RenderOBJ bg;
GLfloat vertices_bg[] = {
	-0.5f, -0.5f, 0.0f,   0.5f, -0.5f, 0.0f,  0.5f,  0.5f, 0.0f,
	0.5f,  0.5f, 0.0f,  -0.5f,  0.5f, 0.0f,   -0.5f, -0.5f, 0.0f
	    };
GLfloat uvs_bg[] = {
	0.0f, 0.0f,  1.0f, 0.0f, 1.0f, 1.0f, 
	1.0f, 1.0f,  0.0f, 1.0f,  0.0f, 0.0f };
glm::mat4 M_bg, V_bg;
glm::mat4 MV_bg;

RenderOBJ model1;
glm::mat4 M_m1,V_m1;
Mat rot_mat;
//glm::vec3 light_pos = { 0.0f, 3.0f, 3.0f };
glm::vec3 light_pos;
float theta = 0.0f;

RenderOBJ light;
glm::mat4 M_l1;

int init_cam(int argc, char* argv[]) {
	CommandLineParser parser(argc, argv, keys);
	parser.about(about);

	if (argc < 2) {
		parser.printMessage();
		return 0;
	}

	dictionaryId = parser.get<int>("d");
	showRejected = parser.has("r");
	estimatePose = parser.has("c");
	markerLength = parser.get<float>("l");

	detectorParams = aruco::DetectorParameters::create();
	if (parser.has("dp")) {
		bool readOk = readDetectorParameters(parser.get<string>("dp"), detectorParams);
		if (!readOk) {
			cerr << "Invalid detector parameters file" << endl;
			return 0;
		}
	}

	if (parser.has("refine")) {
		//override cornerRefinementMethod read from config file
		detectorParams->cornerRefinementMethod = parser.get<int>("refine");
	}
	std::cout << "Corner refinement method (0: None, 1: Subpixel, 2:contour, 3: AprilTag 2): " << detectorParams->cornerRefinementMethod << std::endl;

	camId = parser.get<int>("ci");

	if (parser.has("v")) {
		video = parser.get<String>("v");
	}

	if (!parser.check()) {
		parser.printErrors();
		return 0;
	}
	dictionary =
		aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));


	if (estimatePose) {
		bool readOk = readCameraParameters(parser.get<string>("c"), camMatrix, distCoeffs);
		if (!readOk) {
			cerr << "Invalid camera file" << endl;
			return 0;
		}
	}

	if (!video.empty()) {
		inputVideo.open(video);
		waitTime = 0;
	}
	else {
		inputVideo.open(camId);
		waitTime = 10;
	}

	totalTime = 0;
	totalIterations = 0;
}
int detect_markers() {
	if (inputVideo.grab()) {
		inputVideo.retrieve(frame);

		double tick = (double)getTickCount();

		vector< int > ids;
		vector< vector< Point2f > > corners, rejected;
		vector< Vec3d > rvecs, tvecs;

		// detect markers and estimate pose
		aruco::detectMarkers(frame, dictionary, corners, ids, detectorParams, rejected);
		if (estimatePose && ids.size() > 0)
			aruco::estimatePoseSingleMarkers(corners, markerLength, camMatrix, distCoeffs, rvecs,
				tvecs);

		double currentTime = ((double)getTickCount() - tick) / getTickFrequency();
		totalTime += currentTime;
		totalIterations++;
		if (totalIterations % 30 == 0) {
			cout << "Detection Time = " << currentTime * 1000 << " ms "
				<< "(Mean = " << 1000 * totalTime / double(totalIterations) << " ms)" << endl;
		}

		// draw results
		frame.copyTo(frameCopy);
		if (ids.size() > 0) {
			aruco::drawDetectedMarkers(frameCopy, corners, ids);

			if (estimatePose) {
				for (unsigned int i = 0; i < ids.size(); i++)
					aruco::drawAxis(frameCopy, camMatrix, distCoeffs, rvecs[i], tvecs[i],
						markerLength * 0.5f);
				Rodrigues(rvecs[0], rot_mat);
				V_m1 = { rot_mat.at<double>(0,0), rot_mat.at<double>(0,1), rot_mat.at<double>(0,2), tvecs[0][0], -rot_mat.at<double>(1,0), -rot_mat.at<double>(1,1), -rot_mat.at<double>(1,2), -tvecs[0][1], -rot_mat.at<double>(2,0), -rot_mat.at<double>(2,1), -rot_mat.at<double>(2,2), -tvecs[0][2], 0.0f, 0.0f, 0.0f, 1.0f };
				V_m1 = glm::transpose(V_m1);
			}
			
		}

		if (showRejected && rejected.size() > 0)
			aruco::drawDetectedMarkers(frameCopy, rejected, noArray(), Scalar(100, 0, 255));

		//imshow("out", frameCopy);
		//char key = (char)waitKey(waitTime);
		//if (key == 27) break;
	}

	return 0;
}

void init_Proj() {
	near_z = 0.1f;
	far_z = 500.0f;
	fx = camMatrix.at<double>(0, 0);
	fy = camMatrix.at<double>(1, 1);
	cx = camMatrix.at<double>(0, 2);
	cy = camMatrix.at<double>(1, 2);

	Proj = glm::mat4({
		fx/cx,      0,                                0,  0,
		0,      fy/cy,                                0,  0,
		0,          0,   -(far_z+near_z)/(far_z-near_z), -(2 * far_z * near_z) / (far_z - near_z),
		0,          0,                               -1,  0 });
	Proj = glm::transpose(Proj);

}
void init_bg() {
	glUseProgram(programID_bg);
	glGenVertexArrays(1, &bg.VAO);
	glBindVertexArray(bg.VAO);
	
	glGenBuffers(1, &bg.VBO_vert);
	glBindBuffer(GL_ARRAY_BUFFER, bg.VBO_vert);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices_bg), vertices_bg, GL_STATIC_DRAW);
	
	glGenBuffers(1, &bg.VBO_uv);
	glBindBuffer(GL_ARRAY_BUFFER, bg.VBO_uv);
	glBufferData(GL_ARRAY_BUFFER, sizeof(uvs_bg), uvs_bg, GL_STATIC_DRAW);
	
	

	glBindVertexArray(0); 

	if (!inputVideo.grab()) {
		cout << "ERROR : CAM NOT DETECTED " << endl;

		exit(-1);
	}
	inputVideo.retrieve(frame);

	int width = frame.size().width;
	int height = frame.size().height;
	glGenTextures(1, &bg.texture);
	glBindTexture(GL_TEXTURE_2D, bg.texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, frame.data);
	glGenerateMipmap(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, 0);

	float k = far_z - 1;
	M_bg = glm::mat4(1.0f);
	M_bg = M_bg * glm::translate(glm::vec3(0, 0, -k));
	M_bg = M_bg * glm::scale(glm::vec3(2 * k * cx / fx, 2 * k * cy / fy, 0));
	// viewMat : ID
	V_bg = glm::lookAt(glm::vec3(0, 0, 0),
		glm::vec3(0, 0, -1),
		glm::vec3(0, 1, 0));
	cout << "------bg view------" << endl;
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			cout << V_bg[i][j] << " ";
		}
		cout << endl;
	}
	MV_bg = V_bg * M_bg;
}

void init_model1() {
	glUseProgram(programID_model1);
	model1.load_model(".\\models\\StarSparrow1.obj");

	glGenVertexArrays(1, &model1.VAO);
	glBindVertexArray(model1.VAO);

	glGenBuffers(1, &model1.VBO_vert);
	glBindBuffer(GL_ARRAY_BUFFER, model1.VBO_vert);
	glBufferData(GL_ARRAY_BUFFER, model1.vertices.size() * sizeof(glm::vec3), &model1.vertices[0], GL_STATIC_DRAW);
	
	glGenBuffers(1, &model1.VBO_norm);
	glBindBuffer(GL_ARRAY_BUFFER, model1.VBO_norm);
	glBufferData(GL_ARRAY_BUFFER, model1.normals.size() * sizeof(glm::vec3), &model1.normals[0], GL_STATIC_DRAW);


	glGenBuffers(1, &model1.VBO_uv);
	glBindBuffer(GL_ARRAY_BUFFER, model1.VBO_uv);
	glBufferData(GL_ARRAY_BUFFER, model1.uvs.size() * sizeof(glm::vec2), &model1.uvs[0], GL_STATIC_DRAW);
	
	model1.texture = loadBMP_custom(".\\models\\StarSparrow_Red.bmp");


	glBindVertexArray(0);
	cout << "vertices : " << model1.vertices.size() << endl;
	M_m1 = glm::mat4(1.0);
	//M_m1 = M_m1 * glm::translate(glm::vec3(0.0, 0.0, 10.0));
	M_m1 = M_m1*glm::scale(glm::vec3(0.02,0.02,0.02));
	M_m1 = M_m1 * glm::rotate(glm::radians(180.0f), glm::vec3(0.0, 1.0, 0.0));
	M_m1 = M_m1 * glm::rotate(glm::radians(-90.0f), glm::vec3(1.0, 0.0, 0.0));
	V_m1 = glm::mat4({ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 10, 0, 0, 0, 1 });
	V_m1 = glm::transpose(V_m1);
	//V_m1 = glm::lookAt(glm::vec3(0.0f, -5.0f, 5.0f),
	//	glm::vec3(0.0f, 5.0f, 0.0f),
	//	glm::vec3(0.0f, 0.0f, 1.0f));
	cout << "------model1 view------" << endl;
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			cout << V_m1[j][i] << " ";
		}
		cout << endl;
	}
}
void init_light() {
	glUseProgram(programID_light);
	light.load_model(".\\models\\sphere.obj");

	glGenVertexArrays(1, &light.VAO);
	glBindVertexArray(light.VAO);

	glGenBuffers(1, &light.VBO_vert);
	glBindBuffer(GL_ARRAY_BUFFER, light.VBO_vert);
	glBufferData(GL_ARRAY_BUFFER, light.vertices.size() * sizeof(glm::vec3), &light.vertices[0], GL_STATIC_DRAW);
	
	glBindVertexArray(0);

	

}
void init() {
	//glewExperimental = GL_TRUE;
	glewInit();
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_FLAT);
	programID_bg = LoadShaders(".\\shader\\back_vert.vertexshader", ".\\shader\\back_frag.fragmentshader");
	programID_model1 = LoadShaders(".\\shader\\model1_vert.vertexshader", ".\\shader\\model1_frag.fragmentshader");
	programID_light = LoadShaders(".\\shader\\light_vert.vertexshader", ".\\shader\\light_frag.fragmentshader");
	const char* argv4[] = { "dummy", "-c=calib.txt", "-d=10", "-ci=1", "-l=0.1058", };
	init_cam(5, (char**)argv4);
	cout << "-------init cam---------" << endl;
	init_Proj();
	cout << "--------proj done--------" << endl;
	init_bg();
	init_model1();
	init_light();
}
void draw_bg() {
	glUseProgram(programID_bg);

	glBindVertexArray(bg.VAO);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, bg.texture);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frameCopy.cols, frameCopy.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, frameCopy.data);
	glUniform1i(glGetUniformLocation(programID_bg, "camera_texture"), 0);
	glUniformMatrix4fv(glGetUniformLocation(programID_bg, "MV_bg"), 1, GL_FALSE, glm::value_ptr(MV_bg));
	glUniformMatrix4fv(glGetUniformLocation(programID_bg, "Proj_bg"), 1, GL_FALSE, glm::value_ptr(Proj));
	
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, bg.VBO_vert);
	// 1st attribute buffer : vertices
	glVertexAttribPointer(
		0,                  // attribute. No particular reason for 0, but must match the layout in the shader.
		3,                  // size
		GL_FLOAT,           // type
		GL_FALSE,           // normalized?
		0,                  // stride
		(void*)0            // array buffer offset
	);


	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, bg.VBO_uv);
	glVertexAttribPointer(
		1,                                // attribute
		2,                                // size
		GL_FLOAT,                         // type
		GL_FALSE,                         // normalized?
		0,                                // stride
		(void*)0                          // array buffer offset
	);


	glDrawArrays(GL_TRIANGLES, 0, 6);
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glBindVertexArray(0);
}
void draw_model1() {

	glUseProgram(programID_model1);
	glBindVertexArray(model1.VAO);

	
	glUniformMatrix4fv(glGetUniformLocation(programID_model1, "Model"), 1, GL_FALSE, glm::value_ptr(M_m1));
	glUniformMatrix4fv(glGetUniformLocation(programID_model1, "View"), 1, GL_FALSE, glm::value_ptr(V_m1));
	glUniformMatrix4fv(glGetUniformLocation(programID_model1, "Proj"), 1, GL_FALSE, glm::value_ptr(Proj));
	
	glm::vec3 view_pos = glm::vec3(0.0, 0.0, 0.0);
	glUniform3fv(glGetUniformLocation(programID_model1, "viewPos"), 1, glm::value_ptr(view_pos));
	glUniform3fv(glGetUniformLocation(programID_model1, "light_position"), 1, glm::value_ptr(light_pos));
	glUniform1f(glGetUniformLocation(programID_model1, "light_constant"), 1.0f);
	glUniform1f(glGetUniformLocation(programID_model1, "light_linear"), 0.09);
	glUniform1f(glGetUniformLocation(programID_model1, "light_quadratic"), 0.032);
	glUniform3f(glGetUniformLocation(programID_model1, "light_color"), 1.0f, 1.0f, 1.0f);
	
	
	//Bind our texture in Texture Unit 0
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, model1.texture);
	// Set our "" sampler to user Texture Unit 0
	glUniform1i(glGetUniformLocation(programID_model1, "model_texture"), 0);
	

	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, model1.VBO_vert);
	// 1st attribute buffer : vertices
	glVertexAttribPointer(
		0,                  // attribute. No particular reason for 0, but must match the layout in the shader.
		3,                  // size
		GL_FLOAT,           // type
		GL_FALSE,           // normalized?
		0,                  // stride
		(void*)0            // array buffer offset
	);
	
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, model1.VBO_norm);
	// 1st attribute buffer : vertices
	glVertexAttribPointer(
		1,                  // attribute. No particular reason for 0, but must match the layout in the shader.
		3,                  // size
		GL_FLOAT,           // type
		GL_FALSE,           // normalized?
		0,                  // stride
		(void*)0            // array buffer offset
	);

	glEnableVertexAttribArray(2);
	glBindBuffer(GL_ARRAY_BUFFER, model1.VBO_uv);
	glVertexAttribPointer(
		2,                                // attribute
		2,                                // size
		GL_FLOAT,                         // type
		GL_FALSE,                         // normalized?
		0,                                // stride
		(void*)0                          // array buffer offset
	);
	

	glDrawArrays(GL_TRIANGLES, 0, model1.vertices.size());
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	glBindVertexArray(0);
}

void draw_light() {
	glUseProgram(programID_light);
	glBindVertexArray(light.VAO);

	M_l1 = glm::mat4(1.0);
	M_l1 = M_l1 * glm::rotate(theta, glm::vec3(0.0, 0.0, 1.0));
	M_l1 = M_l1 * glm::translate(glm::vec3(0.2f, 0.0f, 0.0f));
	M_l1 = M_l1 * glm::scale(glm::vec3(0.02, 0.02, 0.02));

	glUniformMatrix4fv(glGetUniformLocation(programID_light, "Model"), 1, GL_FALSE, glm::value_ptr(M_l1));
	glUniformMatrix4fv(glGetUniformLocation(programID_light, "View"), 1, GL_FALSE, glm::value_ptr(V_m1));
	glUniformMatrix4fv(glGetUniformLocation(programID_light, "Proj"), 1, GL_FALSE, glm::value_ptr(Proj));
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, light.VBO_vert);
	// 1st attribute buffer : vertices
	glVertexAttribPointer(
		0,                  // attribute. No particular reason for 0, but must match the layout in the shader.
		3,                  // size
		GL_FLOAT,           // type
		GL_FALSE,           // normalized?
		0,                  // stride
		(void*)0            // array buffer offset
	);
	glDrawArrays(GL_TRIANGLES, 0, light.vertices.size());
	glBindVertexArray(0);
}
void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	if (!inputVideo.grab()) {
		return;
	}
	inputVideo.retrieve(frame);
	detect_markers();
	draw_bg();
	draw_model1();
	draw_light();
	glutSwapBuffers();
}
void reshape(int w, int h)
{

	glViewport(0, 0, (GLsizei)w, (GLsizei)h);


	//Proj = glm::frustum(-1.0, 1.0, -1.0, 1.0, 1.5, 40.0);


	glEnable(GL_DEPTH_TEST);
	//glPolygonOffset(1.0, 2);
}
void timer1(int value) {
	int time_cnt = value;
	time_cnt++;
	if (time_cnt % 500 == 0) {
		
	}
	theta = theta + 0.1f;
	if (theta > 360) {
		theta = 0;
	}
	float r = 0.5f;
	light_pos = glm::vec3(r * cos(theta), r * sin(theta), 0.0f);
	glutPostRedisplay();
	glutTimerFunc(10, timer1, time_cnt);
}

int main(int argc, char** argv){
	
	
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	//glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(1000, 700);
	glutInitWindowPosition(100, 50);
	glutCreateWindow(argv[0]);
	cout << "OpenGL version : " << glGetString(GL_VERSION) << endl;

	init();

	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	//glutIdleFunc(moveObjects);

	glutTimerFunc(10, timer1, 0);
	glutMainLoop();
	return 0;
}
