#include<vector>
#include<iostream>
#include<cmath>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<aruco/markerdetector.h>
#include<opencv2/flann.hpp>

using namespace std;
using namespace cv;
using namespace aruco;

int main(int argc, char** argv) {
	Mat InImg = imread("C:\\Users\\rlckd\\Desktop\\kc\\study\\Computer_Science\\aruco-3.1.12\\cap1.jpg");
	MarkerDetector MDetector;
	vector<Marker> Markers = MDetector.detect(InImg);
	for (int i = 0; i < Markers.size(); i++) {
		cout << Markers[i] << endl;
		Markers[i].draw(InImg, Scalar(0, 0, 255), 2);
	}
	namedWindow("in", 1);
	imshow("in", InImg);
	while (char(waitKey(0)) != 27) {
		;
	}
}