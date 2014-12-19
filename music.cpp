/*********************************************************\
| OpenCV Based Gesture Detection and Music Player Control |
| Author: Alankar Kotwal <alankarkotwal13@gmail.com>	  |
| Main File												  |
\*********************************************************/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>
#include <unistd.h>

#define CAP_NO 0
#define ELLIPSE_SCALE 0.75
#define N_NEIGHBORS 1
#define IMAGE_SCALE 1
#define COLOR_CVT CV_BGR2YCrCb
#define SVM_CL 0
#define KNN_CL 1
#define ANN_CL 2
#define TREE_CL 3
#define AVG_CL 4
#define BAYES_CL 5
#define MAX_X_VEL 75
#define MAX_Y_VEL 75
#define MAX_INC 1000 
#define MAX_DEC 1000
#define MAX_EVENT_COUNTER 15

#ifdef AVG_CL
#define CB_HIGH 130
#define CB_LOW 115
#define CR_HIGH 150
#define CR_LOW 130
#endif

#define CLASSIFIER BAYES_CL // One of KNN_CL, SVM_CL, ANN_CL, TREE_CL, AVG_CL, BAYES_CL

#if CLASSIFIER != SVM_CL && CLASSIFIER != KNN_CL && CLASSIFIER != ANN_CL && CLASSIFIER != TREE_CL && CLASSIFIER != AVG_CL && CLASSIFIER != BAYES_CL
#error "Classifier must be one of KNN_CL, SVM_CL, TREE_CL, AVG_CL, BAYES_CL and ANN_CL, case-sensitive."
#endif

using namespace cv;
using namespace std;

int main() {

	/* Setup stuff */

	system("clear");
	
	VideoCapture cap(CAP_NO);
	Mat frame;
	cap>>frame;
	flip(frame, frame, 1);
	
	int imageY = frame.rows;
	int imageX = frame.cols;
	
	int imageYD = frame.rows*IMAGE_SCALE;
	int imageXD = frame.cols*IMAGE_SCALE;
	
	char key = 'f';
	
	#if CLASSIFIER != AVG_CL
	int leftCenterX = imageX/4;
	int rightCenterX = 3*imageX/4;
	int centerY = imageY - 200*ELLIPSE_SCALE;
	
	/* Left Hand Training */
	
	Mat frameDrawn, leftHand;
	namedWindow("Left Hand Training");
	cout<<"[INFO] Put your  left hand into the white oval as well as possible and press c"<<endl;
	while(key != 'c') {
		cap>>frame;
		flip(frame, frame, 1);
		frame.copyTo(frameDrawn);
		ellipse(frameDrawn, Point(leftCenterX, centerY), Size(100*ELLIPSE_SCALE, 200*ELLIPSE_SCALE), 0, 0, 360, Scalar(255, 255, 255), 2);
		imshow("Left Hand Training", frameDrawn);
		key = waitKey(10);
	}
	destroyWindow("Left Hand Training");
	frame.copyTo(leftHand);
	cvtColor(leftHand, leftHand, COLOR_CVT);
	
	/* Right Hand Training */
	
	Mat rightHand;
	namedWindow("Right Hand Training");
	cout<<"[INFO] Put your right hand into the white oval as well as possible and press c"<<endl;
	key = 'f';	
	while(key != 'c') {
		cap>>frame;
		flip(frame, frame, 1);
		frame.copyTo(frameDrawn);
		ellipse(frameDrawn, Point(rightCenterX, centerY), Size(100*ELLIPSE_SCALE, 200*ELLIPSE_SCALE), 0, 0, 360, Scalar(255, 255, 255), 2);
		imshow("Right Hand Training", frameDrawn);
		key = waitKey(10);
	}
	destroyWindow("Right Hand Training");
	frame.copyTo(rightHand);
	cvtColor(rightHand, rightHand, COLOR_CVT);
	
	//frameDrawn = Mat();
	
	/* Training on the data */
	
	Mat leftMask = Mat::zeros(imageY, imageX, CV_8UC1);
	Mat rightMask = Mat::zeros(imageY, imageX, CV_8UC1);
	ellipse(leftMask, Point(leftCenterX, centerY), Size(100*ELLIPSE_SCALE, 200*ELLIPSE_SCALE), 0, 0, 360, Scalar(255), -1); 
	ellipse(rightMask, Point(rightCenterX, centerY), Size(100*ELLIPSE_SCALE, 200*ELLIPSE_SCALE), 0, 0, 360, Scalar(255), -1); 
	
	Mat trainData = Mat::zeros(2*imageY*imageX, 3, CV_32F);
	Mat trainRes = Mat::zeros(2*imageY*imageX, 1, CV_32F);
	
	leftMask.convertTo(leftMask, CV_32F);
	rightMask.convertTo(rightMask, CV_32F);
	leftHand.convertTo(leftHand, CV_32FC3);
	rightHand.convertTo(rightHand, CV_32FC3);
	
	for(int i=0; i<imageY; i++) {
		for(int j=0; j<imageX; j++) {
			for(int ch=0; ch<3; ch++) {
				trainData.at<float>(j+i*imageX, ch) = leftHand.at<Vec3f>(i, j)[ch];
			}
			trainRes.at<float>(j+i*imageX, 0) = leftMask.at<float>(i, j);
		}
	}
	
	for(int i=0; i<imageY; i++) {
		for(int j=0; j<imageX; j++) {
			for(int ch=0; ch<3; ch++) {
				trainData.at<float>(imageY*imageX-1+j+i*imageX, ch) = rightHand.at<Vec3f>(i, j)[ch];
			}
			trainRes.at<float>(imageY*imageX-1+j+i*imageX, 0) = rightMask.at<float>(i, j);
		}
	}
	#endif
	
	//leftHand = Mat();
	//rightHand = Mat();
	
	#if CLASSIFIER == KNN_CL
	CvKNearest classifier(trainData, trainRes);
	#elif CLASSIFIER == SVM_CL
	CvSVMParams params;
	params.svm_type	= CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
	CvSVM SVM;
	SVM.train(trainData, trainRes, Mat(), Mat(), params);
	#elif CLASSIFIER == TREE_CL
	CvDTree tree;
	tree.train(trainData, CV_ROW_SAMPLE, trainRes);	
	#elif CLASSIFIER == AVG_CL
	// Do nothing
	#elif CLASSIFIER == BAYES_CL
	CvNormalBayesClassifier bayes(trainData, trainRes);
	#else // ANN by default
	CvANN_MLP_TrainParams::CvANN_MLP_TrainParams() {
		term_crit = cvTermCriteria( CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 0.01 );
		train_method = RPROP;
		bp_dw_scale = bp_moment_scale = 0.1;
		rp_dw0 = 0.1; rp_dw_plus = 1.2; rp_dw_minus = 0.5;
		rp_dw_min = FLT_EPSILON; rp_dw_max = 50.;
	}
	CvANN_MLP ann;
	ann.create()
	#endif
	
	//trainData = Mat();
	//trainRes = Mat();
	
	/* Now use the trained data to detect skin */
	Mat skinMap = Mat::zeros(imageYD, imageXD, CV_32F);
	Mat skinMapInt = Mat::zeros(imageYD, imageXD, CV_8U);
	//namedWindow("Skin Map", WINDOW_AUTOSIZE);
	Mat testData = Mat::zeros(imageYD*imageXD, 3, CV_32F);
	Mat testRes = Mat::zeros(imageYD*imageXD, 1, CV_32F);
	
	/*Ptr<BackgroundSubtractor> pMOG; //MOG Background subtractor
	Ptr<BackgroundSubtractor> pMOG2; //MOG2 Background subtractor
	pMOG= new BackgroundSubtractorMOG(); //MOG approach
    pMOG2 = new BackgroundSubtractorMOG2(); //MOG2 approach
	Mat fgMaskMOG; //fg mask generated by MOG method
	Mat fgMaskMOG2; //fg mask generated by MOG2 method*/
	
	Moments mu;
	int cenX=0, cenY=0, oldCenX=0, oldCenY=0;
	
	int eventCounter = 0;
	bool first=1;
	Mat background;
	
	cout<<"[INFO] Starting in 2 seconds..."<<endl;
	sleep(2);
	cout<<"[INFO] Up and running."<<endl;
	
	while(cap.isOpened()) {
	
		//frame = imread("test.jpg");
		cap>>frame;
		flip(frame, frame, 1);
		cvtColor(frame, frame, COLOR_CVT);
		#if CLASSIFIER != AVG_CL
		frame.convertTo(frame, CV_32FC3);
		#endif
		//resize(frame, frame, Size(0, 0), IMAGE_SCALE, IMAGE_SCALE, INTER_LINEAR);
		
		#if CLASSIFIER != AVG_CL
		for(int i=0; i<imageYD; i++) {
			for(int j=0; j<imageXD; j++) {
				for(int ch=0; ch<3; ch++) {
					testData.at<float>(j+i*imageX, ch) = frame.at<Vec3f>(i, j)[ch];
				}
			}
		}
		#endif
		
		#if CLASSIFIER == KNN_CL
		Mat neighborResponses, dists;
		classifier.find_nearest(testData, N_NEIGHBORS, testRes, neighborResponses, dists);
		#elif CLASSIFIER == SVM_CL
		SVM.predict(testData, testRes);
		#elif CLASSIFIER == TREE_CL
		// Ditch for now
		#elif CLASSIFIER == AVG_CL
		inRange(frame, Scalar(0, CR_LOW, CB_LOW), Scalar(255, CR_HIGH, CB_HIGH), skinMap);
		//cvtColor(frame, frame, CV_YCrCb2BGR);
		//imshow("lol", frame);
		//waitKey(0);
		#elif CLASSIFIER == BAYES_CL
		bayes.predict(testData, &testRes);
		#else
		// @TODO: ANN stuff
		#endif
		
		//testData = Mat();
		
		#if CLASSIFIER != AVG_CL
		for(int i=0; i<imageYD; i++) {
			for(int j=0; j<imageXD; j++) {
					skinMap.at<float>(i, j) = testRes.at<float>(j+i*imageX, 0);
			}
		}
		#endif
		
		if(first) {
			background = skinMap;
			first = 0;
		}
		else {
		
			/*Mat element = getStructuringElement(MORPH_RECT, Size(2, 2));
			morphologyEx(skinMap, skinMap, MORPH_OPEN, element);*/
		
			/*skinMap.convertTo(skinMapInt, CV_8U);
			pMOG->operator()(skinMapInt, fgMaskMOG);
			pMOG2->operator()(skinMapInt, fgMaskMOG2);
			
			imshow("FG Mask MOG", fgMaskMOG);
			imshow("FG Mask MOG 2", fgMaskMOG2);*/
			
			//absdiff(skinMap, background, skinMap);
		
			mu = moments(skinMap);
			cenX = mu.m10/mu.m00;
			cenY = mu.m01/mu.m00;
		
			std::cout<<cenX<<endl;
		
			if(oldCenX != 0 && oldCenY != 0) {
				if(cenX - oldCenX > MAX_X_VEL && oldCenX < imageX/3) {
					if(eventCounter == 0) {
						system("rhythmbox-client --next");
						eventCounter = 1;
					}
				}
				else if(oldCenX - cenX > MAX_X_VEL && oldCenX > 2*imageX/3) {
					if(eventCounter == 0) {
						system("rhythmbox-client --previous");
						eventCounter = 1;
					}
				}
			}
		
			if(eventCounter != 0) {
				eventCounter = (eventCounter == MAX_EVENT_COUNTER)?0:eventCounter+1;
			}
		
			oldCenX = cenX;
			oldCenY = cenY;
		
			imshow("Skin Map", skinMap);
			key = waitKey(10);
			if(key == 27) {
				cap.release();
				break;
			}	
		}	
	}

	destroyAllWindows();

	return 0;
}
