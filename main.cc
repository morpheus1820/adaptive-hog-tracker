
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/LEGACY/legacy.hpp>

#include <cv.h>
#include <ctype.h>
#include <math.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <dirent.h> 
#include <string.h>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <locale.h>
#include <iostream>
#include <ctype.h>
#include "my_HOG.h"
#include "iostream"

#define HYPS_UPDATE 1

using namespace std;
using namespace cv;

//condensation----
// (1)The calculation of the likelihood function
float
calc_likelihood (IplImage * img, int x, int y)
{
	float b, g, r;
	float dist = 0.0, sigma = 50.0;
	
	b = img->imageData[img->widthStep * y + x * 3];       //B
	g = img->imageData[img->widthStep * y + x * 3 + 1];   //G
	r = img->imageData[img->widthStep * y + x * 3 + 2];   //R
	dist = sqrt (b * b + g * g + (255.0 - r) * (255.0 - r));
	//if(dist<255.0)
	//printf("rgb %f %f %f dist %f\n",r,g,b,dist);
	return 1.0 / (sqrt (2.0 * CV_PI) * sigma) * expf (-dist * dist / (2.0 * sigma * sigma));
}




// standard hog training configuration
Size windowsz = Size(64,128);
Size blockSize = Size(16,16);
Size cellSize = Size(8,8);
double wratio=(double)windowsz.height/(double)windowsz.width;
const char* Trainpath = "./dataset/train";
const char* Testpath = "./dataset/test";

// hog training functions
int hogTraining();
void writeVec(FILE* file, vector<float> vec, int cl);
void loadSVMfromModelFile(const char* filename, vector<float>* svm);
void saveSVMtoFile(const char*filename, vector<float> svm);
void buildSet(const char* filename, const char* path);
void evaluateTrainset();
float applyClassifier(vector<float> hog_desc, vector<float> classifier);
void onMouse( int event, int x, int y, int, void* );

//hog detector functions
void hogDetect(Mat &img, HOGDescriptor &hog);
void hogDetectAddSelection(Mat img, HOGDescriptor &hog);
void loadSVMfromFile(const char*filename, vector<float>* svm);

//adaptive hog variables
Mat image;
bool selectObject = false;
Point origin;
Rect selection;
int skipAddSamples=4;
int skipOldSamples=10;

// pf vars
int n_stat = 4;
int n_particle = 5000;
int neff_num_particles=n_particle;
CvConDensation *cond;
CvMat *lowerBound = 0;
CvMat *upperBound = 0;

int resample_count=0;
int minParticles=100;
//double Neff_norm;
float Neff=0.0;

// main vars
int frameNumber=0;
Size frameSize;
double t; // detection time




void adapt_num_particles(int n_particle, float Neff, Size frameSize)
{
	double w = frameSize.width, h = frameSize.height;
	if(Neff==0.0) return;
	// (4)Condensation To create a structure.
	cond = cvCreateConDensation (n_stat, 0, (int)((float)n_particle*Neff));
	
	// (5)To specify the maximum possible minimum state vector for each dimension.
	lowerBound = cvCreateMat (4, 1, CV_32FC1);
	upperBound = cvCreateMat (4, 1, CV_32FC1);
	
	cvmSet (lowerBound, 0, 0, 0.0);
	cvmSet (lowerBound, 1, 0, 0.0);
	cvmSet (lowerBound, 2, 0, -10.0);
	cvmSet (lowerBound, 3, 0, -10.0);
	cvmSet (upperBound, 0, 0, w);
	cvmSet (upperBound, 1, 0, h);
	cvmSet (upperBound, 2, 0, 10.0);
	cvmSet (upperBound, 3, 0, 10.0);
	
	// (6)Condensation Initialize a structure
	cvConDensInitSampleSet (cond, lowerBound, upperBound);
	
	// (7)ConDensation To specify the dynamics of the state vector in the algorithm
	cond->DynamMatr[0] = 1.0;
	cond->DynamMatr[1] = 0.0;
	cond->DynamMatr[2] = 1.0;
	cond->DynamMatr[3] = 0.0;
	cond->DynamMatr[4] = 0.0;
	cond->DynamMatr[5] = 1.0;
	cond->DynamMatr[6] = 0.0;
	cond->DynamMatr[7] = 1.0;
	cond->DynamMatr[8] = 0.0;
	cond->DynamMatr[9] = 0.0;
	cond->DynamMatr[10] = 1.0;
	cond->DynamMatr[11] = 0.0;
	cond->DynamMatr[12] = 0.0;
	cond->DynamMatr[13] = 0.0;
	cond->DynamMatr[14] = 0.0;
	cond->DynamMatr[15] = 1.0;
	
	// (8)Parameters to reconfigure the noise.
	cvRandInit (&(cond->RandS[0]), -25, 25, (int) cvGetTickCount (),CV_RAND_UNI);
	cvRandInit (&(cond->RandS[1]), -25, 25, (int) cvGetTickCount (),CV_RAND_UNI);
	cvRandInit (&(cond->RandS[2]), -5, 5, (int) cvGetTickCount (),CV_RAND_UNI);
	cvRandInit (&(cond->RandS[3]), -5, 5, (int) cvGetTickCount (),CV_RAND_UNI);
	
}


int main(int argc,char **argv )
{
	
	if (argc==1 || strcmp(argv[1],"--help")==0)
	{
		cout << "usage: main videoFile startFrame numParticles w h detector "<<endl;
	}
	int i,j;
	// random walk motion model parameters (px, deg)
	int delta_xy=5;  //5
	int delta_h=10;  //10

	
	namedWindow("main");
//	createTrackbar("cov xy","main",&delta_xy,10);
//	createTrackbar("cov a","main",&delta_h,180);

	// open log files
	FILE *resultsFile=fopen("results_rect.txt","w");
	
	FILE *resultsCenterFile=fopen("results_center.csv","w");

	FILE *resultsTime=fopen("results_time.csv","w");
	
	FILE *resultsPf=fopen("results_pf.csv","w");
	
	FILE *resultsNeff=fopen("results_neff.csv","w");
	
	
	fprintf(resultsFile,"%s\n",argv[1]);
	
	// load video
	VideoCapture cap(argv[1]);
	Mat temp; 
	cap >> temp;
	frameSize=temp.size();

	fprintf(resultsFile,"0 0.0 0.0 0.0 0.0\n");
	frameNumber++;
	
	// create "maps"
	Mat img(frameSize,CV_8UC1);
	Mat imgCol(frameSize,CV_8UC3);
	img.setTo(Scalar(255,255,255));
	imgCol.setTo(Scalar(255,255,255));
	imwrite("blank.pgm",img);
	
	
	// init pf-------
	double w = frameSize.width, h = frameSize.height;
	cond=0;
	Point prevDetection;
	
	int xx, yy;
	// (4)Condensation To create a structure.
	cond = cvCreateConDensation (n_stat, 0, n_particle);
	
	// (5)To specify the maximum possible minimum state vector for each dimension.
	lowerBound = cvCreateMat (4, 1, CV_32FC1);
	upperBound = cvCreateMat (4, 1, CV_32FC1);
	
	cvmSet (lowerBound, 0, 0, 0.0);
	cvmSet (lowerBound, 1, 0, 0.0);
	cvmSet (lowerBound, 2, 0, -10.0);
	cvmSet (lowerBound, 3, 0, -10.0);
	cvmSet (upperBound, 0, 0, w);
	cvmSet (upperBound, 1, 0, h);
	cvmSet (upperBound, 2, 0, 10.0);
	cvmSet (upperBound, 3, 0, 10.0);
	
	// (6)Condensation Initialize a structure
	cvConDensInitSampleSet (cond, lowerBound, upperBound);
	
	// (7)ConDensation To specify the dynamics of the state vector in the algorithm
	cond->DynamMatr[0] = 1.0;
	cond->DynamMatr[1] = 0.0;
	cond->DynamMatr[2] = 1.0;
	cond->DynamMatr[3] = 0.0;
	cond->DynamMatr[4] = 0.0;
	cond->DynamMatr[5] = 1.0;
	cond->DynamMatr[6] = 0.0;
	cond->DynamMatr[7] = 1.0;
	cond->DynamMatr[8] = 0.0;
	cond->DynamMatr[9] = 0.0;
	cond->DynamMatr[10] = 1.0;
	cond->DynamMatr[11] = 0.0;
	cond->DynamMatr[12] = 0.0;
	cond->DynamMatr[13] = 0.0;
	cond->DynamMatr[14] = 0.0;
	cond->DynamMatr[15] = 1.0;
	
	// (8)Parameters to reconfigure the noise.
	cvRandInit (&(cond->RandS[0]), -25, 25, (int) cvGetTickCount (),CV_RAND_UNI);
	cvRandInit (&(cond->RandS[1]), -25, 25, (int) cvGetTickCount (),CV_RAND_UNI);
	cvRandInit (&(cond->RandS[2]), -5, 5, (int) cvGetTickCount (),CV_RAND_UNI);
	cvRandInit (&(cond->RandS[3]), -5, 5, (int) cvGetTickCount (),CV_RAND_UNI);
	//	cvRandInit (&(cond->RandS[0]), -10, 10, (int) cvGetTickCount (),CV_RAND_UNI);
	//	cvRandInit (&(cond->RandS[1]), -10, 10, (int) cvGetTickCount (),CV_RAND_UNI);
	//	cvRandInit (&(cond->RandS[2]), -5, 5, (int) cvGetTickCount (),CV_RAND_UNI);
	//	cvRandInit (&(cond->RandS[3]), -5, 5, (int) cvGetTickCount (),CV_RAND_UNI);	

	
	
	VideoWriter vidout("out.mov", CV_FOURCC('D', 'I', 'V', 'X'),  15, frameSize);
	VideoWriter vidout2("outMap.mov", CV_FOURCC('D', 'I', 'V', 'X'),  15, frameSize);
	
	//hog
	if(argc>4)
	{
		windowsz.width=atoi(argv[4]);
		windowsz.height=atoi(argv[5]);
		wratio=(double)windowsz.height/(double)windowsz.width;
	}
	cout << "hog window size: "<< windowsz.width << " " << windowsz.height<< endl;
	cout << "hog window ratio: " <<wratio<<endl;
	
	HOGDescriptor hog(windowsz, Size(16,16), Size(8,8), Size(8,8),9,1,-1,0,0.2,true);
	
    vector<float> model;
    if(argc<7)
	{
		hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
		cout << "Load default detector"<<endl;
    }
	else
	{
		cout << "Loaded model file: " << argv[6]<< endl;
		loadSVMfromFile(argv[6], &model);
		hog.setSVMDetector(model);			
	}

//    my_HOGDescriptor my_hog;
//    my_hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

    // skip frames at start
    Mat frame;
    if(argc>2)
    {
		for (int i=0;i<atoi(argv[2]);i++)
		{
			cap >> frame;
			fprintf(resultsFile,"%d 0.0 0.0 0.0 0.0\n",frameNumber++);
			
		}
    }

	
	
	// vars for adaptive hog
	setMouseCallback( "main", onMouse, 0 );    
	int posCount=0;
	int negCount=0;
	int windowPosCount=0;
	int windowNegCount=0;
	bool training=false;
	char name[30];
	sprintf(name,"%s/pos.lst",Trainpath);
	FILE *posFile=fopen(name,"w");
	sprintf(name,"%s/neg.lst",Trainpath);
	FILE *negFile=fopen(name,"w");
	
	sprintf(name,"%s/old_pos.lst",Trainpath);
	FILE *oldPosFile=fopen(name,"w");
	
	
	// hog search roi (initialized as whole image)
	Rect searchRoi(0,0,frameSize.width,frameSize.height);

	// state vars for adaptive hog
	bool firstTime=true;
	
	bool pause=false;
	bool startTraining=false;
	bool automaticTraining=false;
	int minPositives=5;
	int minNegatives=10;
	int maxPositives=40;
	int maxNegatives=80;
	bool detect=false;
	bool automaticAddSamples=false;
	long loopCount=0;
	
	Mat positives(Size(windowsz.width/2,windowsz.height/2),CV_8UC3);
	Mat old_positives(Size(windowsz.width/2,windowsz.height/2),CV_8UC3);
	
	Point currDetection;

	bool doDetection=true;
	
	// main loop
	while(1)
	{
		// write info...
		//system("clear");
		if(automaticTraining)	cout << "Automatic Training		ON"<<endl;
		if(automaticAddSamples) cout << "Automatic Add Samples	ON"<<endl;
		if(detect)				cout << "HOG detect				ON"<<endl;
		cout << "# Total Positives: " << posCount <<endl;
		cout << "# Total Negatives: " << negCount <<endl;
		cout << "# Window Positives: " << windowPosCount << " over " <<minPositives<<endl;
		cout << "# Window Negatives: " << windowNegCount << " over " <<minNegatives<<endl;
		cout << endl;
		

		
		cap >> frame;
		if(!frame.data)
		    break;
		Mat img2=frame.clone();
		image=frame.clone();
		Mat hogSearchROI=frame(searchRoi).clone(); // roi for hog getection
		frameNumber++;
		
		// prediction phase???
		if(!firstTime)
		{
//			Size delta;
//			delta.width=currDetection.x-prevDetection.x;
//			delta.height=currDetection.y-prevDetection.y;
//			printf("delta %d %d\n",delta.width,delta.height);
//			for (i = 0; i < n_particle; i++) 
//				if(abs(delta.width)<20 && abs(delta.height)<20 )
//				{
//					cond->flSamples[i][0]+=delta.width;
//					cond->flSamples[i][1]+=delta.height;
//				}
//			printf("Delta: %d %d\n",delta.width, delta.height);
		}
		else firstTime=false;

		
		//-------collecting-samples--------
		if( selectObject && selection.width > 0 && selection.height > 0 )
        {
            Mat roi(img2, selection);
            bitwise_not(roi, roi);
        }
		
		if(automaticAddSamples && frameNumber%skipAddSamples==0)
		{
			hogDetectAddSelection(image(searchRoi),hog);
			selection.x+=searchRoi.x;
			selection.y+=searchRoi.y;
			rectangle(img2,selection.tl(), selection.br(),Scalar(0,255,0),2);
		}
		
		// if rectangle selected save pos and neg images
		if(!selectObject && selection.width > windowsz.width/2 && selection.height >  windowsz.height/2
		   && selection.x>0 && selection.y>0) // && pause)
		{
			// save pos
			char name[40];
			sprintf(name,"%s/pos/sel%d.png",Trainpath, posCount);
			Mat resized(windowsz,CV_8UC3);
			
			
			
			posCount++;
			windowPosCount++;
			cout << "selected window " << selection.x <<" "<< selection.y << " "<<selection.width <<" "<< selection.height <<endl;
			
			if(selection.x<0 || selection.y<0 || selection.width+selection.x>=image.cols || selection.height+selection.y>=image.rows)
				cout << "Selection out of bounds!"<<endl;
			else 				
			{
				
				resize(image(selection),resized,windowsz,INTER_CUBIC);
		
				imwrite(name,resized);
				fprintf(posFile,"pos/sel%d.png\n",posCount-1);
				

				Mat resizedHalf;
				resize(resized,resizedHalf,Size(resized.cols/2,resized.rows/2),INTER_LINEAR);
				if(posCount<5)
					positives.push_back(resizedHalf);
				else {
					positives.t(); positives.push_back(resizedHalf); positives.t();
				}
				
				if((posCount-1) % skipOldSamples==0) // save in old samples list
				{	
					cout << "saving old sample\n";
					sprintf(name,"%s/old/old%d.png",Trainpath, posCount-1);
					imwrite(name,resized);
					fprintf(oldPosFile,"old/old%d.png\n",posCount-1);
					old_positives.push_back(resizedHalf);
				}	

				imshow("positives",positives);
				imshow("old positives",old_positives);
				// save negs
				int innerNegCount=0;
				
				Rect roi1(0,0,image.cols,selection.y);
				cout << "roi1 "<<roi1.x<<" "<<roi1.y<<" "<<roi1.width<<" "<<roi1.height<<endl;
				if(roi1.height>windowsz.height)
				{
					char name[40];
					sprintf(name,"%s/neg/neg%d-%d.png",Trainpath, negCount, innerNegCount++);
					
					imwrite(name,image(roi1));
					fprintf(negFile,"neg/neg%d-%d.png\n",negCount, innerNegCount-1);
				}
				Rect roi2(0,selection.br().y,image.cols,image.rows-selection.br().y);
				cout << "roi2 "<<roi2.x<<" "<<roi2.y<<" "<<roi2.width<<" "<<roi2.height<<endl;
				if(roi2.height>windowsz.height)
				{
					char name[40];
					sprintf(name,"%s/neg/neg%d-%d.png",Trainpath, negCount, innerNegCount++);
					
					imwrite(name,image(roi2));
					fprintf(negFile,"neg/neg%d-%d.png\n", negCount, innerNegCount-1);
				}
				Rect roi3(0,0,selection.x,image.rows);
				cout << "roi3 "<<roi3.x<<" "<<roi3.y<<" "<<roi3.width<<" "<<roi3.height<<endl;
				if(roi3.width>windowsz.width)
				{
					char name[40];
					sprintf(name,"%s/neg/neg%d-%d.png",Trainpath, negCount, innerNegCount++);
					
					imwrite(name,image(roi3));
					fprintf(negFile,"neg/neg%d-%d.png\n",negCount, innerNegCount-1);
				}
				Rect roi4(selection.br().x,0,image.cols-selection.br().x,image.rows);
				cout << "roi4 "<<roi4.x<<" "<<roi4.y<<" "<<roi4.width<<" "<<roi4.height<<endl;
				if(roi4.width>windowsz.width)
				{
					char name[40];
					sprintf(name,"%s/neg/neg%d-%d.png",Trainpath, negCount, innerNegCount++);
					
					imwrite(name,image(roi4));
					fprintf(negFile,"neg/neg%d-%d.png\n",negCount, innerNegCount-1);
				}
				
				// negcount & windowNegcount incrementati solo ogni 4
				negCount++;
				windowNegCount++;
				
				rectangle(img2,selection.tl(), selection.br(),Scalar(0,0,255),2);
				
				
				selection.width=0;
				selection.height=0;
			}	
		}
		
		//----automatic-start-training
		if(automaticTraining && windowPosCount > minPositives && windowNegCount > minNegatives)// && !training)
			startTraining=true;
		
		//----start -training
		if(startTraining)	
		{
			startTraining=false;
			cout << "got enought images, start training...\n";
			fclose(posFile);
			fclose(negFile);
			hogTraining();
			cout << "finished training...\n";
			vector<float> model;
			loadSVMfromFile("modelweight", &model);
			hog.setSVMDetector(model);	
			cout << "new model for HOG!\n";
			windowPosCount=0;
			windowNegCount=0;
			
			sprintf(name,"%s/pos.lst",Trainpath);
			posFile = fopen(name,"wa");
			sprintf(name,"%s/neg.lst",Trainpath);
			negFile = fopen(name,"wa");
			
		}
		
		if(posCount>maxPositives || negCount>maxNegatives)
		{
			cout << "Exceeded max positives samples or negatives samples, resetting dataset"<<endl;
			posCount=0;
			negCount=0;
			fclose(posFile);
			fclose(negFile);
			sprintf(name,"%s/pos.lst",Trainpath);
			posFile = fopen(name,"w");
			sprintf(name,"%s/neg.lst",Trainpath);
			negFile = fopen(name,"w");
		}
		
		
		// temp image for drawing
		Mat temp(img2.size(),CV_8UC3);
		temp.setTo(Scalar(0,0,0));	
		
		
		
		// measurement (hog detection)		
		vector<Rect> found, found_filtered;
		found.clear();found_filtered.clear();
		
		
		Mat heatMap(frame.size(),CV_8UC1);	
		heatMap.setTo(0);
		//my_hog.detectMultiScale(frame, heatMap, found, 0, Size(8,8), Size(32,32), 1.05, 2, frameNumber);
		
		//hog.detectMultiScale(frame, found, 0, Size(8,8), Size(32,32), 1.05, 2);
		t = (double)getTickCount();
		hog.detectMultiScale(hogSearchROI, found, 0, Size(8,8), Size(32,32), 1.05, 2);
		t = (double)getTickCount() - t;
		t=t*1000./cv::getTickFrequency();
		
		fprintf(resultsTime,"%d,%f\n",frameNumber,t);
		
		Mat heatMapCol(heatMap.size(),CV_8UC3);
		vector<Mat> vm;
		vm.push_back(heatMap);vm.push_back(heatMap);vm.push_back(heatMap);
		merge(vm,heatMapCol);
		scaleAdd(heatMapCol,0.5,img2,heatMapCol);
#ifndef HYPS_UPDATE	
		imshow("heatMap",heatMap);
		vidout2 << heatMapCol;
#endif


		// update

		bool foundAtLeastOne=false;
		// filter hog detections for display
		size_t i, j;
		for( i = 0; i < found.size(); i++ )
		{
			Rect r = found[i];
			for( j = 0; j < found.size(); j++ )
				if( j != i && (r & found[j]) == r)
					break;
			if( j == found.size() )
				found_filtered.push_back(r);
		}
		for( i = 0; i < found_filtered.size(); i++ )
		{
			Rect r = found_filtered[i];
			//----da roi a immagine----
			r.x+=searchRoi.x;
			r.y+=searchRoi.y;
			currDetection.x= r.x+r.width/2;
			currDetection.y=r.y+r.height/2;
			//circle(temp, currDetection, 10, Scalar(0,255,255),4);
			//-------------------------
			rectangle(temp,r.tl(),r.br(),Scalar(0,0,255),2);
			cout << "detection: " << r.x << " "<< r.y<<endl; 
			cout << "Search roi: "<<searchRoi.x << " "<<searchRoi.y << " "<<searchRoi.width << " " << searchRoi.height<<endl;
			foundAtLeastOne=true;
			fprintf(resultsFile,"%d %f %f %f %f\n", frameNumber,
													(float) (r.x+r.width/10) /(float)frameSize.width,
													(float)(r.y + r.height/10)/(float)frameSize.height,
													(float)(r.width*0.8)/(float)frameSize.width,
													(float)(r.height*0.8)/(float)frameSize.height);
				

			IplImage *result=cvCreateImage(cvSize(frameSize.width,frameSize.height),IPL_DEPTH_8U,3);
			cvZero(result);

			cvCircle(result,  cvPoint(r.x+r.width/2,r.y+r.height/2),20, CV_RGB(100,0,0), -1,8,0);
			cvSmooth(result,result, CV_GAUSSIAN, 27);
			Mat res(result);imshow("result",res);	
			// update phase
			float total=0.0;
				for (i = 0; i < n_particle; i++) {
					xx = (int) (cond->flSamples[i][0]);
					yy = (int) (cond->flSamples[i][1]);
					if (xx < 0 || xx >= w || yy < 0 || yy >= h) {
						cond->flConfidence[i] = 0.0;
					}
					else {				
						cond->flConfidence[i] = calc_likelihood (result, xx, yy);
						total+=cond->flConfidence[i];
						if(cond->flConfidence[i]>0.0001)
							printf("conf %f\n",cond->flConfidence[i]);
						circle (temp, cvPoint (xx, yy), 2, CV_RGB (cond->flConfidence[i]*200, cond->flConfidence[i]*2000000, 255), -1,8,0);
					}
				}
			
			//normalize weights
			float sumWeightsSquare=0.0;
			for (i = 0; i < n_particle; i++)
			{
				cond->flConfidence[i]/=total;
				sumWeightsSquare+=cond->flConfidence[i]*cond->flConfidence[i];
			}
			
			//neff
			Neff=1.0/sumWeightsSquare;
			Neff /= (float)n_particle;
			
			fprintf(resultsCenterFile,"%d, %d, %d\n",	frameNumber, currDetection.x,currDetection.y);	
			fprintf(resultsNeff,"%f\n",Neff);
			
			searchRoi.width=r.width*2;
			searchRoi.height=r.height*2;
			searchRoi.x=r.x-r.width/2;
			searchRoi.y=r.y-r.height/2;
			if(searchRoi.x<0) searchRoi.x=0;
			if(searchRoi.y<0) searchRoi.y=0;
			if(searchRoi.x+searchRoi.width>frameSize.width) searchRoi.width=frameSize.width-searchRoi.x;
			if(searchRoi.y+searchRoi.height>frameSize.height) searchRoi.height=frameSize.height-searchRoi.y;
			
			
				
			}
	
		if(!foundAtLeastOne)
		{
			searchRoi.width=frameSize.width;
			searchRoi.height=frameSize.height;
			searchRoi.x=0;
			searchRoi.y=0;
			fprintf(resultsCenterFile,"%d,0.0,0.0",	frameNumber);
		}
		else {
			
		}


		
		
		
		// resample
		cvConDensUpdateByTime (cond);
		
		
		//adapt_num_particles(n_particle,Neff,frameSize);
		
		//get best hyp
		Point estimatedPosition((int)cond->State[0], (int)cond->State[1]);
		cout << "Estimated position: "<< estimatedPosition.x << " "<< estimatedPosition.y <<endl;
		fprintf(resultsPf,"%d, %d\n", estimatedPosition.x,estimatedPosition.y);
		
		// show info on image
		// show pf state
		
//		for (i = 0; i < n_particle; i++) {
//			xx = (int) (cond->flSamples[i][0]);
//			yy = (int) (cond->flSamples[i][1]);
//			if (xx < 0 || xx >= w || yy < 0 || yy >= h) {
//				circle (temp, Point (xx, yy), 2, CV_RGB (cond->flConfidence[i]*200, cond->flConfidence[i]*2000000, 255), -1,8,0);
//			}
//		}


		// calculated deltaPose (odometry)
		//deltaPose = pf_vector_coord_sub(max_weight_pose,previousPose);

		// update previous pose
		//previousPose.v[0]=max_weight_pose.v[0];
//		previousPose.v[1]=max_weight_pose.v[1];



//		int ix = MAP_GXWX(m_map, max_weight_pose.v[0]);
//		int iy = MAP_GYWY(m_map, -max_weight_pose.v[1]);
//		circle(temp,Point(ix,iy),8,Scalar(0,255,0),-1);

		//rectangle(temp,Point(2,2),Point(200,100),Scalar(0,0,0),-1);
		char s[50];
		//sprintf(s,"Particles: %d",5000);
//		putText(temp,s,Point(4,14),FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,255,255));
//
//		sprintf(s,"Best hyp: %d %d", ix,iy);
//		putText(temp,s,Point(4,24),FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,255,255));

		rectangle(temp,searchRoi.tl(),searchRoi.br(),Scalar(0,255,255),1);
		scaleAdd(temp,0.95,img2,img2);
		
		circle(img2,estimatedPosition,10,Scalar(0,255,255),2);
		
		imshow("main",img2);

		vidout << img2;	

		// write info on image
		Mat imgInfo(500,200,CV_8UC3);
		imgInfo.setTo(Scalar(0,0,0));

		sprintf(s,"Frame number: %d",frameNumber-1);
		putText(imgInfo,s,Point(2,10),CV_FONT_HERSHEY_PLAIN,0.8,Scalar(0,255,255));
		
		sprintf(s,"Particles: %d", (int)((float)n_particle*(1.0-Neff)));
		putText(imgInfo,s,Point(2,20),CV_FONT_HERSHEY_PLAIN,0.8,Scalar(0,255,255));
//		
//		sprintf(s,"Best hyp: %d %d", ix,iy);
//		putText(imgInfo,s,Point(2,30),CV_FONT_HERSHEY_PLAIN,0.8,Scalar(0,255,255));

		sprintf(s,"N_eff norm: %f", Neff);
		putText(imgInfo,s,Point(2,40),CV_FONT_HERSHEY_PLAIN,0.8,Scalar(0,255,255));
		
		if(automaticTraining)
			putText(imgInfo,"Automatic Training ON",Point(2,50),CV_FONT_HERSHEY_PLAIN,0.8,Scalar(0,255,255));
		else
			putText(imgInfo,"Automatic Training OFF",Point(2,50),CV_FONT_HERSHEY_PLAIN,0.8,Scalar(0,255,255));


		if(automaticAddSamples)
			putText(imgInfo,"Automatic Add Samples ON",Point(2,60),CV_FONT_HERSHEY_PLAIN,0.8,Scalar(0,255,255));
		else 
			putText(imgInfo,"Automatic Add Samples OFF",Point(2,60),CV_FONT_HERSHEY_PLAIN,0.8,Scalar(0,255,255));
		
		sprintf(s,"# Total Positives: %d", posCount);
		putText(imgInfo,s,Point(2,70),CV_FONT_HERSHEY_PLAIN,0.8,Scalar(0,255,255));
		sprintf(s,"# Total Negatives: %d", negCount);
		putText(imgInfo,s,Point(2,80),CV_FONT_HERSHEY_PLAIN,0.8,Scalar(0,255,255));
		sprintf(s,"# Window Positives: %d", windowPosCount);
		putText(imgInfo,s,Point(2,90),CV_FONT_HERSHEY_PLAIN,0.8,Scalar(0,255,255));
		sprintf(s,"# Window Negatives: %d (x4)	", windowNegCount);
		putText(imgInfo,s,Point(2,100),CV_FONT_HERSHEY_PLAIN,0.8,Scalar(0,255,255));
		//sprintf(s,"DeltaPose: %.1f %.1f %.1f", deltaPose.v[0], deltaPose.v[1], deltaPose.v[2]);
		putText(imgInfo,s,Point(2,110),CV_FONT_HERSHEY_PLAIN,0.8,Scalar(0,255,255));
		if(foundAtLeastOne)
			putText(imgInfo,"Object detected",Point(2,120),CV_FONT_HERSHEY_PLAIN,0.8,Scalar(0,255,255));
		else
			putText(imgInfo,"No detection",Point(2,120),CV_FONT_HERSHEY_PLAIN,0.8,Scalar(0,0,255));
		sprintf(s,"Search roi: %d %d %d %d",searchRoi.x,searchRoi.y,searchRoi.width,searchRoi.height);
		putText(imgInfo,s,Point(2,130),CV_FONT_HERSHEY_PLAIN,0.8,Scalar(0,255,255));
		sprintf(s,"Detection time (ms): %f",t);
		putText(imgInfo,s,Point(2,140),CV_FONT_HERSHEY_PLAIN,0.8,Scalar(0,255,255));

		imshow("Info",imgInfo);
		
		
		cout << "Loop "<<loopCount++<<"..."<<endl;
		prevDetection=currDetection;
		char c;		
		if(pause)
			c = (char)waitKey(0);
		else
			c = (char)waitKey(200); //25 fps?
        
		if( c == 27 )
		{
			fclose(resultsFile);
			fclose(resultsCenterFile);
			fclose(resultsPf);
			fclose(resultsTime);
            return 0;
		}
		if( c == ' ')
			pause=!pause;
		if(c == 't')
			startTraining=true;
		if(c=='h')
			detect=!detect;
		if(c=='a')
			automaticTraining=!automaticTraining;
		if(c=='s')
			automaticAddSamples=!automaticAddSamples;
		if(c=='r')
		{
			//pf_init_map(pf, m_map);
			searchRoi.width=frameSize.width;
			searchRoi.height=frameSize.height;
			searchRoi.x=0;
			searchRoi.y=0;
			
		}
		if(firstTime==true)
			firstTime=false;
	}

	fclose(resultsNeff);
		fclose(resultsFile);
		fclose(resultsCenterFile);
		fclose(resultsPf);
		fclose(resultsTime);
	return 0;
}




//--------hog-detection--------------------

void hogDetect(Mat &img, HOGDescriptor &hog)
{
	vector<Rect> found, found_filtered;
	double t = (double)getTickCount();
	// run the detector with default parameters. to get a higher hit-rate
	// (and more false alarms, respectively), decrease the hitThreshold and
	// groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
	hog.detectMultiScale(img, found, 0, Size(8,8), Size(32,32), 1.05, 2);
	t = (double)getTickCount() - t;
	printf("HOG detection time = %gms\n", t*1000./cv::getTickFrequency());
	size_t i, j;
	for( i = 0; i < found.size(); i++ )
	{
		Rect r = found[i];
		for( j = 0; j < found.size(); j++ )
			if( j != i && (r & found[j]) == r)
				break;
		if( j == found.size() )
			found_filtered.push_back(r);
	}
	for( i = 0; i < found_filtered.size(); i++ )
	{
		Rect r = found_filtered[i];
		// the HOG detector returns slightly larger rectangles than the real objects.
		// so we slightly shrink the rectangles to get a nicer output.
		r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width*0.8);
		r.y += cvRound(r.height*0.07);
		r.height = cvRound(r.height*0.8);
		rectangle(img, r.tl(), r.br(), cv::Scalar(255,0,0), 1);
	}
	
}

void hogDetectAddSelection(Mat img, HOGDescriptor &hog)
{
	vector<Rect> found, found_filtered;
	double t = (double)getTickCount();
	// run the detector with default parameters. to get a higher hit-rate
	// (and more false alarms, respectively), decrease the hitThreshold and
	// groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
	hog.detectMultiScale(img, found, 0, Size(8,8), Size(32,32), 1.05, 2);
	t = (double)getTickCount() - t;
	printf("HOG detection time = %gms\n", t*1000./cv::getTickFrequency());
	size_t i, j;
	for( i = 0; i < found.size(); i++ )
	{
		Rect r = found[i];
		for( j = 0; j < found.size(); j++ )
			if( j != i && (r & found[j]) == r)
				break;
		if( j == found.size() )
			found_filtered.push_back(r);
	}
	for( i = 0; i < found_filtered.size(); i++ )
	{
		Rect r = found_filtered[i];
		selection=r;
		selectObject=false;
		// the HOG detector returns slightly larger rectangles than the real objects.
		// so we slightly shrink the rectangles to get a nicer output.
		//		r.x += cvRound(r.width*0.1);
		//		r.width = cvRound(r.width*0.8);
		//		r.y += cvRound(r.height*0.07);
		//		r.height = cvRound(r.height*0.8);
		//		rectangle(img, r.tl(), r.br(), cv::Scalar(0,255,0), 1);
	}
	
}



//------------hog-training---------------------------------------

void onMouse( int event, int x, int y, int, void* )
{
    if( selectObject )
    {
        selection.x = MIN(x, origin.x);
        selection.y = MIN(y, origin.y);
        selection.width = std::abs(x - origin.x);
        //selection.height = std::abs(y - origin.y);
		selection.height=floor(selection.width*wratio);
        selection &= Rect(0, 0, image.cols, image.rows);
    }
	
    switch( event )
    {
		case CV_EVENT_LBUTTONDOWN:
			origin = Point(x,y);
			selection = Rect(x,y,0,0);
			selectObject = true;
			break;
		case CV_EVENT_LBUTTONUP:
			selectObject = false;
			break;
    }
}

int hogTraining() {
	
	setlocale (LC_NUMERIC, "en_GB");
	
	bool b_TS = true;		//write training set
	bool b_TeS = false;		//write test set
	bool b_cvtModel = true;	//convert model file to weight vector file
	bool b_evalTest = false;	//reiterate through negative images of training set and append false positives to training set
	bool b_learn = true;	//use SVMlight to learn
	
	if (b_TS){
		cout << "1. Building Training set..." << endl;
		buildSet("train.dat", Trainpath);
	}
	if (b_TeS){
		cout << "2. Building Test set..." << endl;
		buildSet("test.dat", Testpath);
	}
	if (b_learn){
		cout << "3. Learning..." << endl;
		system("./svm_learn -j 3 train.dat model");
	}
	if (b_cvtModel){
		cout << "Converting Model file..." << endl;
		vector<float> test;
		loadSVMfromModelFile("model", &test);
		saveSVMtoFile("modelweight", test);
	}
	if (b_evalTest){
		cout << "Evaluating Train Set Negatives..." << endl;
		evaluateTrainset();
		if (b_learn){
			pid_t retVal = fork();
			if ( retVal )
			{
				waitpid(retVal,NULL,0);
				printf( "Parent PID: %d\n", getpid() );
			}
			else
			{
				printf( "Child PID: %d\n", getpid() );
				execl("./svm_learn", "-j 3",   "train.dat", "model", (char*) 0);
			}
			if (b_cvtModel){
				cout << "Converting Model file..." << endl;
				vector<float> test;
				loadSVMfromModelFile("model", &test);
				saveSVMtoFile("modelweight", test);
			}
		}
		
	}
    return 0;
}

void writeVec(FILE* file, vector<float> vec, int cl){
	fprintf(file, "%i ", cl);
	for (int i=0; i<vec.size(); i++){
		if (vec[i] == 0.0f)
			continue;
		fprintf(file,"%i:%f ", i+1, vec[i]);
	}
	fprintf(file, "\n");
	return;
}

//loads a file from SVMlight and converts the loaded support vectors to the weight vector.
void loadSVMfromModelFile(const char* filename, vector<float>* svm){
	ifstream svinstr (filename);
	string line;
	float d,g,s,r, b;
	int maxidx,numtrain,numsvm, type;
	
	getline(svinstr, line);
	svinstr >> type;
	if (type != 0){
		cout << "Error: Only linear SVM supported" << endl;
		return;
	}
	getline(svinstr, line);
	svinstr >> d;		//Kernel parameter d...
	
	getline(svinstr, line);
	svinstr >>g;
	getline(svinstr, line);
	svinstr >> s;
	getline(svinstr, line);
	svinstr >> r;
	getline(svinstr, line);
	getline(svinstr, line);
	svinstr >> maxidx;	//highest feature idx
	getline(svinstr, line);
	svinstr >> numtrain;	//num of training vecs
	getline(svinstr, line);
	svinstr >> numsvm;	//num of support vecs
	getline(svinstr, line);
	svinstr >> b;		//offset b;
	getline(svinstr, line);
	
	int cur_svidx = 0;
	svm->clear();
	svm->resize(maxidx+1, 0);
	(*svm)[maxidx] = -b;
	while(!svinstr.eof())
	{
		cur_svidx++;
		if (cur_svidx%20 ==0)
		{
			cout << cvRound((double)cur_svidx/(double)numsvm*100) << "%";
			flush(cout);
		}
		getline(svinstr, line);
		if (line.size() < 5){
			cout << "Skipped line" << endl;
			continue;
		}
		istringstream strstream(line);
		float ftemp;
		int itemp;
		double alpha;
		strstream >> alpha;
		int lastitemp = -1;
		while (!strstream.eof()) {
			strstream >> itemp;
			if (itemp == lastitemp){
				break;
			}
			lastitemp = itemp;
			char x;
			strstream >> x;
			strstream >>ftemp;
			(*svm)[itemp-1] += alpha * ftemp;
		}
		svinstr.sync();
	}
	
}


//loads a file in which every line is one parameter of the svm. (first weight vector w, last one is the offset b)
void loadSVMfromFile(const char*filename, vector<float>* svm){
	FILE* svmin = fopen(filename, "r");
	while(!feof(svmin)){
		float temp;
		fscanf(svmin, "%f\n", &temp);
		svm->push_back(temp);
	}
	fclose(svmin);	
}


//writes a SVM to a file in which every line is one parameter of the svm. (first weight vector w, last one is the offset b)
void saveSVMtoFile(const char*filename, vector<float> svm){
	FILE* svmout = fopen(filename, "w");
	for (int i=0;i<svm.size();i++){
		fprintf(svmout, "%g\n", svm[i]);
		fflush(svmout);
	}
	fclose(svmout);
}



//reads a folder with subfolders "pos" and "neg" and writes Training/Testset for SVMlight
void buildSet(const char* outfile, const char*path){
	char pospath[512], negpath[512];
	sprintf(pospath, "%s/pos", path);
	sprintf(negpath, "%s/neg", path);
	FILE* output = fopen (outfile,"w");
	if (output == NULL)
		return;
	DIR * direc = opendir (pospath);
	if (direc == NULL)
		return;
	struct dirent * file;
	HOGDescriptor* hog = new HOGDescriptor(windowsz, blockSize, cellSize, cellSize,9,1,-1,0,0.2,true);
	
	namedWindow("Images", CV_WINDOW_AUTOSIZE);
	cout << "Positives: ";
	
	
	//loop through all files in pos directory
	char name[50];
	sprintf(name,"%s/pos.lst", Trainpath);
	cout << name <<endl;
	FILE *poss=fopen(name,"r");
	//while ( (file = readdir(direc)) != NULL )
	while ( !feof(poss) )
	{
		char filename[512], temp[50];
		fscanf(poss,"%s\n",temp);
		sprintf(filename,"%s/%s",Trainpath, temp);
		
		//if( strcmp(file ->d_name, ".") == 0 )	//think that check is necessary for MacOS filesyste. Not sure about linux.
		//			continue;
		//		if( strcmp(file->d_name, "..") == 0 )
		//			continue;
		//		char filename[512];
		//		sprintf(filename, "%s/%s", pospath, file->d_name);
		
		Mat image = imread(filename);
		if (image.data == NULL)
			continue;
		
		//if image is larger than window size than just take the centered subpart of the image with window size
		Rect roi = Rect(image.cols/2 - windowsz.width/2,
						image.rows/2 - windowsz.height/2, 
						windowsz.width,
						windowsz.height);
		Mat scale =  image(roi).clone();
		vector<float> desc;
		
		//compute feature vector
		hog->compute(scale, desc,Size(8, 8),Size(0,0));
		writeVec(output, desc, 1);
		fflush(output);
		imshow("Images", scale);
		waitKey(10);
		image.release();
	}
	direc = opendir (negpath);
	if (direc == NULL)
		return;
	
	
	srand ( time(NULL) );
	
	int j = 0;
	
	sprintf(name,"%s/neg.lst", Trainpath);
	cout << name <<endl;
	FILE *negs=fopen(name,"r");
	
	//loop through negative images
	//while ( (file = readdir(direc)) != NULL )
	while ( !feof(negs) )
	{
		char filename[512], temp[50];
		fscanf(negs,"%s\n",temp);
		sprintf(filename,"%s/%s",Trainpath, temp);
		
		//		cout << filename <<endl;
		j++;
		cout << "loop "<<j<<endl;
		
		//		if( strcmp(file ->d_name, ".") == 0 )
		//			continue;
		//		if( strcmp(file->d_name, "..") == 0 )
		//			continue;
		
		//sprintf(filename, "%s/%s", negpath, file->d_name);
		Mat image = imread(filename);
		if (image.data == NULL)
			break;
		Rect roi;
		//take 10 random windows of each negative image
		for (int j = 0; j < 10; j++){
			try {
				Point pt;
				Size sc;
				
				//random width and height
				sc.height = rand()%(image.rows-windowsz.height)+windowsz.height-1;
				sc.width = cvRound((double)sc.height/(double)windowsz.height*(double)windowsz.width);
				if (sc.width > image.cols){
					sc.width = image.cols;
					sc.height = cvRound((double)sc.width/(double)windowsz.width*(double)windowsz.height);
				}
				
				//random top-left point
				pt.x = rand()%(image.cols-sc.width+1)-1;
				if (pt.x<0) pt.x = 0;
				pt.y = rand()%(image.rows-sc.height+1)-1;
				if (pt.y<0) pt.y = 0;
				roi = Rect(pt,sc);
				
				Mat scale;
				resize(image(roi),scale,windowsz);
				imshow("Images", scale);
				waitKey(10);
				vector<float> desc;
				hog->compute(scale, desc,Size(8, 8),Size(0,0));
				writeVec(output, desc, -1);
				fflush(output);
			}
			catch (...) {
				cout << "Error" << endl;
			}
		}
		image.release();
	}
	fclose(output);
}


//goes through negative images of training sets and tries to apply the detector. Each false positive is added to the training file as a hard example
void evaluateTrainset(){
	
	vector<float> classify;
	int  false_pos=0;
	const char* trainfile = "train.dat";
	char negpath[512];
	FILE * output = fopen(trainfile, "a");
	if (output == NULL)
		return;
	
	loadSVMfromFile("modelweight", &classify);
	
	
	sprintf(negpath, "%s/neg", Trainpath);
	DIR * direc;
	struct dirent * file;
	
	HOGDescriptor* hog = new HOGDescriptor(windowsz, blockSize, cellSize, cellSize,9,1,-1,0,0.2,true);
	hog->setSVMDetector(classify);
	namedWindow("Images", 0);
	
	direc = opendir (negpath);
	if (direc == NULL)
		return;
	
	
	srand ( time(NULL) );
	
	int j = 0;
	while ( (file = readdir(direc)) != NULL ){
		j++;
		
		if( strcmp(file ->d_name, ".") == 0 )
			continue;
		if( strcmp(file->d_name, "..") == 0 )
			continue;
		char filename[512];
		sprintf(filename, "%s/%s", negpath, file->d_name);
		Mat image = imread(filename);
		Mat secimg = image.clone();
		if (image.data == NULL)
			break;
		vector<Rect> found;
		hog->detectMultiScale(secimg, found, 0, cellSize, Size(0,0), 1.1, 0);
		
		// training configuration for coarse training
		// 		hog->detectMultiScale(secimg, found, 0, Size(16,16), Size(0,0), 1.1, 0);
		
		for (int i = 0; (i < found.size()); i++){	
			try {
				if ((found[i].x <0)||(found[i].y <0)|| (found[i].x + found[i].width >= image.cols) || (found[i].y + found[i].height >= image.rows))
					continue;
				false_pos ++;
				rectangle(image,found[i].tl(), found[i].br() , Scalar(0,0,255), 3);
				Mat scale;
				resize(secimg(found[i]),scale,windowsz);
				vector<float> desc;
				hog->compute(scale, desc,Size(8, 8),Size(0,0));
				scale.release();
				writeVec(output, desc, -1);
				desc.clear();
				fflush(output);
			}
			catch (...) {
				cout<<"Error";
			}
			
		}
		imshow("Images", image);
		waitKey(10);
		image.release();
		secimg.release();
	}
	cout << "Number of false positives: " << false_pos << endl;
	fclose(output);
	return;
}



float applyClassifier(vector<float> hog_desc, vector<float> classifier){
	float s = classifier.back();
	for (int i = 0; i < hog_desc.size(); i++){
		s += hog_desc[i]*classifier[i];
	}
	return s;
}
//--------------------------------------------------
