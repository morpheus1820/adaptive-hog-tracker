/*
 *  my_HOG.h
 *  HOG test
 *
 *  Created by David Lenz on 17.09.10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

/****************************************************************************************\
 *            HOG (Histogram-of-Oriented-Gradients) Descriptor and Object Detector        *
 \****************************************************************************************/

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <iostream>

using namespace cv;

struct  my_HOGDescriptor
{
public:
    enum { L2Hys=0 };
	
    my_HOGDescriptor() : winSize(64,128), blockSize(16,16), blockStride(8,8),   //blockstride (8,8)
	cellSize(8,8), nbins(9), derivAperture(1), winSigma(-1),
	histogramNormType(L2Hys), L2HysThreshold(0.2), gammaCorrection(true), frameNumber(0)
    {}
	
    my_HOGDescriptor(Size _winSize, Size _blockSize, Size _blockStride,
				  Size _cellSize, int _nbins, int _derivAperture=1, double _winSigma=-1,
				  int _histogramNormType=L2Hys, double _L2HysThreshold=0.2, bool _gammaCorrection=false)
	: winSize(_winSize), blockSize(_blockSize), blockStride(_blockStride), cellSize(_cellSize),
	nbins(_nbins), derivAperture(_derivAperture), winSigma(_winSigma),
	histogramNormType(_histogramNormType), L2HysThreshold(_L2HysThreshold),
	gammaCorrection(_gammaCorrection), frameNumber(0)
    {}
	
    my_HOGDescriptor(const String& filename)
    {
        load(filename);
    }
	
    virtual ~my_HOGDescriptor() {}
	
    size_t getDescriptorSize() const;
    bool checkDetectorSize() const;
    double getWinSigma() const;
	
    virtual void setSVMDetector(const vector<float>& _svmdetector);
	
    virtual bool load(const String& filename, const String& objname=String());
    virtual void save(const String& filename, const String& objname=String()) const;
	
    virtual void compute(const Mat& img,
                         vector<float>& descriptors,
                         Size winStride=Size(), Size padding=Size(),
                         const vector<Point>& locations=vector<Point>()) const;
    virtual void detect(const Mat& img, vector<Point>& foundLocations,
                        double hitThreshold=0, Size winStride=Size(),
                        Size padding=Size(),
                        const vector<Point>& searchLocations=vector<Point>()) const;
//    virtual Mat detectAndExtractHeatMap(const Mat& img,
//                        double hitThreshold=0, Size winStride=Size(),
//                        Size padding=Size()) const;
    virtual void detectMultiScale(const Mat& img,Mat & heatMap, vector<Rect>& foundLocations,
                                  double hitThreshold=0, Size winStride=Size(),
                                  Size padding=Size(), double scale=1.05,
                                  int groupThreshold=2, int frameNumber=0) const;
	virtual void detectMultiScale(const Mat& img, vector<Rect>& foundLocations,
						  double hitThreshold, Size winStride, Size padding,
						  double scale0, int groupThreshold, double scalemin, double scalemax,
						  int frameNumber) const;
	
    virtual void computeGradient(const Mat& img, Mat& grad, Mat& angleOfs,
                                 Size paddingTL=Size(), Size paddingBR=Size()) const;
	
    static vector<float> getDefaultPeopleDetector();
	
    vector<Mat> getResult();

    Size winSize;
    Size blockSize;
    Size blockStride;
    Size cellSize;
    int nbins;
    int derivAperture;
    double winSigma;
    int histogramNormType;
    double L2HysThreshold;
    bool gammaCorrection;
    vector<float> svmDetector;

    long frameNumber;
    vector<Mat> m_result;
};
