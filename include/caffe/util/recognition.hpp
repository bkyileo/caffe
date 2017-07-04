/*******************************************
*Author:yifan zhang(1.0)
*Date:02/07/2017
*Mail:bkyifanleo@gmail.com
*Copyright:BUPT
********************************************/
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "boost/algorithm/string.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/signal_handler.h"
#include "caffe/util/vis.hpp"
#include "caffe/util/detector.hpp"
#include "caffe/util/license_plate_division.h"
#include "caffe/util/classifier.hpp"
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <vector>
#include <string>
#include <iostream>
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cassert>
#ifndef RECOGNITIONER
#define RECOGNITIONER
using namespace caffe;
using namespace std;
using namespace cv;

class Recognitioner{
    private:
		string detector_proto_file_;
		string detector_model_file_;
		string detector_config_file_;
		
		string digit_classifier_proto_file_;
		string digit_classifier_model_file_;
		string digit_classifier_label_file_;
		
		string characters_classifier_proto_file_;
		string characters_classifier_model_file_;
		string characters_classifier_label_file_;
		
		vector<Mat> crops_;
		vector<caffe::BBox<float> > detector_results_;
		vector<vector<Mat>> segment_results_;
		vector<std::string> digit_classifier_results_;
		vector<std::string> characters_classifier_results_;
		vector<bool> mask_;
		
        std::shared_ptr<Detector> detector_;
        std::shared_ptr<Classifier> digit_classifier_;
        std::shared_ptr<Classifier> characters_classifier_;
		
    public:
        void SetDetectorProtoFile(string proto_file);
        void SetDetectorModelFile(string model_file);
        void SetDetectorConfigFile(string config_file);
        void SetSetDetectorParm();
        
        void SetDigitClassifierProtoFile(string proto_file);
        void SetDigitClassifierModelFile(string model_file);
        void SetDigitClassifierLabelFile(string label_file);
        
        void SetCharactersClassifierProtoFile(string proto_file);
        void SetCharactersClassifierModelFile(string model_file);
        void SetCharactersClassifierLabelFile(string label_file);
        
        void DetectorInit();
        void DigitClassifierInit();
        void CharactersClassifierInit();
        
		void DetectionProcess(Mat image);
		void SegmentProcess(Mat image);
		void DigitClassifyProcess();
		void CharactersClassifyProcess();
		void RecogProcess(Mat image);
		   
 		vector<Mat> GetDetectorCrops();
		vector<caffe::BBox<float> > GetDetectorResults();
		vector<vector<Mat>> GetSegmentResults();
		vector<string> GetDigitClassifierResults();
		vector<string> GetCharactersClassifierResults();       

};
        
int getdir (string dir, vector<string> &files);

#endif 
