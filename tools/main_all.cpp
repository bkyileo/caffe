#include <gflags/gflags.h>
#include <glog/logging.h>
#include "boost/algorithm/string.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/signal_handler.h"
#include "caffe/util/vis.hpp"
#include "caffe/util/recognition.hpp"
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <vector>
#include <string>
#include <iostream>
#include <caffe/caffe.hpp>
//using namespace caffe;
using namespace std;
using namespace cv;

int main(int argc, char** argv){
  FLAGS_alsologtostderr = 1;
  gflags::SetVersionString(AS_STRING(CAFFE_VERSION));
  caffe::Caffe::SetDevice(0);
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
  caffe::GlobalInit(&argc,&argv);
  
  string pwd = "/home/bk/test/caffe/models/";
  
  Recognitioner recoger;
  recoger.SetDetectorProtoFile( pwd + "detector.prototxt" );
  recoger.SetDetectorModelFile( pwd + "detector.caffemodel");
  recoger.SetDetectorConfigFile( pwd + "config.json");
  recoger.SetSetDetectorParm();
  
  recoger.SetDigitClassifierProtoFile( pwd + "digit.prototxt" );
  recoger.SetDigitClassifierModelFile( pwd + "digit.caffemodel" );
  recoger.SetDigitClassifierLabelFile( pwd + "digit.txt" );
  
  recoger.DetectorInit();
  recoger.DigitClassifierInit();

  Mat image;
  image = imread(pwd+"100.jpg", CV_LOAD_IMAGE_COLOR); 
  recoger.RecogProcess( image );

  return 0;
}
