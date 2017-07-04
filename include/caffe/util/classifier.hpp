
#include <vector>
#include <string>
#include <iostream>
#include <caffe/caffe.hpp>

#include "boost/algorithm/string.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/signal_handler.h"
#include "caffe/util/vis.hpp"
#include "boost/algorithm/string.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/signal_handler.h"
#include "caffe/util/vis.hpp"

using namespace caffe;
using namespace std;
using namespace cv;

typedef std::pair<string, float> Prediction;
class Classifier {
 public:
  Classifier(const string& model_file,const string& trained_file,const string& label_file);
  std::vector<Prediction> Classify(const cv::Mat& img, int N = 1);
 private:
  std::vector<float> Predict(const cv::Mat& img);
  void WrapInputLayer(std::vector<cv::Mat>* input_channels);
  void Preprocess(const cv::Mat& img,std::vector<cv::Mat>* input_channels);
 private:
  std::shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  std::vector<string> labels_;
};
