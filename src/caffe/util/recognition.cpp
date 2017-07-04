#include "caffe/util/recognition.hpp"

void Recognitioner::SetDetectorProtoFile(string proto_file)
{
	detector_proto_file_ = proto_file;
}
void Recognitioner::SetDetectorModelFile(string model_file)
{
	detector_model_file_ = model_file;
}
void Recognitioner::SetDetectorConfigFile(string config_file)
{
	detector_config_file_ = config_file;
}
void Recognitioner::SetSetDetectorParm()
{
	// check weather detector_config_file_ is null
	caffe::FrcnnParam::load_param(detector_config_file_);
}
     
void Recognitioner::SetDigitClassifierProtoFile(string proto_file)
{
	digit_classifier_proto_file_ = proto_file;
}
void Recognitioner::SetDigitClassifierModelFile(string model_file)
{
	digit_classifier_model_file_ = model_file;
}
void Recognitioner::SetDigitClassifierLabelFile(string label_file)
{
	digit_classifier_label_file_ = label_file;
}
        
void Recognitioner::SetCharactersClassifierProtoFile(string proto_file)
{
	characters_classifier_proto_file_ = proto_file;
}
void Recognitioner::SetCharactersClassifierModelFile(string model_file)
{
	characters_classifier_model_file_ = model_file;
}
void Recognitioner::SetCharactersClassifierLabelFile(string label_file)
{
	characters_classifier_label_file_ = label_file;
}
void Recognitioner::DetectorInit()
{
	// check 
	detector_.reset(new Detector(detector_proto_file_, detector_model_file_));
}
void Recognitioner::DigitClassifierInit()
{
	// check 
	digit_classifier_.reset(new Classifier(digit_classifier_proto_file_, digit_classifier_model_file_,digit_classifier_label_file_));
}
void Recognitioner::CharactersClassifierInit()
{
	// check 
	characters_classifier_.reset(new Classifier(characters_classifier_proto_file_, characters_classifier_model_file_,characters_classifier_label_file_));
}

void Recognitioner::DetectionProcess(Mat image)
{
	int max_per_image = 5;
	
    detector_->predict(image, detector_results_);

    float image_thresh = 0;
    if ( max_per_image > 0 ) 
	{
		std::vector<float> image_score ;
		for (size_t obj = 0; obj < detector_results_.size(); obj++) 
		{
			image_score.push_back(detector_results_[obj].confidence) ;
		}
		std::sort(image_score.begin(), image_score.end(), std::greater<float>());
      	if ( max_per_image > image_score.size() ) 
		{
        	if ( image_score.size() > 0 )
          	image_thresh = image_score.back();
        } 
		else 
		{
          	image_thresh = image_score[max_per_image-1];
        }
    }
    std::vector<caffe::BBox<float> > filtered_res;
    for (size_t obj = 0; obj < detector_results_.size(); obj++) 
	{
    	if ( detector_results_[obj].confidence >= image_thresh ) 
		{
        	filtered_res.push_back( detector_results_[obj] );
        }
    }   
    
    detector_results_ = filtered_res;
}
void Recognitioner::SegmentProcess(Mat image)
{
    for (size_t obj = 0; obj < detector_results_.size(); obj++) 
	{
    	cv::Mat crop = image(cv::Rect_<double>(detector_results_[obj][0],detector_results_[obj][1],detector_results_[obj][2]-detector_results_[obj][0],detector_results_[obj][3]-detector_results_[obj][1])); // x,y,w,h
    	cv::Mat crop_out;
    	cv::resize( crop , crop_out , cv::Size(180,60),0,0,CV_INTER_LINEAR );
    	crops_.push_back(crop_out);

    	
    	LicensePlate license_plate_data ;
    	InitLicensePlate( crop_out , license_plate_data );
    	TestColorLicensePlate( license_plate_data );
    	
    	ReSetPosition( license_plate_data );
    	DivisionLicensePlate( license_plate_data );
    	WorkSubImage( license_plate_data );
	    
		segment_results_.push_back( license_plate_data.GetSubImage() );
   }
}
/*
	vector<Mat> crops_;
	vector<caffe::BBox<float> > detector_results_;
	vector<vector<Mat>> segment_results_;
	vector<vector<char>> digit_classifier_results_;
	vector<char> characters_classifier_results_;
*/
void Recognitioner::DigitClassifyProcess()
{
	// loop for multi crops multi pos
	cout<<segment_results_.size()<<endl;
	for (int crop_size=0 ; crop_size<crops_.size() ; ++crop_size )
	{
        
	    cout<<segment_results_[crop_size].size()<<endl;
        std::string res="";
		for(int num=1;num<7;++num)
		{
            cv::Mat curr_img;
            cv::resize( segment_results_[crop_size][num] , curr_img , cv::Size(28,28),0,0,CV_INTER_LINEAR );
            Mat image_data = curr_img;
            /*
            for(int i=0;i<image_data.rows;i++){
                for(int j=0;j<image_data.cols;j++){
                    if(image_data.at<uchar>(i,j)==255)image_data.at<uchar>(i,j)=1;
                }
            }
            */
		    std::vector<Prediction> predictions = digit_classifier_->Classify(curr_img);
		    for (size_t i = 0; i < predictions.size(); ++i) 
			{
                
		    	Prediction p = predictions[i];
                res += p.first;
		    }
		}
        //cout<<res<<endl;
        digit_classifier_results_.push_back(res);
	}
    
}
void Recognitioner::CharactersClassifyProcess()
{
	// loop for multi crops multi pos
	/*
	for (int crop_size=0 ; crop_size<crops_.size() ; ++crop_size )
	{
		std::vector<Prediction> predictions = digit_classifier_->Classify(segment_results_[crop_size][0]);
		for (size_t i = 0; i < predictions.size(); ++i) 
		{
		    Prediction p = predictions[i];
		    digit_classifier_results[crop_size]_.push_back(p.first);
		}
	}
    */
}
void Recognitioner::RecogProcess(Mat image)
{
	crops_.clear();
	detector_results_.clear();
	segment_results_.clear();
	digit_classifier_results_.clear();
	characters_classifier_results_.clear();
	
	DetectionProcess(image);
	SegmentProcess(image);
	CharactersClassifyProcess();
	DigitClassifyProcess();
	
}
vector<Mat> Recognitioner::GetDetectorCrops()
{
	return crops_;
}
vector<caffe::BBox<float> > Recognitioner::GetDetectorResults()
{
	return detector_results_;
}
vector<vector<Mat>> Recognitioner::GetSegmentResults()
{
	return segment_results_;
}
vector<string> Recognitioner::GetDigitClassifierResults()
{
	return digit_classifier_results_;
}
vector<string> Recognitioner::GetCharactersClassifierResults()
{
	return characters_classifier_results_;
}
int getdir (string dir, vector<string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening " << dir << endl;
        return errno;
    }

    while ((dirp = readdir(dp)) != NULL) {
        files.push_back(string(dirp->d_name));
    }
    closedir(dp);
    return 0;
}








