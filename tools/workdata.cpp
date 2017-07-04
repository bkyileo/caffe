#include <gflags/gflags.h>
#include <glog/logging.h>
#include "boost/algorithm/string.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/signal_handler.h"
#include "caffe/util/vis.hpp"
#include "caffe/util/detector.hpp"
#include "caffe/util/license_plate_division.h"
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

int main(int argc, char** argv){
  LicensePlate license_plate_division;
  string gen="/home/bk/caffe/DigitWord/";
  string newgen="newDigitWord/";
  for(int i=33;i<=33;i++){
      string dir=gen;
      if(i>9)dir+=(i/10+'0');
      dir+=(i%10+'0');
      dir+='/';
      vector<string>files=vector<string>();
      getdir(dir,files);
      cout<<i<<endl;
      for(unsigned int ct=0;ct<files.size();ct++){
          if(files[ct].size()>3){
              string filename=dir+files[ct];
              Mat image=imread(filename);
              InitLicensePlate(image,license_plate_division);
              image=license_plate_division.GetPlateBinaryData();
              string newdir=newgen;
              if(i>9)newdir+=(i/10+'0');
              newdir+=(i%10+'0');
              newdir+='/';
              newdir+=files[ct];
              if(i==0)cout<<newdir<<endl;
              imwrite(newdir,image);
          }
     }
  }
}
