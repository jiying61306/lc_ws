/* -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.


* File Name : opencvPCtest.cpp

* Purpose :

* Creation Date : 2018-05-07

* Last Modified : Mon 07 May 2018 11:33:29 PM CST

* Created By :  

*_._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._.*/


#include <opencv2/surface_matching.hpp>
#include <iostream>
#include <opencv2/surface_matching/ppf_helpers.hpp>
#include <opencv2/core/utility.hpp>

int main(int argc, char *argv[])
{
  std::string filename = argv[1];
  cv::Mat pc = cv::ppf_match_3d::loadPLYSimple(filename.c_str(), 1);
  std::cout << pc.size() << std::endl;
  std::cout << pc.cols << std::endl;
  std::cout << pc.rows << std::endl;
  std::cout << pc << std::endl;
  
  return 0;
}
