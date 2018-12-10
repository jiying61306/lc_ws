/* -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.


* File Name : LoopClosing.cpp

* Purpose :

* Creation Date : 2018-05-07

* Last Modified : Wed 16 May 2018 05:24:05 AM CST

* Created By :  

*_._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._.*/

#include <ros/ros.h>

#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PolygonStamped.h>
#include <sensor_msgs/Image.h>

#include <cv_bridge/cv_bridge.h>

#include <DBoW2/DBoW2.h>

#include <vector>
#include <queue>

#include <mutex>

#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

#include <boost/thread.hpp>

#include <tf/tf.h>

#include <eigen_conversions/eigen_msg.h>
#include <opencv2/surface_matching.hpp>
#include <Eigen/Dense>

#include <cmath>
#include <message_filters/synchronizer.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/transforms.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/registration/transformation_estimation_svd.h>

#include <fstream>
std::mutex mMutex;
using namespace std;
using namespace DBoW2;

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> Cloud;
typedef Cloud::Ptr CloudPtr;
typedef Cloud::ConstPtr CloudConstPtr;

const double a_threshold = 0.5; //similiarity threshold
const int num_threshold = 10; //similiarity threshold
const double ransac_thresh = 2.5f;
const double nn_match_ratio = 0.8f;
bool updateOut = false;
bool updateOut1 = false;
OrbVocabulary* voc;
OrbDatabase* db;
namespace {
  struct Edge {
    geometry_msgs::Pose RT;
    unsigned int src;
    unsigned int dst;
  };
}
cv::Mat K;
int focal_length;
cv::Point2d principal_point;
std::vector<cv::Mat> keyframes;
std::vector<cv::Mat> keydepth;
std::vector<cv::Mat> keyconf;
std::vector<std::vector<cv::Mat>> features;
std::vector<cv::Mat> descs;
std::vector<std::vector<cv::KeyPoint>> keypoints;
// std::vector<CloudPtr> keypoint3D;
std::queue<::Edge> qROSPoseMsgs;
cv::Mat out, out1;
cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");


// CloudPtr Mat2Cloud(cv::Mat& depth)
// {
//   CloudPtr cloud; cloud.reset(new Cloud);
//   int imgsize = depth.cols*depth.rows;
//   cloud->points.resize(imgsize);
//   auto fptr = (float*) depth.data;
//   auto end = fptr + imgsize;
//   for (int i = 0; i < depth.rows; ++i) {
//     for (int j = 0; j < depth.cols; ++j) {
//       PointT pt;
//       float d = *fptr++;
//       if (d!=0 && d==d) {
//         pt.x = pt.y = pt.z = std::numeric_limits<float>::quiet_NaN();
//       } else {
//         pt.z = d;
//         pt.x = (j-principal_point.x)* d/ focal_length;
//         pt.y = (i-principal_point.y)* d/ focal_length;
//       }
//     }
//   }
//   cloud->width = depth.cols;
//   cloud->height = depth.rows;
//   return cloud;
// }
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
  {
    out[i] = plain.row(i);
  }
}


geometry_msgs::Pose getMsgFromRT(const cv::Mat& R, const cv::Mat& t)
{
  Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      m(i,j) = R.at<double>(i,j);
    }
  }
  for (int i = 0; i < 3; ++i) {
    m(i, 3) = t.at<double>(i);
  }
  std::cout << "eigen pose" << std::endl;
  std::cout << m << std::endl;
  Eigen::Isometry3d tmpE(m);
  geometry_msgs::Pose msg;
  tf::poseEigenToMsg(tmpE, msg);
  return msg;
}
geometry_msgs::Pose getMsgFromMatx44(const cv::Matx44d & pose)
{
  Eigen::Matrix4d m;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      m(i,j) = pose(i,j);
    }
  }
  std::cout << "eigen pose" << std::endl;
  std::cout << m << std::endl;
  Eigen::Isometry3d tmpE(m);
  geometry_msgs::Pose msg;
  tf::poseEigenToMsg(tmpE, msg);
  return msg;
}
geometry_msgs::Pose getPoseFromOpencvRT(const cv::Mat& R, const cv::Mat& translation){
  float* fv = (float*)R.data;
  tf::Matrix3x3 mr(*fv, *(fv+1), *(fv+2), *(fv+3), *(fv+4), *(fv+5), *(fv+6), *(fv+7), *(fv+8));
  tf::Quaternion q;
  mr.getRotation(q);
  q.normalized();
  fv = (float*) translation.data;
  tf::Transform bt(q, tf::Vector3(*fv, *(fv+1), *(fv+2)));
  geometry_msgs::Pose tmp;
  tf::poseTFToMsg(bt, tmp);
  return tmp;
}

void updateFeatures(cv::Mat& keyframe)
{
  
  cv::Ptr<cv::ORB> orb = cv::ORB::create();
  cv::Mat mask;
  std::vector<cv::KeyPoint> kps;
  cv::Mat descriptors;

  orb->detectAndCompute(keyframe, mask, kps, descriptors);

  keypoints.push_back(kps);
  features.push_back(vector<cv::Mat >());
  descs.push_back(descriptors);
  changeStructure(descriptors, features.back());

}

void getMatches(const std::vector<cv::Mat>& bows, std::vector<unsigned int>& matches )
{
  QueryResults ret;
  db->query(bows, ret, 10);
  std::cout << "QueryResults size:" << ret.size() << std::endl;
  for (auto &i : ret ) {
    if(i.Score > a_threshold) matches.push_back(i.Id);
  }
  
}


void get3Dpoint( std::vector<cv::Point3f>& point3D, const cv::Mat& depth, std::vector<cv::Point2f>& point2Dsrc, std::vector<cv::Point2f>& point2Ddst ) {
  std::vector<cv::Point2f> newSrc, newDst;
  for (int i = 0; i < point2Dsrc.size(); ++i) {
    auto& pt2D = point2Dsrc[i];
    cv::Point3f pt3D;
    float d = depth.at<float>(pt2D);
    if (std::isfinite(d)) {
      pt3D.x = (pt2D.x - principal_point.x)*d/ focal_length;
      pt3D.y = (pt2D.y - principal_point.y)*d/ focal_length;
      pt3D.z = d;
      point3D.push_back( pt3D);
      newSrc.push_back(pt2D);
      newDst.push_back(point2Ddst[i]);
    } else {
      std::cout << "nan" << std::endl;
    }
  }
  point2Dsrc = newSrc;
  point2Ddst = newDst;
}

pcl::PointCloud<pcl::PointNormal>::Ptr cal_normal(CloudPtr keysrc, CloudPtr srcCloud, cv::Mat& d) {
  int width = d.cols;
  int height = d.rows;
  pcl::PointCloud<pcl::PointNormal>::Ptr srcn(new pcl::PointCloud<pcl::PointNormal>);
	//PointCloud<PointXYZRGB> ref;
	int xpos, ypos, xpos_, ypos_;
	Eigen::Vector4f plane_param;
	float curvature;
  pcl::PointXYZ pr;
	float check;
	for (size_t i = 0; i<keysrc->points.size(); i++){
		//ref.reset(new pcl::PointCloud<PointXYZRGB>);
		Cloud ref;
		PointT p = keysrc->points[i];
    pcl::PointNormal pn;
		xpos = int(floor((focal_length / p.z * p.x) + principal_point.x)); //warped image coordinate x
      ypos = int(floor((focal_length / p.z * p.y) + principal_point.y)); //warped image coordinate y
      for (int xx = -6; xx < 7; xx += 2){
        for (int yy = -6; yy < 7; yy += 2){
        	xpos_ = xpos+xx;
        	ypos_ = ypos+yy;
        	if (xpos_ >= (width) || ypos_ >= (height) || xpos_<0 || ypos_<0) { continue; }
      		pr = srcCloud->points[(ypos_) * width + (xpos_)];
      		check = pr.z - p.z;
      	if(pr.z != pr.z) continue;
      	if((fabs(check))>0.03) continue;
      		ref.push_back(pr);
      		//cout << "p.z: " << p.z << " pr.z: "<< pr.z << endl;
      }}
      pcl::computePointNormal(ref, plane_param, curvature);
      pcl::flipNormalTowardsViewpoint(p, 0, 0, 0, plane_param);
      pn.x = p.x;
      pn.y = p.y;
      pn.z = p.z;
      pn.normal_x = plane_param[0];
      pn.normal_y = plane_param[1];
      pn.normal_z = plane_param[2];
      srcn->points.push_back(pn);
		}
		/*pcl::visualization::PCLVisualizer show;
      pcl::visualization::PointCloudColorHandlerCustom<PointXYZRGB> srcc(m_src, 255, 0, 0);
      show.addPointCloud<PointXYZRGB>(m_src, srcc, "cloud1");
      show.addPointCloudNormals<pcl::PointXYZRGB, pcl::PointNormal> (cloud1, srcn, 10, 0.05, "normals");
      show.spin();*/
		return srcn;
  
}

CloudPtr getCloud(const cv::Mat& depth)
{
  CloudPtr cloud(new Cloud);
  cloud->points.reserve(depth.rows*depth.cols);
  float* fd = (float*)depth.data;
  for (int i = 0; i < depth.rows; ++i) {
    for (int j = 0; j < depth.cols; ++j) {
      PointT pt;   
      float d = *fd++;
      if (std::isfinite(d)) {
        pt.z = d;
        pt.x = (j - principal_point.x) *d / focal_length;
        pt.y = (i - principal_point.y) *d / focal_length;
        cloud->points.push_back(pt);
      } else {
        std::cout << "nan" << std::endl;
      }
    }
  }
  return cloud;
}
void getCloudfromKeypoint(const cv::Mat& srcconf, const cv::Mat& dstconf, const cv::Mat& srcDepth, const cv::Mat& dstDepth, const std::vector<cv::Point2f>&  point2Dsrc, const std::vector<cv::Point2f>&  point2Ddst, CloudPtr& srcPCL, CloudPtr& dstPCL)
{
  srcPCL.reset(new Cloud);
  dstPCL.reset(new Cloud);
  for (int i = 0; i < point2Dsrc.size(); ++i) {
    auto& s2D = point2Dsrc[i];
    auto& d2D = point2Ddst[i];
    auto& sconf = srcconf.at<float>(s2D);
    auto& dconf = dstconf.at<float>(d2D);
    float src_d = srcDepth.at<float>(s2D);
    float dst_d = dstDepth.at<float>(d2D);
    if (std::isfinite(src_d)&& std::isfinite(dst_d) && sconf < 80 && dconf < 80) {
      PointT sd, dd;
      sd.x = (s2D.x - principal_point.x) * src_d / focal_length;
      sd.y = (s2D.y - principal_point.y) * src_d / focal_length;
      sd.z = src_d;
      dd.x = (d2D.x - principal_point.x) * dst_d / focal_length;
      dd.y = (d2D.y - principal_point.y) * dst_d / focal_length;
      dd.z = dst_d;
      srcPCL->points.push_back(sd);
      dstPCL->points.push_back(dd);
    } 
  }
}

void getpcmat(cv::Mat& pc, const cv::Mat& depth, const std::vector<cv::Point2f>& point2D, cv::OutputArray mask = cv::noArray()) {
  pc = cv::Mat(point2D.size(), 3, CV_32FC1);
  auto pcptr = (float*) pc.data;
  //pc format x y z
  if(!mask.empty()){
    auto m = mask.getMat();
    for (auto &pt : point2D) {
      if(m.at<char>(pt)){
        float d = depth.at<float>(pt);
        if(d == d){
          *pcptr++ = (pt.x - principal_point.x) * d / focal_length;
          *pcptr++ = (pt.y - principal_point.y) * d / focal_length;
          *pcptr++ = d;
        }
        else
          std::cout << "nan" << std::endl;
      }
    }
  }
}

int imgIndex = 0;

bool getRT(int src, int dst, geometry_msgs::Pose& RT) {
  if(src-20 < dst){ 
    return false;
  }
  auto& kp_src = keypoints[src];
  auto& kp_dst = keypoints[dst];
  auto& desc_src = descs[src];
  auto& desc_dst = descs[dst];
  std::vector< std::vector<cv::DMatch>> matches;
  std::vector<cv::KeyPoint> matches_src, matches_dst;
  matcher->knnMatch(desc_src, desc_dst, matches, 2);
  for(unsigned i = 0; i < matches.size(); i++) {
    if(matches[i][0].distance < nn_match_ratio * matches[i][1].distance) {
      matches_src.push_back(kp_src[matches[i][0].queryIdx]);
      matches_dst.push_back(kp_dst[matches[i][0].trainIdx]);
    }
  }

  cv::Mat inlier_mask, homography;
  cv::Mat inlier_mask_, fund_, E;
  vector<cv::KeyPoint> inliers1, inliers2;
  vector<cv::KeyPoint> inliers1_, inliers2_;
  vector<cv::DMatch> inlier_matches, inlier_matches_;

  std::vector<cv::Point2f> msrc, mdst; 
  if(matches_src.size() >= 4) {
    for (int i = 0; i < matches_src.size(); ++i) {
      msrc.push_back(matches_src[i].pt);
      mdst.push_back(matches_dst[i].pt);
    }
    homography = findHomography((msrc), (mdst), cv::RANSAC, ransac_thresh, inlier_mask);
    // fund_ = findFundamentalMat((msrc), (mdst), cv::FM_RANSAC, 3, .99, inlier_mask_);
    // E = findEssentialMat(msrc, mdst, focal_length, principal_point, cv::RANSAC, 0.999, 1.0, inlier_mask_);
    // recoverPose(E, msrc, mdst, R, translation, focal_length, principal_point);
    // cv::decomposeEssentialMat(E, R, R, translation);
    
    // return true;
  }
  else {
    return false;
  }
  msrc.clear(); mdst.clear();
  for (int i = 0; i < matches_src.size(); ++i) {
    if (inlier_mask.at<uchar>(i)) {
      int new_i = static_cast<int>(inliers1.size());
      inliers1.push_back(matches_src[i]);
      msrc.push_back(matches_src[i].pt);
      inliers2.push_back(matches_dst[i]);
      mdst.push_back(matches_dst[i].pt);
      inlier_matches.push_back(cv::DMatch(new_i, new_i, 0));
    }
  }

  // auto m_srcCloud = getCloud(keydepth[src]);
  CloudPtr key_src_cloud(new Cloud), key_dst_cloud(new Cloud);
  getCloudfromKeypoint(keyconf[src], keyconf[dst], keydepth[src], keydepth[dst], msrc, mdst, key_src_cloud, key_dst_cloud);
  // auto src_N = cal_normal(key_src_cloud, m_srcCloud, keydepth[0]);
  pcl::registration::TransformationEstimationSVD<PointT, PointT, float> est;
  Eigen::Matrix4f transform;
  est.estimateRigidTransformation(*key_src_cloud, *key_dst_cloud, transform);
  std::cout << "cooresp size : " << key_src_cloud->size() << std::endl;
  Eigen::Affine3f ti;
  ti.matrix() = transform;
  tf::poseEigenToMsg(ti.cast<double>(), RT);
  cv::Mat tmpOut;
  cv::drawMatches(keyframes[src], inliers1, keyframes[dst], inliers2, inlier_matches, tmpOut, cv::Scalar(255,0,0), cv::Scalar(255,0,0));
  if(updateOut1) {
    out1 = tmpOut;
    updateOut = true;
  }
  return true;
 
}

void updateFeture_cb(const sensor_msgs::ImageConstPtr& rgb, const sensor_msgs::ImageConstPtr& depth, const sensor_msgs::ImageConstPtr& conf)
{

  std::cout << "update:" << std::endl;
  cv_bridge::CvImagePtr cv_ptr_rgb = cv_bridge::toCvCopy(rgb, "rgb8");
  cv_bridge::CvImagePtr cv_ptr_depth = cv_bridge::toCvCopy(depth, sensor_msgs::image_encodings::TYPE_32FC1);
  cv_bridge::CvImagePtr cv_ptr_conf = cv_bridge::toCvCopy(conf, sensor_msgs::image_encodings::TYPE_32FC1);
  updateFeatures(cv_ptr_rgb->image); //update db and feature and keypoint
  keyframes.push_back(cv_ptr_rgb->image);
  keydepth.push_back(cv_ptr_depth->image);
  keyconf.push_back(cv_ptr_conf->image);
  
  if(features.size() < num_threshold) {
    db->add(features.back());
    return;
  }
  std::vector<unsigned int> matches;

  getMatches(features.back(), matches); //matching images
  db->add(features.back());
  int src = features.size()-1;
  for (int i = 0; i < matches.size(); ++i) {
    geometry_msgs::Pose RT;
    if (i == matches.size() - 1 && (((int)src) - 20) >=(int)matches.at(i)) {
      updateOut1 = true;
      std::cout << "print the " << i << "-th " << std::endl;
    }
    else {
      updateOut1 = false;
    }
    if (getRT(src, (int)matches.at(i), RT)) {
      ::Edge tmp = {RT, src, matches.at(i)};
      // unique_lock<std::mutex> lock(mMutex);
      mMutex.lock();
      qROSPoseMsgs.push(tmp);
      mMutex.unlock();
    }
  }
  
}

void publishLoopConstraints()
{
  ros::NodeHandlePtr n = boost::make_shared<ros::NodeHandle>();
  ros::Publisher RT_pub = n->advertise<geometry_msgs::PoseArray>("/loop_closing/edge_constraints", 1);
  ros::Publisher index_pub= n->advertise<geometry_msgs::PolygonStamped>("/loop_closing/vertex_indice", 1); 
  geometry_msgs::PoseArray _poseArray;
  geometry_msgs::PolygonStamped _ToFrom;
  _poseArray.header.frame_id = "/loop_closing";
  while (ros::ok()) {
    if(features.size()<num_threshold || qROSPoseMsgs.empty()) {
      usleep(5000);
      continue;
    }
    (mMutex).lock();
    ros::Time t = ros::Time::now();
    _poseArray.header.stamp = t;
    _ToFrom.header = _poseArray.header;
    _ToFrom.polygon.points.clear();
    _ToFrom.polygon.points.reserve(qROSPoseMsgs.size());
    _poseArray.poses.clear();
    _poseArray.poses.reserve(qROSPoseMsgs.size());
    while (!qROSPoseMsgs.empty()) {
      auto tmp = qROSPoseMsgs.front();
      _poseArray.poses.push_back(tmp.RT);
      geometry_msgs::Point32 point;
      point.x = tmp.src;
      point.y = tmp.dst;
      _ToFrom.polygon.points.push_back(point);
      // if(tmp.src-1 == tmp.dst){
      //   std::cout << "src id: " << tmp.src << std::endl;
      //   std::cout << "dst id: " << tmp.dst << std::endl;
      //   std::cout << "rt: " << kk tmp.RT << std::endl;
      // }
      qROSPoseMsgs.pop();
      std::cout << "src id: " << tmp.src << std::endl;
      std::cout << "dst id: " << tmp.dst << std::endl;
      std::cout << "rt: " << tmp.RT << std::endl;
    }
    mMutex.unlock();
    RT_pub.publish(_poseArray);
    index_pub.publish(_ToFrom);
    usleep(5000);
  }
}
void drawout()
{
  ros::NodeHandlePtr n = boost::make_shared<ros::NodeHandle>();
  // cv::namedWindow("homography");
  cv::namedWindow("fundamental");
  while (ros::ok()) {
    if (updateOut ) {
      // cv::imshow("homography", out);
      cv::imshow("fundamental", out1);
      if(27 == cv::waitKey(33)){
        break;
      }
    }
  }
  cv::destroyAllWindows();
  
}

int main(int argc, char *argv[])
{
  ros::init(argc, argv, "loop_closure"); 
  float data[] = {338.8659973144531, 0.0, 356.92974853515625, 0.0, 338.8659973144531, 206.3419647216797, 0.0, 0.0, 1.0};
  focal_length = 339;
  principal_point = cv::Point2d(357, 206);
  K = cv::Mat(3, 3, CV_32FC1, &data);
  
  /* TODO
   * 1. init
   *    a. load voc and db
   *
   * 2. get key frame topic
   * 3. get ORB feature vector (vector<cv::Mat>)
   * 4. get BoWV and insert into db
   * 5. query matching image
   * 6. get direct index table and compute the correspondence
   * 7. using opencv pose estimation
   *
   */

  // const int k = 9;
  // const int L = 3;
  // const WeightingType weight = TF_IDF;
  // const ScoringType score = L1_NORM;

  boost::thread  threadPub(&publishLoopConstraints);
  boost::thread  threadDraw(&drawout);
  ros::NodeHandlePtr n = boost::make_shared<ros::NodeHandle>();
  std::string dbfile = argv[1];
  // // OrbVocabulary v(k, L, weight, score);
  // v.load(argv[1]);
  OrbVocabulary v(argv[1]);
  OrbDatabase d(v, false, 0);
  
  voc = &v;
  db = &d;


  int buffersize;
  ros::param::get("/buffersize", buffersize);
  message_filters::Subscriber<sensor_msgs::Image> sub_rgb(*n,  "/icp/key_rgb"    , buffersize);
  message_filters::Subscriber<sensor_msgs::Image> sub_depth(*n,"/icp/key_depth"  , buffersize);
  message_filters::Subscriber<sensor_msgs::Image> sub_conf(*n,"/icp/key_conf"  , buffersize);
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,sensor_msgs::Image, sensor_msgs::Image> RGBDepthSyncPolicy;
  message_filters::Synchronizer<RGBDepthSyncPolicy> sync(RGBDepthSyncPolicy(10), sub_rgb, sub_depth, sub_conf);
  sync.registerCallback(boost::bind(&updateFeture_cb, _1, _2, _3));

  ros::MultiThreadedSpinner spinner(3);
  spinner.spin();

  return 0;
}
