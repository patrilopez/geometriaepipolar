#include <stdio.h>
#include <iostream>
#include <opencv2/features2d.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <stdio.h>
#include "opencv2/flann.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/imgcodecs.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
//#include "opencv2/flann/miniflann.hpp"






using namespace cv;
using namespace cv::xfeatures2d;

 static float distancePointLine(const cv::Point_<double> point, const cv::Vec<double,3>& line);
//template <typename T1, typename T2>
int main( int argc, char** argv )
{

	
	
//------------------------------------------------Cargamos las imagenes--------------------------

// img_1 = cv2.imread('myleft.jpg',0)  #queryimage # left image
//img_2 = cv2.imread('myright.jpg',0) #trainimage # right image

	Mat img_1 = imread( argv[1], 1 );
  	Mat img_2 = imread( argv[1], 1 );

//Compruebo si he leido bien las imagenes
	if( !img_1.data || !img_2.data )
	   { return -1; }


//-----------------------Computar las caracteristicas, usamos sift pq es el usado en el tutorial de python, es parecido (creo) al surf de los otros tutoriales--------------------

/*Extract features and computes their descriptors using SIFT algorithm

void SIFT::operator()(InputArray img, InputArray mask, vector<KeyPoint>& keypoints, OutputArray descriptors, bool useProvidedKeypoints=false)
    
Parameters:	

        img – Input 8-bit grayscale image
        mask – Optional input mask that marks the regions where we should detect features.
        keypoints – The input/output vector of keypoints
        descriptors – The output matrix of descriptors. Pass cv::noArray() if you do not need them.
        useProvidedKeypoints – Boolean flag. If it is true, the keypoint detector is not run. Instead, the provided vector of keypoints is used and the algorithm just computes their descriptors.

*/
	/*sift = cv2.SIFT()

	find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img_1,None)
	kp2, des2 = sift.detectAndCompute(img_2,None)
    */

	std::vector<KeyPoint> keypoints_1, keypoints_2;
	Mat descriptors_1, descriptors_2;
	//Ptr<SIFT> detector = cv::xfeatures2d::SIFT::create();
	Ptr<SIFT> detector = SIFT::create(400);
	detector->detectAndCompute( img_1, Mat(), keypoints_1, descriptors_1 );
     	detector->detectAndCompute( img_2, Mat(), keypoints_2, descriptors_2 );

//Hay que emparejar las caracteristicas de las dos imagenes, FLANN( FAST LIBRARY for APPROXIMATE NEAREST NEIGHBORS)

    /*# FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50) //Por ahora esto lo ignoramos
    
   flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)*/

	cv::FlannBasedMatcher matcher; //Creo un objeto de clase flannbasedmatcher llamado matcher, es equivalente a flann en python(?)

	std::vector< DMatch > matches;
	//std::vector<vector<DMatch > > matches;
	//matcher.knnMatchImpl(descriptors_1,matches,2);
	//matcher.knnMatch(descriptors_1,descriptors_2,matches,2);
	matcher.match( descriptors_1, descriptors_2, matches );
	

	 double max_dist = 0; double min_dist = 100;

	 //-- Aquí quiero calcular la distancia maxima y minima entre keypoints
	 for( int i = 0; i < descriptors_1.rows; i++ )
	  	{ double dist = matches[i].distance;
	   		 if( dist < min_dist ) min_dist = dist;
	    		 if( dist > max_dist ) max_dist = dist;
	  	}


	//Ahora creamos un vector para quedarnos solo con las qeu nos parecen adecuadas
	std::vector< DMatch > good_matches;

	for( int i = 0; i < descriptors_1.rows; i++ )
		{ if( matches[i].distance <= 0.8*max_dist)//Creo que en python es esta la condicion
			{ good_matches.push_back( matches[i]); }// si la cumple lo añado a mi vector de puntos buenos
	}

//Una vez tengo los puntos buenos los emparejo

	std::vector<Point2f> obj; //esto seria pts1
	std::vector<Point2f> scene;//esto pts2

	for(int i = 0; i < good_matches.size(); i++)
	{
	    // queryIdx and trainIdx allow us to get the original points
	    // of our good matches by referring back to the keypoints arrays
	    obj.push_back(keypoints_1[good_matches[i].queryIdx].pt);
	    scene.push_back(keypoints_2[good_matches[i].trainIdx].pt);
	}
 //Hay que calcular la matriz fundamental, bucamos en la API la funcion findFundamentalMat 


/*Calculates a fundamental matrix from the corresponding points in two images.

Parameters
    points1	Array of N points from the first image. The point coordinates should be floating-point (single or double precision).
    points2	Array of the second image points of the same size and format as points1 .
    method	Method for computing a fundamental matrix.

        CV_FM_7POINT for a 7-point algorithm. N=7
        CV_FM_8POINT for an 8-point algorithm. N≥8
        CV_FM_RANSAC for the RANSAC algorithm. N≥8
        CV_FM_LMEDS for the LMedS algorithm. N≥8

    param1	Parameter used for RANSAC. It is the maximum distance from a point to an epipolar line in pixels, beyond which the point is considered an outlier and is not used for computing the final fundamental matrix. It can be set to something like 1-3, depending on the accuracy of the point localization, image resolution, and the image noise.
    param2	Parameter used for the RANSAC or LMedS methods only. It specifies a desirable level of confidence (probability) that the estimated matrix is correct.
    mask	The epipolar geometry is described by the following equation: ([p2;1]^T)F[p1;1]=0

where F is a fundamental matrix, p1 and p2 are corresponding points in the first and the second images, respectively.*/

	Mat fundamental_matrix =findFundamentalMat(obj, scene, CV_FM_LMEDS, 3, 0.99); // El 3 y el 0.99 son los que venia de momento los dejo asi
   	/*pts1 = np.int32(pts1)
     	pts2 = np.int32(pts2)
	F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
     	*/


//-------------------------Una vez tengo F tengo todos los parámetros que necesitaba en el otro codigo en c, por tanto cojo sus funciones  

   
/*# We select only inlier points
     pts1 = pts1[mask.ravel()==1]
     pts2 = pts2[mask.ravel()==1]         pts1.append(kp1[m.queryIdx].pt)
def drawlines(img_1,img_2,lines,pts1,pts2):
      ''' img_1 - image on which we draw the epilines for the points in img_2
             lines - corresponding epilines '''
     r,c = img_1.shape
         img_1 = cv2.cvtColor(img_1,cv2.COLOR_GRAY2BGR)
         img_2 = cv2.cvtColor(img_2,cv2.COLOR_GRAY2BGR)
         for r,pt1,pt2 in zip(lines,pts1,pts2):
             color = tuple(np.random.randint(0,255,3).tolist())
             x0,y0 = map(int, [0, -r[2]/r[1] ])
            x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
          img_1 = cv2.line(img_1, (x0,y0), (x1,y1), color,1)
            img_1 = cv2.circle(img_1,tuple(pt1),5,color,-1)
           img_2 = cv2.circle(img_2,tuple(pt2),5,color,-1)
        return img_1,img_2

# drawing its lines on left image
     lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
     lines1 = lines1.reshape(-1,3)
     img5,img6 = drawlines(img_1,img_2,lines1,pts1,pts2)
     
     # Find epilines corresponding to points in left image (first image) and
     # drawing its lines on right image
     lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img_2,img_1,lines2,pts2,pts1)
    }
*/
	//const std::string title;

	//drawEpipolarLines( title, fundamental_matrix, img_1, img_2,obj,scene,-1); //Creo que esta fncion sustituye al codigo restante de python
	const float inlierDistance = -1;
 	CV_Assert(img_1.size() == img_2.size() && img_1.type() == img_2.type());
	  cv::Mat outImg(img_1.rows, img_1.cols*2, CV_8UC3);
	  cv::Rect rect1(0,0, img_1.cols, img_1.rows);
	  cv::Rect rect2(img_1.cols, 0, img_1.cols, img_1.rows);
	  /*
	   * Allow color drawing
	   */
	  if (img_1.type() == CV_8U)
	  {
	    cv::cvtColor(img_1, outImg(rect1), CV_GRAY2BGR);
	    cv::cvtColor(img_2, outImg(rect2), CV_GRAY2BGR);
	  }
	  else
	  {
	    img_1.copyTo(outImg(rect1));
	    img_2.copyTo(outImg(rect2));
	  }
	  std::vector<cv::Vec<double,3> > epilines1, epilines2;
	  cv::computeCorrespondEpilines(obj, 1, fundamental_matrix, epilines1); //Index starts with 1
	  cv::computeCorrespondEpilines(scene, 2,fundamental_matrix, epilines2);
	 
	  CV_Assert(obj.size() == scene.size() &&
		scene.size() == epilines1.size() &&
		epilines1.size() == epilines2.size());
	 
	  cv::RNG rng(0);
	  for(size_t i=0; i<obj.size(); i++)
	  {
	    if(inlierDistance > 0)
	    {
	      if(distancePointLine(obj[i], epilines2[i]) > inlierDistance ||
		distancePointLine(scene[i], epilines1[i]) > inlierDistance)
	      {
		//The point match is no inlier
		continue;
	      }
	    }
	    /*
	     * Epipolar lines of the 1st point set are drawn in the 2nd image and vice-versa
	     */
	    cv::Scalar color(rng(256),rng(256),rng(256));
	 
	    cv::line(outImg(rect2),
	      cv::Point(0,-epilines1[i][2]/epilines1[i][1]),
	      cv::Point(img_1.cols,-(epilines1[i][2]+epilines1[i][0]*img_1.cols)/epilines1[i][1]),
	      color);
	    cv::circle(outImg(rect1), obj[i], 3, color, -1, CV_AA);
	 
	    cv::line(outImg(rect1),
	      cv::Point(0,-epilines2[i][2]/epilines2[i][1]),
	      cv::Point(img_2.cols,-(epilines2[i][2]+epilines2[i][0]*img_2.cols)/epilines2[i][1]),
	      color);
	    cv::circle(outImg(rect2),scene[i], 3, color, -1, CV_AA);
	  }
	  cv::imshow("title", outImg);
	  cv::waitKey(1);

 waitKey(0);
return 0 ;

}


 static float distancePointLine(const cv::Point_<double> point, const cv::Vec<double,3>& line)
{
  //Line is given as a*x + b*y + c = 0
  return fabsf(line(0)*point.x + line(1)*point.y + line(2))
      / std::sqrt(line(0)*line(0)+line(1)*line(1));

}


