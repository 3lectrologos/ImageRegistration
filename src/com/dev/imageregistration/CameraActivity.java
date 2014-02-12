package com.dev.imageregistration;

import java.util.ArrayList;
import java.util.List;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.KeyPoint;
import org.opencv.imgproc.Imgproc;

import android.graphics.Bitmap;
import android.hardware.Camera.Size;
import android.os.Bundle;
import android.support.v7.app.ActionBarActivity;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.Toast;

public class CameraActivity extends ActionBarActivity implements
    CvCameraViewListener2 {
  private static final String TAG = "foo";

  private CustomCameraView mOpenCvCameraView;
  private Bitmap mBitmap;
  private Mat mTargetMat;
  private FeatureDetector mDetector;
  private DescriptorExtractor mDescriptor;
  private DescriptorMatcher mMatcher;
  private MatOfKeyPoint mSrcKeypoints;
  private List<Point> mGoodSrcPointsList;
  private List<Point> mGoodTargetPointsList;
  private Mat mSrcDescriptors;
  private MatOfKeyPoint mTargetKeypoints;
  private List<DMatch> mGoodMatchesList;
  private boolean mValidMatches = false;
  private Size mResolution;
  
  private Mat mSrcCorners;
  private Mat mTransformedCorners;

  private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
    @Override
    public void onManagerConnected(int status) {
      switch (status) {
        case LoaderCallbackInterface.SUCCESS:
          mOpenCvCameraView.enableView();
          mDetector = FeatureDetector.create(FeatureDetector.PYRAMID_STAR);
          mDescriptor = DescriptorExtractor.create(DescriptorExtractor.FREAK);
          mMatcher = DescriptorMatcher.create(
              DescriptorMatcher.BRUTEFORCE_HAMMING);
          Mat srcMat = new Mat();
          Utils.bitmapToMat(getBitmap(), srcMat);
          Imgproc.cvtColor(srcMat, srcMat, Imgproc.COLOR_RGB2GRAY);
          mSrcDescriptors = new Mat();
          mSrcKeypoints = new MatOfKeyPoint();
          mDetector.detect(srcMat, mSrcKeypoints);
          mDescriptor.compute(srcMat, mSrcKeypoints, mSrcDescriptors);
          break;
        default:
          super.onManagerConnected(status);
          break;
      }
    }
  };

  public CameraActivity() {
  }
  
  private Bitmap getBitmap() {
    if(mBitmap == null) {
      String filename = "dominos.jpg";
      mBitmap = XUtil.getBitmapFromAsset(this, filename);
    }
    return mBitmap;
  }
  
  @Override
  public void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    requestWindowFeature(Window.FEATURE_NO_TITLE);
    getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                         WindowManager.LayoutParams.FLAG_FULLSCREEN);
    getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    setContentView(R.layout.activity_camera);
    ImageView bgView = (ImageView)findViewById(R.id.bg_image);
    bgView.setImageBitmap(getBitmap());
    
    mOpenCvCameraView = (CustomCameraView)findViewById(R.id.java_surface_view);
    mOpenCvCameraView.setCvCameraViewListener(this);
    mOpenCvCameraView.setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View v) {
      	mValidMatches = false;
        Mat targetDescriptors = new Mat();
        mTargetKeypoints = new MatOfKeyPoint();
        mDetector.detect(mTargetMat, mTargetKeypoints);
        mDescriptor.compute(mTargetMat, mTargetKeypoints, targetDescriptors);
        MatOfDMatch matches = new MatOfDMatch();
        
        if(targetDescriptors.empty()) {
          matchFailed("not enough keypoints");
          return;
        }
        mMatcher.match(targetDescriptors, mSrcDescriptors, matches);
        List<DMatch> matchesList = matches.toList();
        
        float maxDist = 0;
        float minDist = Float.MAX_VALUE;
        for(int i = 0; i < matchesList.size(); i++) {
          float dist = matchesList.get(i).distance;
          if(dist < minDist) minDist = dist;
          if(dist > maxDist) maxDist = dist;
        }

        mGoodMatchesList = new ArrayList<DMatch>();
        List<KeyPoint> srcKeypointsList = mSrcKeypoints.toList();
        List<KeyPoint> targetKeypointsList = mTargetKeypoints.toList();
        mGoodSrcPointsList = new ArrayList<Point>();
        mGoodTargetPointsList = new ArrayList<Point>();
        if(minDist > 30) {
          matchFailed("min. distance too high");
          return;
        }
        float goodDist = 3*minDist;
        for(int i = 0; i < matchesList.size(); i++) {
          DMatch match = matchesList.get(i);
          if(match.distance < goodDist) {
          	mGoodMatchesList.add(match);
          	mGoodSrcPointsList.add(srcKeypointsList.get(match.trainIdx).pt);
            match.trainIdx = mGoodSrcPointsList.size() - 1;
            mGoodTargetPointsList.add(
                targetKeypointsList.get(match.queryIdx).pt);
            match.queryIdx =  mGoodTargetPointsList.size() - 1;
          }
        }
        
        MatOfPoint2f goodSrcPoints = new MatOfPoint2f();
        goodSrcPoints.fromList(mGoodSrcPointsList);
        MatOfPoint2f goodTargetPoints = new MatOfPoint2f();
        goodTargetPoints.fromList(mGoodTargetPointsList);
        if(mGoodSrcPointsList.size() < 8) {
          matchFailed("not enough matches");
          return;
        }
        Mat hom = Calib3d.findHomography(
            goodSrcPoints, goodTargetPoints, Calib3d.RANSAC, 4);
        
        Core.perspectiveTransform(mSrcCorners, mTransformedCorners, hom);
        MatOfPoint pointCorners = new MatOfPoint();
        mTransformedCorners.convertTo(pointCorners, CvType.CV_32S);
        if(!Imgproc.isContourConvex(pointCorners)) {
          matchFailed("non-convex bounding box");
          return;
        }
        
        MatOfPoint2f transformedPoints = new MatOfPoint2f();
        Core.perspectiveTransform(goodSrcPoints, transformedPoints, hom);
        List<Point> transformedPointsList = transformedPoints.toList();
        List<DMatch> toRemove = new ArrayList<DMatch>();
        for(int i = 0; i < mGoodMatchesList.size(); i++) {
          DMatch match = mGoodMatchesList.get(i);
          Point p1 = transformedPointsList.get(match.trainIdx);
          Point p2 = mGoodTargetPointsList.get(match.queryIdx);
          if(dist2(p1, p2) > 4) {
            toRemove.add(match);
          }
        }
        mGoodMatchesList.removeAll(toRemove);
        if(mGoodMatchesList.size() < 8) {
          matchFailed("not enough good matches");
          return;
        }
        mValidMatches = true;

        Toast.makeText(
            CameraActivity.this,
            "Match successful (found " + mGoodMatchesList.size() +
            " good matches)",
            Toast.LENGTH_LONG).show();
      }
    });
  }
  
  private void matchFailed(String reason) {
    if(reason == null || reason.isEmpty()) {
      reason = "";
    } else {
      reason = " (" + reason + ")";
    }
    Toast.makeText(CameraActivity.this,
                   "Matching unsuccessful" + reason,
                   Toast.LENGTH_SHORT)
         .show();
  }
  
  private double dist2(Point p1, Point p2) {
    double x = p1.x - p2.x;
    double y = p1.y - p2.y;
    return Math.sqrt(x*x + y*y);
  }

  @Override
  public void onPause() {
    if (mOpenCvCameraView != null) {
      mOpenCvCameraView.disableView();
    }
    super.onPause();
  }

  @Override
  public void onResume() {
    super.onResume();
    OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3,
                           this,
                           mLoaderCallback);
  }

  @Override
  public void onDestroy() {
    if (mOpenCvCameraView != null) {
      mOpenCvCameraView.disableView();
    }
    super.onDestroy();
  }
  
  public void onCameraViewStarted(int width, int height) {
    mResolution = mOpenCvCameraView.getResolution();
    mSrcCorners = new Mat(4, 1, CvType.CV_32FC2);
    Point p;
    p = new Point(0, 0);
    mSrcCorners.put(0, 0, new double[] {p.x, p.y});
    p = new Point(getBitmap().getWidth(), 0);
    mSrcCorners.put(1, 0, new double[] {p.x, p.y});
    p = new Point(getBitmap().getWidth(), getBitmap().getHeight());
    mSrcCorners.put(2, 0, new double[] {p.x, p.y});
    p = new Point(0, getBitmap().getHeight());
    mSrcCorners.put(3, 0, new double[] {p.x, p.y});
    mTransformedCorners = new Mat(4, 1, CvType.CV_32FC2);
  }

  public void onCameraViewStopped() {
  }
  
  private Point transform(Point pt) {
    double ratio = mResolution.height/(1.0*getBitmap().getHeight());
    double xoffset = 0.5*(mResolution.width - getBitmap().getWidth()*ratio);
    return new Point(xoffset + ratio*pt.x, ratio*pt.y);
  }

  public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
    mTargetMat = inputFrame.gray();
    Mat targetRgb = inputFrame.rgba();
    //MatOfKeyPoint targetKeypoints = new MatOfKeyPoint();
    //mDetector.detect(mTargetMat, targetKeypoints);
    //if(targetKeypoints != null) {
      //for(KeyPoint kp : targetKeypoints.toList()) {
      //  Core.circle(targetRgb, new Point(kp.pt.x, kp.pt.y), 10,
      //     new Scalar(255, 0, 0));
      // }
      //for(KeyPoint kp : mSrcKeypoints.toList()) {
      //  Core.circle(targetRgb, transform(new Point(kp.pt.x, kp.pt.y)), 10,
      //      new Scalar(0, 100, 255));
      //}
      if(mValidMatches) {
        if(mGoodMatchesList != null) {
  	      for(DMatch dm : mGoodMatchesList) {
  	      	Point sp = transform(mGoodSrcPointsList.get(dm.trainIdx));
  	      	Point tp = mGoodTargetPointsList.get(dm.queryIdx);
  	      	Core.circle(targetRgb, sp, 15, new Scalar(0, 255, 0));
  	      	Core.circle(targetRgb, tp, 15, new Scalar(0, 255, 0));
  	      	Core.line(targetRgb, sp, tp, new Scalar(0, 255, 0));
  	      }
  	      Core.line(targetRgb,
  	                new Point(mTransformedCorners.get(0, 0)),
  	                new Point(mTransformedCorners.get(1, 0)),
  	                new Scalar(155, 155, 0), 3);
          Core.line(targetRgb,
              new Point(mTransformedCorners.get(1, 0)),
              new Point(mTransformedCorners.get(2, 0)),
              new Scalar(155, 155, 0), 3);
          Core.line(targetRgb,
              new Point(mTransformedCorners.get(2, 0)),
              new Point(mTransformedCorners.get(3, 0)),
              new Scalar(155, 155, 0), 3);
          Core.line(targetRgb,
              new Point(mTransformedCorners.get(3, 0)),
              new Point(mTransformedCorners.get(0, 0)),
              new Scalar(155, 155, 0), 3);
        }
      }
    //}
    return targetRgb; 
  }
}