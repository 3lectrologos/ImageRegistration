package com.dev.imageregistration;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
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
import org.opencv.core.Size;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.KeyPoint;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Environment;
import android.support.v7.app.ActionBarActivity;
import android.util.Log;
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
  private MatOfKeyPoint mTargetKeypoints;
  private List<KeyPoint> mSrcKeypointList;
  private List<KeyPoint> mTargetKeypointList;
  private List<Point> mGoodSrcKeypointList;
  private List<Point> mGoodTargetKeypointList;
  private Mat mSrcDescriptors;
  private List<DMatch> mGoodMatchesList;
  private boolean mProcessingTargetMat = false;
  private boolean mValidMatches = false;
  private android.hardware.Camera.Size mResolution;
  
  private Mat mSrcCorners;
  private Mat mTransformedCorners;

  private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
    @Override
    public void onManagerConnected(int status) {
      switch (status) {
        case LoaderCallbackInterface.SUCCESS:
          mOpenCvCameraView.enableView();
          initDetector();
          mDescriptor = DescriptorExtractor.create(DescriptorExtractor.ORB);
          mMatcher = DescriptorMatcher.create(
              DescriptorMatcher.BRUTEFORCE_HAMMING);
          Mat srcMat = new Mat();
          Utils.bitmapToMat(getBitmap(), srcMat);
          Imgproc.cvtColor(srcMat, srcMat, Imgproc.COLOR_RGB2GRAY);
          preprocess(srcMat, srcMat);
          mSrcDescriptors = new Mat();
          mSrcKeypoints = new MatOfKeyPoint();
          mDetector.detect(srcMat, mSrcKeypoints);
          mDescriptor.compute(srcMat, mSrcKeypoints, mSrcDescriptors);
          mSrcKeypointList = mSrcKeypoints.toList();
          break;
        default:
          super.onManagerConnected(status);
          break;
      }
    }
  };
  
  private static final boolean DETECTOR_EXTRA_OPT = true;
  
  private void initDetector() {
    mDetector = FeatureDetector.create(FeatureDetector.ORB);
    if(DETECTOR_EXTRA_OPT) {
      try {
        File outDir = getCacheDir();
        File outFile = File.createTempFile("detectorParams", ".YAML", outDir);
        String str = "%YAML:1.0\n" +
        		"nFeatures: 50000\n";
        OutputStreamWriter writer =
            new OutputStreamWriter(new FileOutputStream(outFile));
        writer.write(str);
        writer.close();
        mDetector.read(outFile.getPath());
      }
      catch(IOException e) {
        Log.e(TAG, "Unable to create detector param file");
      }
    }
  }

  public CameraActivity() {
  }
  
  private Bitmap getBitmap() {
    if(mBitmap == null) {
      String filename = "back.jpg";
      mBitmap = XUtil.getBitmapFromAsset(this, filename);
    }
    return mBitmap;
  }
  
  private Mat getBitmap2() {
    String filename = "test.jpg";
    Bitmap bitmap = XUtil.getBitmapFromAsset(this, filename);
    Mat targetMat = new Mat();
    Utils.bitmapToMat(bitmap, targetMat);
    return targetMat;
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
        mProcessingTargetMat = true;
      	mValidMatches = false;
      	
      	// Extract keypoints from camera preview
      	//mTargetMat = getBitmap2();
      	saveMatAsImage(mTargetMat, "test.jpg");
        Mat targetDescriptors = new Mat();
        mTargetKeypoints = new MatOfKeyPoint();
        mDetector.detect(mTargetMat, mTargetKeypoints);
        mDescriptor.compute(mTargetMat, mTargetKeypoints, targetDescriptors);
        mTargetKeypointList = mTargetKeypoints.toList();
        Log.i("foo", "keypoints = " + mSrcKeypointList.size() + " -- " + mTargetKeypointList.size());
        mProcessingTargetMat = false;
        // Abort if no keypoints found
        if(targetDescriptors.empty()) {
          matchFailed("not enough keypoints");
          return;
        }
        // Compute matches
        List<MatOfDMatch> matches = new ArrayList<MatOfDMatch>();
        mMatcher.knnMatch(targetDescriptors, mSrcDescriptors, matches, 2);
        // Filter matches to keep only "unambiguous" ones
        mGoodMatchesList = new ArrayList<DMatch>();
        mGoodSrcKeypointList = new ArrayList<Point>();
        mGoodTargetKeypointList = new ArrayList<Point>();
        for(MatOfDMatch match : matches) {
        	List<DMatch> dmlist = match.toList();
        	DMatch dm0 = dmlist.get(0);
        	DMatch dm1 = dmlist.get(1);
        	if(dm0.distance < 0.7 * dm1.distance) {
        		mGoodMatchesList.add(dm0);
        		mGoodSrcKeypointList.add(mSrcKeypointList.get(dm0.trainIdx).pt);
        		mGoodTargetKeypointList.add(
        				mTargetKeypointList.get(dm0.queryIdx).pt);
        	}
        }
        Log.i("foo", "Good: " + mGoodMatchesList.size() + " of " + matches.size());
        // Compute homography
        MatOfPoint2f goodSrcPoints = new MatOfPoint2f();
        goodSrcPoints.fromList(mGoodSrcKeypointList);
        MatOfPoint2f goodTargetPoints = new MatOfPoint2f();
        goodTargetPoints.fromList(mGoodTargetKeypointList);
        if(mGoodSrcKeypointList.size() < 8) {
          matchFailed("not enough matches [" + mGoodSrcKeypointList.size() + "]");
          return;
        }
        Mat inlierMask = new Mat();
        Mat hom = Calib3d.findHomography(
            goodSrcPoints, goodTargetPoints, Calib3d.RANSAC, 6.0, inlierMask);
        // Transform base frame rectangle
        Core.perspectiveTransform(mSrcCorners, mTransformedCorners, hom);
        MatOfPoint pointCorners = new MatOfPoint();
        mTransformedCorners.convertTo(pointCorners, CvType.CV_32S);
        //if(!Imgproc.isContourConvex(pointCorners)) {
        //  matchFailed("non-convex bounding box");
        //  return;
        //}
        // Remove outlier matches according to homography
        List<DMatch> toRemove = new ArrayList<DMatch>();
        for(int i = 0; i < mGoodMatchesList.size(); i++) {
        	DMatch match = mGoodMatchesList.get(i);
          if(Math.round(inlierMask.get(i, 0)[0]) == 0) {
            toRemove.add(match);
          }
        }
        int foo = mGoodMatchesList.size();
        mGoodMatchesList.removeAll(toRemove);
        Log.i("foo", "Homography: " + mGoodMatchesList.size() + " out of " + foo);
        if(mGoodMatchesList.size() < 11) {
          matchFailed("not enough good matches [" +
                      mGoodMatchesList.size() + "]");
          mValidMatches = true;
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
  
  private void saveMatAsImage(Mat mat, String filename) {
    File dir = Environment.getExternalStoragePublicDirectory(
        Environment.DIRECTORY_PICTURES);
    File file = new File(dir, filename); 
    Highgui.imwrite(file.toString(), mat);
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
  
  private Point transform(Point pt, int width, int height) {
    double ratio = mResolution.height/(1.0*height);
    double xoffset = 0.5*(mResolution.width - width*ratio);
    return new Point(xoffset + ratio*pt.x, ratio*pt.y);
  }
  
  private void preprocess(Mat src, Mat dst) {
    Imgproc.resize(src, dst, new Size(1280, 720));
    //Imgproc.medianBlur(src, dst, 3);
    //Imgproc.GaussianBlur(src, dst, new Size(9, 9), 2);
  }

  private static final boolean SHOW_KP = false;
  
  public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
    if(!mProcessingTargetMat) {
      mTargetMat = inputFrame.gray();
      preprocess(mTargetMat, mTargetMat);
    }
    Mat targetRgb = inputFrame.rgba();
    if(SHOW_KP) {
      MatOfKeyPoint targetKeypoints = new MatOfKeyPoint();
      mDetector.detect(mTargetMat, targetKeypoints);
      if(targetKeypoints != null) {
        for(KeyPoint kp : targetKeypoints.toList()) {
          Point tp = transform(new Point(kp.pt.x, kp.pt.y), 1280, 720);
          Core.circle(targetRgb,
                      tp,
                      10,
                      new Scalar(255, 0, 0));
        }
        for(KeyPoint kp : mSrcKeypoints.toList()) {
          Point sp = transform(new Point(kp.pt.x, kp.pt.y),
                               getBitmap().getWidth(),
                               getBitmap().getHeight());
          Core.circle(targetRgb,
                      sp,
                      10,
                      new Scalar(0, 100, 255));
        }
      }
    }
    if(mValidMatches) {
      if(mGoodMatchesList != null) {
	      for(DMatch dm : mGoodMatchesList) {
	      	Point sp = transform(mSrcKeypointList.get(dm.trainIdx).pt,
	      	                     getBitmap().getWidth(),
	      	                     getBitmap().getHeight());
	      	Point tp = transform(mTargetKeypointList.get(dm.queryIdx).pt,
	      	                     1280,
	      	                     720);
	      	Core.circle(targetRgb, sp, 15, new Scalar(0, 255, 0));
	      	Core.circle(targetRgb, tp, 15, new Scalar(0, 255, 0));
	      	Core.line(targetRgb, sp, tp, new Scalar(0, 255, 0));
	      }
	      Core.line(targetRgb,
	          transform(new Point(mTransformedCorners.get(0, 0)), 1280, 720),
	          transform(new Point(mTransformedCorners.get(1, 0)), 1280, 720),
	          new Scalar(155, 155, 0), 3);
        Core.line(targetRgb,
            transform(new Point(mTransformedCorners.get(1, 0)), 1280, 720),
            transform(new Point(mTransformedCorners.get(2, 0)), 1280, 720),
            new Scalar(155, 155, 0), 3);
        Core.line(targetRgb,
            transform(new Point(mTransformedCorners.get(2, 0)), 1280, 720),
            transform(new Point(mTransformedCorners.get(3, 0)), 1280, 720),
            new Scalar(155, 155, 0), 3);
        Core.line(targetRgb,
            transform(new Point(mTransformedCorners.get(3, 0)), 1280, 720),
            transform(new Point(mTransformedCorners.get(0, 0)), 1280, 720),
            new Scalar(155, 155, 0), 3);
      }
    }
    return targetRgb;
  }
}