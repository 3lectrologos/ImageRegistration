package com.dev.imageregistration;

import org.opencv.android.JavaCameraView;

import android.content.Context;
import android.hardware.Camera.Size;
import android.util.AttributeSet;

public class CustomCameraView extends JavaCameraView {
  
  public CustomCameraView(Context ctx, AttributeSet attrs) {
    super(ctx, attrs);
  }
  
  public Size getResolution() {
    return mCamera.getParameters().getPreviewSize();
  }
}