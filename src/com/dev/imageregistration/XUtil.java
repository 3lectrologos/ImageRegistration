package com.dev.imageregistration;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;

import android.app.ActivityManager;
import android.app.ActivityManager.RunningServiceInfo;
import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.DisplayMetrics;
import android.util.TypedValue;

public class XUtil {
  public static Bitmap getBitmapFromAsset(Context context, String strName) {
    AssetManager assetManager = context.getAssets();
    InputStream istr;
    Bitmap bitmap = null;
    try {
      istr = assetManager.open(strName);
      bitmap = BitmapFactory.decodeStream(istr);
    } catch (IOException e) {
      return null;
    }
    return bitmap;
  }
  
  public static boolean isServiceRunning(Context ctx, Class<?> serviceClass) {
    ActivityManager mgr =
      (ActivityManager)ctx.getSystemService(Context.ACTIVITY_SERVICE);
    for(RunningServiceInfo svc : mgr.getRunningServices(Integer.MAX_VALUE)) {
      if(serviceClass.getName().equals(svc.service.getClassName())) {
        return true;
      }
    }
    return false;
  }
  
  public static Bitmap getBitmapFromFile(Context context, String filename) {
    try {
      InputStream in = context.openFileInput(filename);
      return BitmapFactory.decodeStream(in);
    } catch (FileNotFoundException e) {
      return null;
    }
  }
  
  public static float dp2px(Context ctx, float dp) {
    DisplayMetrics metrics = ctx.getResources().getDisplayMetrics();
    return TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, dp, metrics);
  }
}
