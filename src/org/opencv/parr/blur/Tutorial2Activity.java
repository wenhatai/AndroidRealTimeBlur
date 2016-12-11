package org.opencv.parr.blur;

import android.os.Handler;
import android.view.MotionEvent;
import android.view.View;
import android.widget.SeekBar;
import android.widget.TextView;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.WindowManager;

public class Tutorial2Activity extends Activity implements CvCameraViewListener2 {
    private static final String TAG = "OCVSample::Activity";

    private static final int MAXPROGRESS = 100;
    private Mat mRgba;
    private Mat mYUV;
    private int mCurrnetProgress = 0;
    private SeekBar mSeekBar;
    private TextView mProcessTimeText;
    private Handler mHandler;
    private int mCurrentX;


    private CameraBridgeViewBase mOpenCvCameraView;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");

                    // Load native library after(!) OpenCV initialization
                    System.loadLibrary("blur");

                    mOpenCvCameraView.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    public Tutorial2Activity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        mHandler = new Handler();
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.tutorial2_surface_view);
        mSeekBar = (SeekBar)findViewById(R.id.seekbar);
        mProcessTimeText = (TextView)findViewById(R.id.processtime);
        mSeekBar.setMax(MAXPROGRESS);
        mSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                mCurrnetProgress =  progress;
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {

            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {

            }
        });
        mSeekBar.setProgress(MAXPROGRESS/2);
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.tutorial2_activity_surface_view);
        mOpenCvCameraView.setMaxFrameSize(1000, 1000);
        mOpenCvCameraView.enableFpsMeter();
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                float scaleX = mRgba.width()*1.0f/mOpenCvCameraView.getWidth();
                float scaleY = mRgba.height()*1.0f/mOpenCvCameraView.getHeight();
                if(scaleX > scaleY){
                    mCurrentX = (int) (event.getX()*scaleX);
                }else{
                    int translationX = (int) ((mOpenCvCameraView.getWidth() - mRgba.width()/scaleY)/2);
                    mCurrentX = (int)((event.getX() - translationX)*scaleY);
                    if(mCurrentX < 0){
                        mCurrentX = 0;
                    }else if(mCurrentX > mRgba.width()){
                        mCurrentX = mRgba.width();
                    }
                }
                return true;
            }
        });
    }


    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
    }

    public void onCameraViewStopped() {
        mRgba.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        long startTime = System.currentTimeMillis();
        if(mCurrnetProgress != 0 && mCurrentX != 0){
            mYUV = inputFrame.yuv();
            Blur(mYUV.getNativeObjAddr(), mRgba.getNativeObjAddr(), 0, 0, mCurrentX, mYUV.height()  *2 / 3, mCurrnetProgress);
        }else{
            mRgba = inputFrame.rgba();
        }
        mHandler.post(new Runnable() {
            @Override
            public void run() {
                if(mProcessTimeText != null){
                    mProcessTimeText.setText((System.currentTimeMillis() - startTime)+"ms");
                }
            }
        });
        return mRgba;
    }

    public native void Blur(long matAddYUV, long matAddrRGB, int startX, int startY, int width, int height, int radius);
}
