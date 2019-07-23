#include <stdio.h>
#include <math.h>
#include <iostream>

#include <imgproc.hpp>
#include <video.hpp>
#include <highgui.hpp>
#include <calib3d.hpp>


using namespace std;
using namespace cv;

class TransParam {
public:
    double dx;
    double dy;
    double da;
    
    TransParam() {};
    
    TransParam(double _dx, double _dy, double _da)
    : dx(_dx), dy(_dy), da(_da) {};
    
    void getMatrix(Mat &_T) {
        _T.at<double>(0, 0) = cos(da);
        _T.at<double>(0, 1) = -sin(da);
        _T.at<double>(1, 0) = sin(da);
        _T.at<double>(1, 1) = cos(da);
        _T.at<double>(0, 2) = dx;
        _T.at<double>(1, 2) = dy;
    }
};

class TrajParam {
public:
    double x;
    double y;
    double a;
    
    TrajParam () {};
    
    TrajParam (double _x, double _y, double _a) {
        x = _x;
        y = _y;
        a = _a;
    }
    
    TrajParam (const TrajParam& _T) {
        x = _T.x;
        y = _T.y;
        a = _T.a;
    }
    
    TrajParam operator= (const TrajParam& _T) {
        TrajParam outT(_T);
        return outT;
    }
    
    TrajParam operator+ (const TrajParam& _T) {
        TrajParam outT;
        outT.x = x + _T.x;
        outT.y = y + _T.y;
        outT.a = a + _T.a;
        return outT;
    };
    
    TrajParam operator- (const TrajParam& _T) {
        TrajParam outT;
        outT.x = x - _T.x;
        outT.y = y - _T.y;
        outT.a = a - _T.a;
        return outT;
    };
    
    TrajParam operator* (const TrajParam& _T) {
        TrajParam outT;
        outT.x = x * _T.x;
        outT.y = y * _T.y;
        outT.a = a * _T.a;
        return outT;
    };
    
    TrajParam operator/ (const TrajParam& _T) {
        TrajParam outT;
        outT.x = x / _T.x;
        outT.y = y / _T.y;
        outT.a = a / _T.a;
        return outT;
    };
};

int main(int argc, char* argv[]) {
    // read input video
    VideoCapture cap(argv[1]);
    int nFrame = int(cap.get(CAP_PROP_FRAME_COUNT));
    int w = int(cap.get(CAP_PROP_FRAME_WIDTH));
    int h = int(cap.get(CAP_PROP_FRAME_HEIGHT));
    int fps = int(cap.get(CAP_PROP_FPS));
    
    // init output video
    VideoWriter outV(argv[2], VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(2*w, h));
    
    Mat prev, prevGray;
    Mat curr, currGray;
    
    Mat TLast(2, 3, CV_32F);
    
    // init trajectory params
    double x = 0;
    double y = 0;
    double a = 0;
    
    // init kalman filter params;
    TrajParam Xp; // previous state estimate
    TrajParam X; // current state estimate
    TrajParam Pp; // error variance of previous state estimate
    TrajParam P; // error variance of current state estimate
    TrajParam K; // gain
    TrajParam z; // actual measurement
    float qStd = 4e-3; // init value for process error
    float rStd = .25;  // init value for measurement error
    TrajParam Q(qStd, qStd, qStd); // process error
    TrajParam R(rStd, rStd, rStd); // measuremment error
    
    // loop over frames
    vector<TransParam> tps(nFrame-1);
    
    for (int i=0; i<nFrame-1; i++) {
        // get prior frame
        if (i == 0) {
            cap >> curr;
            continue;
        }
        else {
            curr.copyTo(prev);
        }
        cvtColor(prev, prevGray, COLOR_BGR2GRAY);
        
        // get current frame
        cap >> curr;
        cvtColor(curr, currGray, COLOR_BGR2GRAY);
        
        // detect feats
        vector<Point2f> ptsPrev, ptsCurr;
        goodFeaturesToTrack(prevGray, ptsPrev, 200, .01, 30);
        
        // estimation motion
        vector<uchar> status;
        vector<float> err;
        calcOpticalFlowPyrLK(prevGray, currGray, ptsPrev, ptsCurr, status, err);
        
        // remove points that are poorly estimated
        auto pPtsPrev = ptsPrev.begin();
        auto pPtsCurr = ptsCurr.begin();
        for (size_t si=0; si<status.size(); si++) {
            if (status[si]) {
                pPtsCurr++;
                pPtsPrev++;
            }
            else {
                ptsCurr.erase(pPtsCurr);
                ptsPrev.erase(pPtsPrev);
            }
        }
        
        // estimate transform matrix
        Mat T = estimateAffinePartial2D(ptsPrev, ptsCurr);
        if (T.data == NULL) {
            cout << "No transformation estimated" << endl;
            TLast.copyTo(T);
        }
        T.copyTo(TLast);
        
        double dx = T.at<double>(0, 2);
        double dy = T.at<double>(1, 2);
        double da = atan2(T.at<double>(1, 0), T.at<double>(0, 0));
        
        // append to trajectory
        x += dx;
        y += dy;
        a += da;
        
        // create 1d kalman filter
        if (i == 0) {
            TrajParam X(0, 0, 0);
            TrajParam P(0, 0, 0);
        }
        else {
            Xp = X; // save prior
            Pp = P + Q; // save prior
            K = Pp / (Pp + Q); // update kalman gain
            X = Xp + K * (z - Xp); // update state
            P = (TrajParam(1, 1, 1) - K) * Pp; // update error variance of state
        }
        
        // smooth using kalman filter
        double diffX = X.x - x;
        double diffY = X.y - y;
        double diffA = X.a - a;
        dx += diffX;
        dy += diffY;
        da += diffA;
        
        // create transformatin matrix
        TransParam tp(dx, dy, da);
        tps.at(i) = tp;
    }
    
    
    // apply transformation matrix
    cap.set(CAP_PROP_POS_FRAMES, 0);
    Mat frame, frameStab, frameOut;
    Mat transMat(2, 3, CV_64F);
    
    for (int i=0; i<nFrame-1; i++) {
        // read frame
        bool success = cap.read(frame);
        if (!success) break;
        
        if (i == 0) {
            hconcat(frame, frame, frameOut);
            outV.write(frameOut);
            continue;
        }
        
        // get transformation matrix
        tps.at(i).getMatrix(transMat);
        // transform
        warpAffine(frame, frameStab, transMat, frame.size());
        // amplify and crop frame
        Mat TScale = getRotationMatrix2D(Point2f(int(frameStab.cols/2), int(frameStab.rows/2)), 0, 1.04);
        warpAffine(frameStab, frameStab, TScale, frameStab.size());
        // write
        hconcat(frame, frameStab, frameOut);
        outV.write(frameOut);
    }
    
    // save video
    cap.release();
    outV.release();
    
    return 0;
}






