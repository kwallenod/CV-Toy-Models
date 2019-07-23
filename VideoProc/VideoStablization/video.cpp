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

class Trajectory {
public:
    vector<double> xs;
    vector<double> ys;
    vector<double> as;
    int cnt = 0;

    Trajectory() {};
    
    Trajectory(const Trajectory &_T) {
        this->xs.assign(_T.xs.begin(), _T.xs.end());
        this->ys.assign(_T.ys.begin(), _T.ys.end());
        this->as.assign(_T.as.begin(), _T.as.end());
        cnt = int(xs.size());
    };
    
    Trajectory(const vector<TransParam> &_T) {
        double x = 0;
        double y = 0;
        double a = 0;
        
        for (auto t: _T) {
            x += t.dx;
            y += t.dy;
            a += t.da;
            xs.push_back(x);
            ys.push_back(y);
            as.push_back(a);
            cnt++;
        }
    };
    
    void append(const TransParam &_T) {
        double x, y, a;
        if (xs.size() == 0) {
            x = 0;
            y = 0;
            a = 0;
        }
        else {
            int last = int(xs.size()) - 1;
            x = xs.at(last);
            y = ys.at(last);
            a = as.at(last);
        }
        xs.push_back(x + _T.dx);
        ys.push_back(y + _T.dy);
        as.push_back(a + _T.da);
        cnt ++;
    }
    
    void smooth(const int radius=30) {
        vector<double> xsCp(xs);
        vector<double> ysCp(ys);
        vector<double> asCp(as);
        
        const int num = 2 * radius + 1;
        
        for(size_t i=radius; i<xsCp.size()-radius; i++) {
            double x = 0;
            double y = 0;
            double a = 0;
            
            for (int j=-radius; j<radius+1; j++) {
                x += xsCp.at(i+j);
                y += ysCp.at(i+j);
                a += asCp.at(i+j);
            }
            
            double xAvg = x / num;
            double yAvg = y / num;
            double aAvg = a / num;
            
            xs.at(i) = xAvg;
            ys.at(i) = yAvg;
            as.at(i) = aAvg    ;
        }
    }
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
    
    Trajectory tj;
    
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
        TransParam tp(dx, dy, da);
        
        tps.at(i) = tp;
        tj.append(tp);
    }
    
    // smooth trajectory
    Trajectory tjSth(tj);
    tjSth.smooth(50);
    
    // create new transformation matrix using smoothing results
    for (int i=0; i<tj.cnt; i++) {
        double diffx = tjSth.xs.at(i) - tj.xs.at(i);
        double diffy = tjSth.ys.at(i) - tj.ys.at(i);
        double diffa = tjSth.as.at(i) - tj.as.at(i);
        tps.at(i).dx += diffx;
        tps.at(i).dy += diffy;
        tps.at(i).da += diffa;
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
    
    




