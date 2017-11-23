#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

static void help()
{
    cout << "\nThis program demonstrates the cascade recognizer. Now you can use Haar or LBP features.\n"
            "This classifier can recognize many kinds of rigid objects, once the appropriate classifier is trained.\n"
            "It's most known use is for faces.\n"
            "Usage:\n"
            "./facedetect [--cascade=<cascade_path> this is the primary trained classifier such as frontal face]\n"
            "   [--nested-cascade[=nested_cascade_path this an optional secondary classifier such as eyes]]\n"
            "   [--scale=<image scale greater or equal to 1, try 1.3 for example>]\n"
            "   [--try-flip]\n"
            "   [filename|camera_index]\n\n"
            "see facedetect.cmd for one call:\n"
            "./facedetect --cascade=\"data/haarcascades/haarcascade_frontalface_alt.xml\" --nested-cascade=\"data/haarcascades/haarcascade_eye_tree_eyeglasses.xml\" --scale=1.3\n\n"
            "During execution:\n\tHit any key to quit.\n"
            "\tUsing OpenCV version " << CV_VERSION << "\n" << endl;
}

void detectAndDraw(Mat& img, CascadeClassifier& cascade,
    CascadeClassifier& nestedCascade,
    double scale, bool tryflip);

//void mySuperimpose(Mat& img, Mat& msk_img, int tx, int ty, double th);

string cascadeName;
string nestedCascadeName;

int main(int argc, const char** argv)
{
    VideoCapture capture;
    Mat frame, image;
    string inputName;
    bool tryflip;
    CascadeClassifier cascade, nestedCascade;
    double scale;


    /****************/
    /*
    cv::CommandLineParser parser(argc, argv,
        "{help h||}"
        "{cascade|../../data/haarcascades/haarcascade_frontalface_alt.xml|}"
        "{nested-cascade|../../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml|}"
        "{scale|1|}{try-flip||}{@filename||}");
    */
    cv::CommandLineParser parser(argc, argv,
        "{help h||}"
        "{cascade|data/haarcascade_frontalface_alt.xml|}"
        "{nested-cascade|data/haarcascade_eye_tree_eyeglasses.xml|}"
        "{scale|1|}{try-flip||}{@filename||}");
    /****************/
    if (parser.has("help")) {
        help();
        return 0;
    }
    cascadeName = parser.get<string>("cascade");
    nestedCascadeName = parser.get<string>("nested-cascade");
    scale = parser.get<double>("scale");
    if (scale < 1)
        scale = 1;
    tryflip = parser.has("try-flip");
    inputName = parser.get<string>("@filename");
    if (!parser.check()) {
        parser.printErrors();
        return 0;
    }
    if (!nestedCascade.load(nestedCascadeName))
        cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
    if (!cascade.load(cascadeName)) {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        help();
        return -1;
    }
    if (inputName.empty() || (isdigit(inputName[0]) && inputName.size() == 1)) {
        int camera = inputName.empty() ? 0 : inputName[0] - '0';
        if (!capture.open(camera))
            cout << "Capture from camera #" << camera << " didn't work" << endl;
    } else if (inputName.size()) {
        image = imread(inputName, 1);
        if (image.empty()) {
            if (!capture.open(inputName))
                cout << "Could not read " << inputName << endl;
        }
    } else {
        image = imread("../data/lena.jpg", 1);
        if (image.empty())
            cout << "Couldn't read ../data/lena.jpg" << endl;
    }

    if (capture.isOpened()) {
        cout << "Video capturing has been started ..." << endl;


        for (;;) {
            capture >> frame;
            if (frame.empty())
                break;

            Mat frame1 = frame.clone();
            detectAndDraw(frame1, cascade, nestedCascade, scale, tryflip);
            char c = (char)waitKey(10);
            if (c == 27 || c == 'q' || c == 'Q')
                break;
        }
    } else {
        cout << "Detecting face(s) in " << inputName << endl;
        if (!image.empty()) {
            detectAndDraw(image, cascade, nestedCascade, scale, tryflip);
            waitKey(0);
        } else if (!inputName.empty()) {
            /* assume it is a text file containing the
            list of the image filenames to be processed - one per line */
            FILE* f = fopen(inputName.c_str(), "rt");
            if (f) {
                char buf[1000 + 1];
                while (fgets(buf, 1000, f)) {
                    int len = (int)strlen(buf);
                    while (len > 0 && isspace(buf[len - 1]))
                        len--;
                    buf[len] = '\0';
                    cout << "file " << buf << endl;
                    image = imread(buf, 1);
                    if (!image.empty()) {
                        detectAndDraw(image, cascade, nestedCascade, scale, tryflip);
                        char c = (char)waitKey(0);
                        if (c == 27 || c == 'q' || c == 'Q')
                            break;
                    } else {
                        cerr << "Aw snap, couldn't read image " << buf << endl;
                    }
                }
                fclose(f);
            }
        }
    }

    return 0;
}
/*
void mySuperimpose(Mat& img, Mat& msk_img, int tx, int ty, double th)
{
    for (int l = 0; l < msk_img.cols; l++) {
        for (int m = 0; m < msk_img.rows; m++) {
            int p = m * msk_img.cols * 3 + l * 3;
            int my = (m + ty + 0);
            int mx = (l + tx + 20);
            int q = my * img.cols * 3 + mx * 3;
            if (0 <= mx && mx < img.cols && 0 < my && my < img.rows && emt_img.data[p + 0] * emt_img.data[p + 1] * emt_img.data[p + 2]) {
                img.data[q + 0] = emt_img.data[p + 0];
                img.data[q + 1] = emt_img.data[p + 1];
                img.data[q + 2] = emt_img.data[p + 2];
            }
        }
    }
}
*/
void detectAndDraw(Mat& img, CascadeClassifier& cascade,
    CascadeClassifier& nestedCascade,
    double scale, bool tryflip)
{
    double t = 0;
    vector<Rect> faces, faces2;
    const static Scalar colors[] = {
        Scalar(255, 0, 0),
        Scalar(255, 128, 0),
        Scalar(255, 255, 0),
        Scalar(0, 255, 0),
        Scalar(0, 128, 255),
        Scalar(0, 255, 255),
        Scalar(0, 0, 255),
        Scalar(255, 0, 255)};
    Mat gray, smallImg;

    cvtColor(img, gray, COLOR_BGR2GRAY);
    double fx = 1 / scale;
    resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR);
    equalizeHist(smallImg, smallImg);

    t = (double)getTickCount();
    cascade.detectMultiScale(smallImg, faces,
        1.1, 2, 0
                    //|CASCADE_FIND_BIGGEST_OBJECT
                    //|CASCADE_DO_ROUGH_SEARCH
                    | CASCADE_SCALE_IMAGE,
        Size(30, 30));
    if (tryflip) {
        flip(smallImg, smallImg, 1);
        cascade.detectMultiScale(smallImg, faces2,
            1.1, 2, 0
                        //|CASCADE_FIND_BIGGEST_OBJECT
                        //|CASCADE_DO_ROUGH_SEARCH
                        | CASCADE_SCALE_IMAGE,
            Size(30, 30));
        for (vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); ++r) {
            faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
        }
    }
    t = (double)getTickCount() - t;

    /****************/
    //printf("detection time = %g ms\n", t * 1000 / getTickFrequency());
    /****************/

    /****************/
    //if (faces.size() == 0) {
    //    cv::GaussianBlur(img, img, cv::Size(63, 63), 15, 15);
    //}
    /****************/

    for (size_t i = 0; i < faces.size(); i++) {

        /****************/
        Rect r2 = faces[i];
        Rect r;

        double emt_ratio = 1.3;
        //double emt_ratio = 2.0;
        r.x = r2.x - r2.width * ((emt_ratio - 1.0) / 2.0);
        r.y = r2.y - r2.height * ((emt_ratio - 1.0) / 2.0);
        r.width = r2.width * emt_ratio;
        r.height = r2.height * emt_ratio;
        /*
        double emt_ratio = 1.5;
        r.x = r2.x - r2.width / 4;
        r.y = r2.y - r2.height / 4;
        r.width = r2.width * emt_ratio;
        r.height = r2.height * emt_ratio;
        */
        //rectangle(img, Point(r.x * scale, r.y * scale), Point((r.x + r.width) * scale, (r.y + r.height) * scale), Scalar(0, 0, 0));
        /****************/

        /****************/
        Mat emt_img;
        //if (rand() % 100 > 90) {
            emt_img = cv::imread("image/tlm.png", 1);
        //} else {
            emt_img = cv::imread("image/emt.png", 1);
        //}

        double ratio = (double)r.width / emt_img.cols;
        resize(emt_img, emt_img, Size(), ratio, ratio);
        for (int l = 0; l < emt_img.cols; l++) {
            for (int m = 0; m < emt_img.rows; m++) {
                int p = m * emt_img.cols * 3 + l * 3;
                int my = (m + r.y + r.height / 2 - emt_img.rows / 2);
                int mx = (l + r.x);
                int q = my * img.cols * 3 + mx * 3;
                if (0 <= mx && mx < img.cols && 0 < my && my < img.rows && emt_img.data[p + 0] * emt_img.data[p + 1] * emt_img.data[p + 2]) {
                    img.data[q + 0] = emt_img.data[p + 0];
                    img.data[q + 1] = emt_img.data[p + 1];
                    img.data[q + 2] = emt_img.data[p + 2];
                }
            }
        }

        /****************/

        /****************/
/*        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;
        Scalar color = colors[i % 8];
        int radius;

        double aspect_ratio = (double)r.width / r.height;
        if (0.75 < aspect_ratio && aspect_ratio < 1.3) {
            center.x = cvRound((r.x + r.width * 0.5) * scale);
            center.y = cvRound((r.y + r.height * 0.5) * scale);
            radius = cvRound((r.width + r.height) * 0.25 * scale);
            circle(img, center, radius, color, 3, 8, 0);
        } else
            rectangle(img, cvPoint(cvRound(r.x * scale), cvRound(r.y * scale)),
                cvPoint(cvRound((r.x + r.width - 1) * scale), cvRound((r.y + r.height - 1) * scale)),
                color, 3, 8, 0);

        if (nestedCascade.empty())
            continue;
        smallImgROI = smallImg( r );
        nestedCascade.detectMultiScale( smallImgROI, nestedObjects,
            1.1, 2, 0
            //|CASCADE_FIND_BIGGEST_OBJECT
            //|CASCADE_DO_ROUGH_SEARCH
            //|CASCADE_DO_CANNY_PRUNING
            |CASCADE_SCALE_IMAGE,
            Size(30, 30) );
        for ( size_t j = 0; j < nestedObjects.size(); j++ )
        {
            Rect nr = nestedObjects[j];
            center.x = cvRound((r.x + nr.x + nr.width*0.5)*scale);
            center.y = cvRound((r.y + nr.y + nr.height*0.5)*scale);
            radius = cvRound((nr.width + nr.height)*0.25*scale);
            circle( img, center, radius, color, 3, 8, 0 );
        }
        */
    }
    //cv::cvtColor(img, img,CV_BGR2RGB);

    double tx = 440;
    double ty = 100;
    cv::Mat sagiri_img = cv::imread("image/sagiri.png", 1);
    for (int l = 0; l < sagiri_img.cols; l++) {
        for (int m = 0; m < sagiri_img.rows; m++) {
            int p = m * sagiri_img.cols * 3 + l * 3;
            int my = (m + ty);
            int mx = (l + tx);
            int q = my * img.cols * 3 + mx * 3;
            if (0 <= mx && mx < img.cols && 0 < my && my < img.rows && sagiri_img.data[p + 0] * sagiri_img.data[p + 1] * sagiri_img.data[p + 2]) {
                img.data[q + 0] = sagiri_img.data[p + 0];
                img.data[q + 1] = sagiri_img.data[p + 1];
                img.data[q + 2] = sagiri_img.data[p + 2];
            }
        }
    }

    if (faces.size() == 0) {
        std::cout << "sagiri love" << std::endl;
        cv::Mat emt_img = cv::imread("image/emt-mini.png", 1);
        for (int l = 0; l < emt_img.cols; l++) {
            for (int m = 0; m < emt_img.rows; m++) {
                int p = m * emt_img.cols * 3 + l * 3;
                int my = (m + ty + 0);
                int mx = (l + tx + 20);
                int q = my * img.cols * 3 + mx * 3;
                if (0 <= mx && mx < img.cols && 0 < my && my < img.rows && emt_img.data[p + 0] * emt_img.data[p + 1] * emt_img.data[p + 2]) {
                    img.data[q + 0] = emt_img.data[p + 0];
                    img.data[q + 1] = emt_img.data[p + 1];
                    img.data[q + 2] = emt_img.data[p + 2];
                }
            }
        }
    }
    imshow("result", img);
}
