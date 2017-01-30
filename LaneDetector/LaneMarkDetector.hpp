#pragma once
#include <opencv2\imgproc.hpp>

class LaneMarkDetector
{
public:
	int tau_ = 10;
	int verticalOffset_ = 440;
	int threshold_ = 160;

	void Process(cv::Mat& img, cv::Mat& out)
	{
		unsigned char *raw = (unsigned char*)(img.data);
		out.setTo(0);

		int aux = 0;
		int x = 0, y = 0;
		const int w = img.cols - tau_ - 1;

		for (y = verticalOffset_; y < img.rows; ++y)
		{
			auto raw = img.ptr(y);
			for (x = tau_; x < w; ++x)
			{
				aux = 2 * raw[x];
				aux -= raw[x - tau_];
				aux -= raw[x + tau_];

				aux -= abs(raw[x - tau_] - raw[x + tau_]);

				aux *= 4;// more contrast

				if (aux >= threshold_)
				{
					out.at<unsigned char>(y, x) = 255;
				}
			}
		}

		cv::erode(out, out, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
		//cv::dilate(out, out, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

		////cv::Mat line = cv::Mat::ones(10, 1, CV_8UC1);
		////now apply the morphology open operation
		cv::Mat line2 = cv::Mat::ones(1, 3, CV_8UC1);
		morphologyEx(out, out, cv::MORPH_OPEN, line2, cv::Point(-1, -1));
	}
};