#include <opencv2\highgui.hpp>
#include "LaneDetector.hpp"
#include <thread>
#include <queue>
#include <mutex>
#include <string>
#include <iostream>

struct Frame
{
	cv::Mat       frame;
	unsigned long seqNo;
};

class SafeQueue
{
public:

	SafeQueue() : counter_(0)
	{

	}

	void push(cv::Mat frame)
	{
		std::lock_guard<std::mutex> lock(mtx_);
		frames_.emplace(frame);
	}

	bool pop(cv::Mat& frame)
	{
		std::lock_guard<std::mutex> lock(mtx_);
		if (!frames_.empty())
		{
			frame = frames_.front();
			frames_.pop();
			return true;
		}
		return false;
	}

private:
	std::queue<cv::Mat> frames_;
	unsigned long      counter_;
	std::mutex             mtx_;
};

SafeQueue queueFrames;
SafeQueue queueOutput;


cv::Mat DrawRoadModel(const vision::RoadModel& roadModel, const cv::Mat& inputFrame)
{
	cv::Mat color{ inputFrame };
	const cv::Vec3b a = { 0, 255, 0 };
	const cv::Vec3b b = { 0, 255, 255 };
	//for (auto& parabola : roadModel.lanes_)
	//{
	//	std::vector<cv::Point2f> parabolaLine;
	//	//for (int y = 140; y < 650; ++y)// color.rows; ++y)
	//	for (int y = 0; y < color.rows; ++y)
	//	{
	//		const auto x = 2 * parabola(y*0.5);
	//		if (x > 0 && x < 1280 && y > 0 && y < 720)
	//			parabolaLine.emplace_back(x, y);
	//	}

	//	if (parabolaLine.size() > 0) cv::perspectiveTransform(parabolaLine, parabolaLine, roadModel.invPerspTransform);
	//	for (auto p : parabolaLine)
	//	{
	//		if (p.x > 0 && p.x < 1280 && p.y > 0 && p.y < 720)
	//			color.at<cv::Vec3b>(p) = b;	
	//	}

	//	//std::cout << parabola.ToString() << std::endl;
	//}

	//for (auto& parabola : roadModel.current_lane_)
	//{
	//	std::vector<cv::Point2f> parabolaLine;
	//	for (int y = 0; y < color.rows; ++y)
	//	{
	//		const auto x = 2 * parabola(y*0.5);
	//		if (x > 1 && x < 1280-1 && y > 0 && y < 720)
	//		{
	//			parabolaLine.emplace_back(x-1, y);
	//			parabolaLine.emplace_back(x, y);
	//			parabolaLine.emplace_back(x+1, y);
	//		}
	//	}

	//	if (parabolaLine.size() > 0) cv::perspectiveTransform(parabolaLine, parabolaLine, roadModel.invPerspTransform);
	//	for (auto p : parabolaLine)
	//	{
	//		if (p.x > 0 && p.x < 1280 && p.y > 0 && p.y < 720)
	//			color.at<cv::Vec3b>(p) = a;
	//	}

	//}

	//std::cout << "----" << std::endl;
	
	return color;
}

#include <atomic>

std::atomic<bool> end = false;

void processFrame()
{
	vision::LaneDetector laneDetector;
	cv::Mat frame;

	unsigned long frameNumber = 0;
	unsigned long avg = 0;
	while (end == false)
	{
		if (queueFrames.pop(frame) == false)
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			continue;
		}

		auto beg = std::chrono::high_resolution_clock::now();

		auto local = frame.clone();
		auto roadModel = laneDetector.DetectLane(local);
		auto output = DrawRoadModel(roadModel, local);
		queueOutput.push(output);

		++frameNumber;
		auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - beg).count();
		avg += diff;
		std::cout << "frame:\t" << frameNumber << "\ttime:\t" << diff  << "\tms\t" << avg / frameNumber << "\tms\t" << 1000 / diff << std::endl;
	}
}


void grabFrames()
{
	cv::VideoCapture videoCapture("F://highway.avi");// F://road.wmv");

	if (videoCapture.isOpened() == false)
		end = true;
	cv::Mat frame;
	while(end == false)
	{
		videoCapture >> frame;
		if (frame.empty())
			end = true;
		else
			queueFrames.push(frame);

		std::this_thread::sleep_for(std::chrono::milliseconds(5));
	}

	end = true;
}

void displayResult()
{
	//cv::VideoWriter writer;
	//writer.open("F://output4.avi", writer.fourcc('M', 'J', 'P', 'G'), 25, cv::Size(1280, 720));

	cv::namedWindow("LaneDetector");

	cv::Mat frame;
	while (end == false)
	{
		if (queueOutput.pop(frame) == false)
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			continue;
		}

		//writer.write(frame);
		cv::imshow("LaneDetector", frame);

		if (cv::waitKey(3) == 27)
		{
			break;
		}
	}
	end = true;
	cv::destroyAllWindows();
}

int main(int argc, char**argv)
{
	std::vector<std::thread> threads;

	threads.emplace_back(displayResult);
	threads.emplace_back(grabFrames);
	threads.emplace_back(processFrame);

	for (auto& t : threads) t.join();

	return 0;
}