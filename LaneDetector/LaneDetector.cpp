#include <algorithm>
#include <iterator>
#include "LaneDetector.hpp"
#include "LaneModels.hpp"
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include "PointsToLanesConverter.hpp"
#include "LaneMarkDetector.hpp"

namespace vision
{

LaneDetector::LaneDetector()
{

}

void LaneDetector::ComputePerspectiveTransformationMatrix(const int width, const int height)
{
	cv::Point2f src[] =
	{
		{ 0,       470 },
		{ 1280,    470 },
		{ 0,       static_cast<float>(height) },
		{ 1280 - 100,    static_cast<float>(height) }
	};

	const float offset = width / 4;
	const float wo = width - offset;

	cv::Point2f dst[] =
	{
		{ (offset - 800),     0 },
		{ (wo + 900),         0 },
		{ (offset + 120), static_cast<float>(height) },
		{ (wo - 160 ),     static_cast<float>(height) }
	};

	perspectiveTransform    = cv::getPerspectiveTransform(src, dst);
	invPerspectiveTransform = cv::getPerspectiveTransform(dst, src);
}

cv::Mat LaneDetector::GetRoadOnlyImage(const cv::Mat& input)
{
	cv::Mat road(input.clone());
	const int width = input.cols;
	const int height = input.rows;

	static bool inited = false;

	if (!inited)
	{
		ComputePerspectiveTransformationMatrix(width, height);
		inited = true;
	}
	cv::warpPerspective(input, road, perspectiveTransform, { width, height });

	return road;
}

cv::Mat LaneDetector::FindPixelsThatMayBelongToLane(const cv::Mat& input)
{
	cv::Mat grayscale;
	cv::cvtColor(input, grayscale, CV_BGR2GRAY);

	cv::Mat out(cv::Size(grayscale.cols, grayscale.rows), CV_8U);
	LaneMarkDetector laneMarkDetector;
	laneMarkDetector.tau_ = 12;
	laneMarkDetector.verticalOffset_ = 150;
	laneMarkDetector.Process(grayscale, out);

	return out;
}

std::vector<cv::Point2f> ConvertImageToPoints(const cv::Mat& input)
{
	std::vector<cv::Point2f> output;

	unsigned char *raw = (unsigned char*)(input.data);

	int aux = 0;
	int x = 0, y = 0;
	const int w = input.cols;

	for (y = 0; y < input.rows; ++y)
	{
		const auto raw = input.ptr(y);
		for (x = 0; x < w; ++x)
		{	
			if (raw[x])
			{
				output.emplace_back(cv::Point2f(x, y));
			}
		}
	}

	return output;
}

bool InRange(lane_model::Parabola a, const int starty, const int endy, const int xstart, const int xend, const int dy = 2)
{
	for (int y = starty; y <= endy; y += dy)
	{
		const auto x = 2 * a(y * 0.5);
		if (x >= xstart && x <= xend) return true;
	}
	return false;
}

void DetectCurrentLane(RoadModel& roadModel)
{
	std::copy_if(roadModel.lanes_.begin(), roadModel.lanes_.end(), std::back_inserter(roadModel.current_lane_), [](const lane_model::Parabola& a) { return InRange(a, 720 - 210, 720 - 100, 1280/2 - 160, 1280/2 + 200 ); });
}

void BuildCurrentLaneModel(RoadModel& roadModel)
{
	roadModel.current_lane_model_.valid = false;

	if (roadModel.current_lane_.size() > 2 || roadModel.current_lane_.size() < 1)
	{
		return;
	}

	const int CENTER_PROBE = 720 - 200;
	const int CAR_CENTER = 1280 / 2;
	const int MIN_ROAD_WIDTH = 100;
	const int ROAD_WIDTH = 200 / 2;

	if (roadModel.current_lane_.size() == 2)
	{
		auto x0 = 2 * roadModel.current_lane_[0](CENTER_PROBE * 0.5);
		auto x1 = 2 * roadModel.current_lane_[1](CENTER_PROBE * 0.5);

		if (abs(x0 - x1) < MIN_ROAD_WIDTH)
		{
			return;
		}

		if (x1 < x0) std::swap(roadModel.current_lane_[0], roadModel.current_lane_[1]);

		roadModel.current_lane_model_.left_  = roadModel.current_lane_[0];
		roadModel.current_lane_model_.right_ = roadModel.current_lane_[1];

		roadModel.current_lane_model_.center = lane_model::Parabola((roadModel.current_lane_[0].a + roadModel.current_lane_[1].a) * 0.5,
			(roadModel.current_lane_[0].b + roadModel.current_lane_[1].b) * 0.5,
			(roadModel.current_lane_[0].c + roadModel.current_lane_[1].c) * 0.5);

	}
	else
	{
		auto x0 = 2 * roadModel.current_lane_[0](CENTER_PROBE * 0.5);

		if (x0 < CAR_CENTER)
		{
			roadModel.current_lane_model_.left_ = roadModel.current_lane_[0];
			//roadModel.current_lane_model_.right_ = lane_model::Parabola(roadModel.current_lane_[0].a, roadModel.current_lane_[0].b, roadModel.current_lane_[0].c + ROAD_WIDTH);
			roadModel.current_lane_model_.center = lane_model::Parabola(roadModel.current_lane_[0].a, roadModel.current_lane_[0].b, roadModel.current_lane_[0].c + ROAD_WIDTH / 2);
		}
		else
		{
			//roadModel.current_lane_model_.left_ = lane_model::Parabola(roadModel.current_lane_[0].a, roadModel.current_lane_[0].b, roadModel.current_lane_[0].c - ROAD_WIDTH);
			roadModel.current_lane_model_.center = lane_model::Parabola(roadModel.current_lane_[0].a, roadModel.current_lane_[0].b, roadModel.current_lane_[0].c - ROAD_WIDTH / 2);
			roadModel.current_lane_model_.right_ = roadModel.current_lane_[0];
		}
	}


	roadModel.current_lane_model_.valid = true;
}

void DrawParabola(cv::Mat& image, const lane_model::Parabola& parabola, const cv::Vec3b color)
{
	std::vector<cv::Point2f> parabolaLine;

	auto p = lane_model::Parabola(parabola.a * 0.25 * 2, parabola.b * 0.5 * 2, parabola.c * 2); // upscale
	for (int y = 0; y < image.rows; ++y)
	{
		const auto x = p(y);
		if (x > 0 && x < 1280 && y > 100 && y < 700)
		{
			parabolaLine.emplace_back(x - 3, y);
			parabolaLine.emplace_back(x - 2, y);
			parabolaLine.emplace_back(x - 1, y);
			parabolaLine.emplace_back(x, y);
			parabolaLine.emplace_back(x + 1, y);
			parabolaLine.emplace_back(x + 2, y);
			parabolaLine.emplace_back(x + 3, y);
		}
	}

	for (auto p : parabolaLine)
	{
		if (p.x > 0 && p.x < 1280 && p.y > 0 && p.y < 720)
			image.at<cv::Vec3b>(p) = color;
	}
}

RoadModel LaneDetector::BuildRoadModelFromPoints(const std::vector<cv::Point2f>& points)
{
	RoadModel roadModel;

	PointsToLanesConverter pointsToLanesCovnerter;
	roadModel.lanes_ = pointsToLanesCovnerter.Convert(points);
	roadModel.invPerspTransform = invPerspectiveTransform;

	DetectCurrentLane(roadModel);
	BuildCurrentLaneModel(roadModel);

	return roadModel;
}

cv::Mat LaneDetector::DownsampleImageByHalf(const cv::Mat& input)
{
	cv::Mat minified;
	cv::resize(input, minified, cv::Size(input.cols / 2, input.rows / 2));
	return minified;
}

RoadModel LaneDetector::DetectLane(cv::Mat& inputFrame)
{
	auto road        = GetRoadOnlyImage(inputFrame);
	auto lanesPixels = FindPixelsThatMayBelongToLane(road);
	auto points      = ConvertImageToPoints(DownsampleImageByHalf(lanesPixels));
	auto roadModel   = BuildRoadModelFromPoints(points);
	
	/////////////////////////////////////////
	cv::Mat tocolor[] = { lanesPixels, lanesPixels, lanesPixels };
	cv::merge(tocolor, 3, road);


	//cv::rectangle(road, cv::Rect(road.cols / 2 - 100, road.rows - 200, 200, 200), cv::Scalar(255, 0, 0), -1);

	//cv::rectangle(road, cv::Rect(road.cols / 2 - 10, road.rows - 200, 20, 20), cv::Scalar(0, 0, 255), -1);

	//cv::rectangle(road, cv::Rect(road.cols / 2 - 150, road.rows - 200, 400 - 50, 10), cv::Scalar(0, 255, 0), -1);


	//190, 210, 1280 / 2 - 160, 1280 / 2 + 200
	//cv::rectangle(road, cv::Point(1280/2 - 160, 720 - 210), cv::Point(1280/2 + 200, 720 - 100), cv::Scalar(255, 255, 0), -1);

	//DBScan dbscan(50 * 50, 100);
	//dbscan.fit(points);

	//const std::vector<cv::Vec3b> colorPallette = 
	//{
	//	{ 226 ,43, 138 },
	//	{ 0, 0, 255 },
	//	{ 0, 255, 0 },
	//	{ 255, 0, 0 },
	//	{255, 255, 0},
	//	{0, 255, 255},
	//	{255, 0, 255},
	//	{128, 0, 128},
	//	{0, 0, 128},
	//	{0, 100, 0},
	//	{128, 0, 0}
	//};
	//const cv::Vec3b black = { 0, 0, 0 };

	//int i = 0;
	//std::transform(points.begin(), points.end(), points.begin(), [](cv::Point2f & p) { return p * 2; });
	////if (points.size() > 0) cv::perspectiveTransform(points, points, invPerspectiveTransform);
	//const size_t colorsNumber = colorPallette.size();
	//for (auto pt : points)
	//{
	//	int label = dbscan.GetLabels()[i++];
	//	if (label == dbscan.NOISE)
	//		road.at<cv::Vec3b>(pt) = black;
	//	else
	//	{
	//		label--;
	//		road.at<cv::Vec3b>(pt) = colorPallette[label % colorsNumber];
	//	}
	//}

	//for (int y = 0; y < inputFrame.rows; ++y)
	//{
	//	road.at<cv::Vec3b>(y, inputFrame.cols / 2) = { 255, 0, 0 };
	//}

	if (roadModel.current_lane_model_.valid)
	{
		DrawParabola(road, roadModel.current_lane_model_.center, { 0,255,0 });
		DrawParabola(road, roadModel.current_lane_model_.left_, { 255,0,0 });
		DrawParabola(road, roadModel.current_lane_model_.right_, { 0,0,255});
	}

//inputFrame = road;
	cv::Mat warped;
	cv::warpPerspective(road, warped, roadModel.invPerspTransform, inputFrame.size());

	//inputFrame = warped;
	double alpha = 0.5;
	cv::addWeighted(warped, alpha, inputFrame, 1 - alpha, 0, inputFrame);
	///////////////////////////////////////

	return roadModel;
}

}  // namespace vision
