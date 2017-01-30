#pragma once
#include <opencv2\core.hpp>
#include <vector>

#include "LaneModels.hpp"

namespace vision
{
	
	struct CurrentLaneModel
	{
		CurrentLaneModel() : valid(false) {}
		
		lane_model::Parabola left_;
		lane_model::Parabola right_;
		lane_model::Parabola center;

		bool valid = false;
	};

	struct RoadModel
	{
		RoadModel() {}

		std::vector<lane_model::Parabola> lanes_;
		std::vector<lane_model::Parabola> current_lane_;
		CurrentLaneModel current_lane_model_;

		cv::Mat invPerspTransform;
	};

	class LaneDetector
	{
	public:
		LaneDetector();
		RoadModel DetectLane(cv::Mat& inputFrame);
	private:
		cv::Mat FindPixelsThatMayBelongToLane(const cv::Mat& input);
		cv::Mat GetRoadOnlyImage(const cv::Mat& input);
		void ComputePerspectiveTransformationMatrix(const int width, const int height);
		cv::Mat LaneDetector::DownsampleImageByHalf(const cv::Mat& input);
		RoadModel BuildRoadModelFromPoints(const std::vector<cv::Point2f>& points);

		cv::Mat perspectiveTransform;
		cv::Mat invPerspectiveTransform;
	};


}   // namespace vision
