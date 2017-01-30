#pragma once

#include <vector>
#include "LaneModels.hpp"
#include "DBSCAN.hpp"
#include "RANSAC.hpp"

namespace vision
{

class PointsToLanesConverter
{
public:
	std::vector<lane_model::Parabola> Convert(std::vector<cv::Point2f> points)
	{
		std::vector<lane_model::Parabola> model;

		const int EPS = 50;
		const int MIN_DENSITY = 100;
		DBScan dbscan(EPS*EPS, MIN_DENSITY);
		dbscan.fit(points);
		auto& labels = dbscan.GetLabels();

		for (int i = 1; i <= dbscan.GetEstimatedClusterNumber(); ++i)
		{
			decltype(points) oneGroup;
			for (int j = 0; j < points.size(); ++j)
			{
				if (labels[j] == i) oneGroup.emplace_back(points[j]);
			}

			if (oneGroup.size() < 250)
				continue;

			auto parabola = RANSAC_Parabola(RANSAC_ITERATIONS, RANSAC_MODEL_SIZE, static_cast<int>(RANSAC_INLINERS * oneGroup.size()), RANSAC_ERROR_THRESHOLD, oneGroup);
			model.emplace_back(parabola);
		}

		RemoveSimilarOnes(model);

		return model;
	}

	void RemoveSimilarOnes(std::vector<lane_model::Parabola>& input)
	{
		const int MIN_DIFFERENCE_BETWEEN_LANES = 60;
		for (int i = 0; i < input.size(); ++i)
		{
			for (int j = i + 1; j < input.size(); ++j)
			{
				if (MinimumDistance(input[i], input[j], 150, 650) < MIN_DIFFERENCE_BETWEEN_LANES)
				{
					//input[i] = lane_model::Parabola((input[i].a  + input[j].a) / 2, (input[i].b + input[j].b) / 2, (input[i].c + input[j].c) / 2);
					if (abs(input[i].a) < abs(input[j].a)) // remove the more curved one
					{
						input.erase(input.begin() + j);
					}
					else
					{
						input.erase(input.begin() + i);
						break;
					}
				}
			}
		}
	}

private:
	const int RANSAC_ITERATIONS = 100;
	const int RANSAC_MODEL_SIZE = 3;
	const int RANSAC_ERROR_THRESHOLD = 30;
	const double RANSAC_INLINERS = 0.55;

	float MinimumDistance(lane_model::Parabola a, lane_model::Parabola b, const int starty, const int endy, const int dy = 2)
	{
		double min = std::numeric_limits<double> ::max();
		for (int i = starty; i <= endy; i += dy)
		{
			min = std::min(abs(a(i) - b(i)), min);
		}
		return static_cast<float>(min);
	}
};

} // namespace vision