#pragma once
#include "LaneModels.hpp"
#include "random_unique.hpp"

/// Model is valid when it's not degenerated (all 0's) or is too much curved
inline bool IsModelValid(vision::lane_model::Parabola& parabola)
{
	return abs(parabola.a) <= 0.0007 && (parabola.a != 0 && parabola.b != 0 && parabola.c != 0);
}

vision::lane_model::Parabola RANSAC_Parabola(int iterations, int init_samples, int n, double error_threshold, std::vector<cv::Point2f> inputData)
{
	auto best_fit = vision::lane_model::Parabola();
	double best_error = std::numeric_limits<double>::max();
	int consensus_set;

	for (int i = 0; i < iterations; ++i)
	{
		random_unique(inputData.begin(), inputData.end(), init_samples);
		auto model = vision::lane_model::fit(inputData, init_samples);
		auto straight_model = model;
		straight_model.a = 0;

		if (!IsModelValid(model)) continue;

		consensus_set = 0;
		auto model_error = model.a*model.a * 10000; // bias towards straight lines
		auto straight_model_error = 0;
		for (auto p : inputData)
		{
			const auto err = abs(model(p.y) - p.x);
			if (err < error_threshold) {
				consensus_set += 1;
				model_error += err;
				straight_model_error += abs(straight_model(p.y) - p.x);
			}
		}

		if (consensus_set >= n)
		{
			if (straight_model_error < model_error)
			{
				model_error = straight_model_error;
				model = straight_model;
			}

			if (model_error < best_error)
			{
				best_fit = model;
				best_error = model_error;
			}
		}
	}

	return best_fit;
}
