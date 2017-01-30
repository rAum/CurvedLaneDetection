#pragma once

#include <vector>
#include <opencv2/core.hpp>


class DBScan
{
public:

	const int NOISE = -1;
	const int UNKNOW = 0;

	DBScan(float eps, int minSamples) : eps_(eps), minSamples_(minSamples)
	{

	}

	void fit(std::vector<cv::Point2f> input)
	{
		labels.resize(input.size(), UNKNOW);
		visited.resize(input.size(), false);

		int cluster = UNKNOW;
		for (int i = 0; i < input.size(); ++i)
		{
			if (visited[i]) continue;

			visited[i] = true;
			auto neighbours = regionQuery(input, i);
			if (neighbours.size() < minSamples_)
			{
				labels[i] = NOISE;
			}
			else
			{
				expandCluster(input, i, neighbours, ++cluster);
			}
		}

		estimatedClusterNumber = cluster;
	}

	std::vector<int> regionQuery(std::vector<cv::Point2f> input, int i)
	{
		std::vector<int> neighbours;
		auto p = input[i];
		for (int j = 0; j < input.size(); ++j)
		{
			auto dst = p - input[j];
			dst.y *= 0.5; // more eliptical distance
			if (dst.dot(dst) < eps_)
				neighbours.emplace_back(j);
		}
		return neighbours;
	}

	void expandCluster(std::vector<cv::Point2f> input, int i, std::vector<int> neighbours, const int cluster)
	{
		labels[i] = cluster;

		for (size_t k = 0; k < neighbours.size(); ++k)
		{
			int p = neighbours[k];

			if (!visited[p])
			{
				visited[p] = true;
				auto newNeighbours = regionQuery(input, p);
				if (newNeighbours.size() > minSamples_)
				{
					neighbours.insert(neighbours.end(), newNeighbours.begin(), newNeighbours.end());
				}
			}

			if (labels[p] == UNKNOW)
				labels[p] = cluster;
		}
	}

	const std::vector<int>& GetLabels() const
	{
		return labels;
	}

	const int GetEstimatedClusterNumber() const
	{
		return estimatedClusterNumber;
	}

private:
	const float eps_;
	const int minSamples_;
	int estimatedClusterNumber;
	std::vector<int> labels;
	std::vector<bool> visited;
};