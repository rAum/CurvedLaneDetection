#pragma once

#include <vector>
#include <string>
#include <opencv2\imgproc.hpp>

namespace vision
{
namespace lane_model
{

struct Parabola
{
	double a, b, c;

	Parabola(double A = 0, double B = 0, double C = 0) : a(A), b(B), c(C)
	{ }

	double value(double x) const
	{
		return ((a * x) + b) * x + c;
	}

	double operator()(double x) const
	{
		return value(x);
	}

	bool IsValid() const
	{
		return !(a == 0 && b == 0 && c == 0);
	}

	std::string ToString() const
	{
		return "f(y) = " + std::to_string(a) + " y^2 + " + std::to_string(b) + " y + " + std::to_string(c);
	}
};  // class Parabola

Parabola fit(std::vector<cv::Point2f>& points,  const int count);


}   // namespace lane_model
}  // namespace vision
