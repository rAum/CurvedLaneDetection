#include "LaneModels.hpp"
#include <opencv2\imgproc.hpp>

namespace vision
{
namespace lane_model
{

Parabola fit(std::vector<cv::Point2f>& points, const int count)
{
	// Sjk = sum of p.x^j * p.y^k
	long double
		s00 = count,
		s01 = 0,
		s10 = 0,
		s11 = 0,
		s21 = 0,
		s20 = 0,
		s30 = 0,
		s40 = 0;
	long double x, xx, y;

	for (int i = 0; i < count; ++i)
	{
		x = points[i].y;  // swap x and y to better model line
		y = points[i].x;
		xx = x * x;

		//s00 += 1; //s00 = input.Count
		s10 += x;
		s20 += xx;
		s30 += xx * x;
		s40 += xx * xx;

		s21 += xx * y;
		s11 += x * y;
		s01 += y;
	}

	// Cramer method:    
	long double s20s00_s10s10 = s20 * s00 - s10 * s10;
	long double s30s00_s10s20 = s30 * s00 - s10 * s20;
	long double s30s10_s20s20 = s30 * s10 - s20 * s20;

	long double D = s40 * s20s00_s10s10 - s30 * s30s00_s10s20 + s20 * s30s10_s20s20;
	if (D == 0)
		return Parabola(0, 0, 0);
	long double Da = s21 * s20s00_s10s10 - s11 * s30s00_s10s20 + s01 * s30s10_s20s20; // <- blad!!!!
	long double Db = s40 * (s11 * s00 - s01 * s10) - s30 * (s21 * s00 - s01 * s20) + s20 * (s21 * s10 - s11 * s20);
	long double Dc = s40 * (s20 * s01 - s10 * s11) - s30 * (s30 * s01 - s10 * s21) + s20 * (s30 * s11 - s20 * s21);

	return Parabola(Da / D, Db / D, Dc / D);
}
}  // namespace lane_models
}  // namespace vision