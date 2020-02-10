package distance

/*
This module provides common distance functions for measuring distance
between observations.

Minkowski Distance is one the most inclusive among one as other distances are only
a specific case of Minkowski Distance(Chebyshev Distance is not straightforward, though).

when p=1 in MinkowskiDistance, it becomes ManhattanDistance,
when p=2 in MinkowskiDistance, it becomes EuclideanDIstance,
when p goes infinity, it becomes ChebyshevDistance.

Since the ManhattanDistance and EuclideanDistance are very frequently used, they are
implemented separately.
*/

import (
	"math"
)

// LPNorm of an array, given p >= 1
func LPNorm(vector []float64, p float64) (float64, error) {
	distance := 0.
	for _, jj := range vector {
		distance += math.Pow(math.Abs(jj), p)
	}
	return math.Pow(distance, 1/p), nil
}

// DistanceManhattan 1-norm distance (l_1 distance)
func Manhattan(firstVector, secondVector []float64) (float64, error) {
	distance := 0.
	for ii := range firstVector {
		distance += math.Abs(firstVector[ii] - secondVector[ii])
	}
	return distance, nil
}

// Euclidean 2-norm distance (l_2 distance)
func Euclidean(firstVector, secondVector []float64) (float64, error) {
	distance := 0.
	for ii := range firstVector {
		distance += (firstVector[ii] - secondVector[ii]) * (firstVector[ii] - secondVector[ii])
	}
	return math.Sqrt(distance), nil
}

// SquaredEuclidean Higher weight for the points that are far apart
// Not a real metric as it does not obey triangle inequality
func SquaredEuclidean(firstVector, secondVector []float64) (float64, error) {
	distance, err := Euclidean(firstVector, secondVector)
	return distance * distance, err
}

// Minkowski p-norm distance (l_p distance)
func Minkowski(firstVector, secondVector []float64, p float64) (float64, error) {
	distance := 0.
	for ii := range firstVector {
		distance += math.Pow(math.Abs(firstVector[ii]-secondVector[ii]), p)
	}
	return math.Pow(distance, 1/p), nil
}

// WeightedMinkowski p-norm distance with weights (weighted l_p distance)
func WeightedMinkowski(firstVector, secondVector, weightVector []float64, p float64) (float64, error) {
	distance := 0.
	for ii := range firstVector {
		distance += weightVector[ii] * math.Pow(math.Abs(firstVector[ii]-secondVector[ii]), p)
	}
	return math.Pow(distance, 1/p), nil
}

// Chebyshev infinity norm distance (l_inf distance)
func Chebyshev(firstVector, secondVector []float64) (float64, error) {
	distance := 0.
	for ii := range firstVector {
		if math.Abs(firstVector[ii]-secondVector[ii]) >= distance {
			distance = math.Abs(firstVector[ii] - secondVector[ii])
		}
	}
	return distance, nil
}

func Hamming(firstVector, secondVector []float64) (float64, error) {
	distance := 0.
	for ii := range firstVector {
		if firstVector[ii] != secondVector[ii] {
			distance++
		}
	}
	return distance, nil
}

func BrayCurtis(firstVector, secondVector []float64) (float64, error) {
	numerator, denominator := 0., 0.
	for ii := range firstVector {
		numerator += math.Abs(firstVector[ii] - secondVector[ii])
		denominator += math.Abs(firstVector[ii] + secondVector[ii])
	}
	return numerator / denominator, nil
}

func Canberra(firstVector, secondVector []float64) (float64, error) {
	distance := 0.
	for ii := range firstVector {
		distance += (math.Abs(firstVector[ii]-secondVector[ii]) / (math.Abs(firstVector[ii]) + math.Abs(secondVector[ii])))
	}
	return distance, nil
}
