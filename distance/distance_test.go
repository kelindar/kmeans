package distance

/*
TODO: Figure out the limit in the column number and
how to format the unfinished lines due to the limitation

Test for Weighted Minkowski Distance should be improved
*/

import (
	"github.com/stretchr/testify/assert"
	"math"
	"testing"
)

func TestLPNorm(t *testing.T) {
	vector := []float64{3., 4.}
	const l1Out, l2Out = 7., 5.

	l1Norm, _ := LPNorm(vector, 1.)
	l2Norm, _ := LPNorm(vector, 2.)
	if l1Norm != l1Out {
		t.Errorf("Computed l1 Norm: %f\nActual l1 Norm: %f", l1Norm, l1Out)
	}
	if l2Norm != l2Out {
		t.Errorf("Computed l2 Norm: %f\nActual l2 Norm: %f", l2Norm, l2Out)
	}
}

func TestManhattanDistance(t *testing.T) {
	firstVector := []float64{1., 2., 3., 2}
	secondVector := []float64{3., 4., 5., -1}
	const out = 9.
	mDistance, _ := Manhattan(firstVector, secondVector)
	if mDistance != out {
		t.Errorf("\nComputed Manhattan Distance: %f\nActual Manhattan Distance: %f", mDistance, out)
	}
}

func TestEuclidean(t *testing.T) {
	firstVector := []float64{5., 12.}
	secondVector := []float64{0., 0.}
	thirdVector := []float64{8., 15.}
	fourthVector := []float64{20., 20.}
	out2 := math.Sqrt(18)
	const out1, out3 = 13., 17.
	firstEuclidean, _ := Euclidean(firstVector, secondVector)
	secondEuclidean, _ := Euclidean(firstVector, thirdVector)
	thirdEuclidean, _ := Euclidean(thirdVector, secondVector)
	anotherFirst, _ := Euclidean(thirdVector, fourthVector)

	if out1 != firstEuclidean {
		t.Errorf("\nComputed Euclidean Distance: %f\nActual Euclidean Distance: %f", firstEuclidean, out1)
	}
	if out1 != anotherFirst {
		t.Errorf("\nComputed Euclidean Distance: %f\nActual Euclidean Distance: %f", firstEuclidean, out1)
	}
	if out2 != secondEuclidean {
		t.Errorf("\nComputed Euclidean Distance: %f\nActual Euclidean Distance: %f", secondEuclidean, out2)
	}
	if out3 != thirdEuclidean {
		t.Errorf("\nComputed Euclidean Distance: %f\nActual Euclidean Distance: %f", thirdEuclidean, out3)
	}
}

func TestSquareEuclidean(t *testing.T) {
	firstVector := []float64{5., 12.}
	secondVector := []float64{0., 0.}
	thirdVector := []float64{8., 15.}
	fourthVector := []float64{20., 20.}
	const out1, out3 = 169., 289.
	firstSquaredEuclidean, _ := SquaredEuclidean(firstVector, secondVector)
	thirdSquaredEuclidean, _ := SquaredEuclidean(thirdVector, secondVector)
	anotherFirst, _ := SquaredEuclidean(thirdVector, fourthVector)

	if out1 != firstSquaredEuclidean {
		t.Errorf("\nComputed Euclidean Distance: %f\nActual Euclidean Distance: %f", firstSquaredEuclidean, out1)
	}
	if out1 != anotherFirst {
		t.Errorf("\nComputed Euclidean Distance: %f\nActual Euclidean Distance: %f", firstSquaredEuclidean, out1)
	}
	if out3 != thirdSquaredEuclidean {
		t.Errorf("\nComputed Euclidean Distance: %f\nActual Euclidean Distance: %f", thirdSquaredEuclidean, out3)
	}
}

func TestMinkowski(t *testing.T) {
	// p = 1 Test
	// Should be equal to Manhattan Distance
	firstVector := []float64{1., 2., 3., 2}
	secondVector := []float64{3., 4., 5., -1}
	mDistance, _ := Manhattan(firstVector, secondVector)
	l1Minkowski, _ := Minkowski(firstVector, secondVector, 1.)
	if mDistance != l1Minkowski {
		t.Errorf("\nComputed l1 Minkowski Distance: %f\nComputed Manhattan Distance: %f", l1Minkowski, mDistance)
	}

	// p = 2 Test
	// Should be equal to Euclidean Distance
	firstVector = []float64{5., 12.}
	secondVector = []float64{0., 0.}
	thirdVector := []float64{8., 15.}
	fourthVector := []float64{20., 20.}

	firstEuclidean, _ := Euclidean(firstVector, secondVector)
	anotherFirstEuclidean, _ := Euclidean(thirdVector, fourthVector)
	secondEuclidean, _ := Euclidean(firstVector, thirdVector)
	thirdEuclidean, _ := Euclidean(thirdVector, secondVector)

	firstl2Minkowski, _ := Minkowski(firstVector, secondVector, 2.)
	anotherFirstl2Minkowski, _ := Minkowski(thirdVector, fourthVector, 2.)
	secondl2Minkowski, _ := Minkowski(firstVector, thirdVector, 2.)
	thirdl2Minkowski, _ := Minkowski(thirdVector, secondVector, 2.)

	if firstEuclidean != firstl2Minkowski {
		t.Errorf("\nComputed l2 Minkowski Distance: %f\nComputed Euclidean Distance: %f", firstl2Minkowski, firstEuclidean)
	}
	if secondEuclidean != secondl2Minkowski {
		t.Errorf("\nComputed l2 Minkowski Distance: %f\nComputed Euclidean Distance: %f", secondl2Minkowski, secondEuclidean)
	}
	if thirdEuclidean != thirdl2Minkowski {
		t.Errorf("\nComputed l2 Minkowski Distance: %f\nComputed Euclidean Distance: %f", thirdl2Minkowski, thirdEuclidean)
	}
	if anotherFirstEuclidean != firstl2Minkowski {
		t.Errorf("\nComputed l2 Minkowski Distance: %f\nComputed Euclidean Distance: %f", anotherFirstl2Minkowski, anotherFirstEuclidean)
	}

	// p = 3 and p = 4 Test
	const l3Minkowski, l4Minkowski, precision = 12.282642, 12.089418, 1000000.

	computedl3Minkowski, _ := Minkowski(firstVector, secondVector, 3.)
	computedl4Minkowski, _ := Minkowski(firstVector, secondVector, 4.)
	computedl3Minkowski = float64(int(computedl3Minkowski*precision)) / precision
	computedl4Minkowski = float64(int(computedl4Minkowski*precision)) / precision

	if l3Minkowski != computedl3Minkowski {
		t.Errorf("\nComputed l3 Minkowski Distance: %f\nActual l3 Minkowski Distance: %f", computedl3Minkowski, l3Minkowski)
	}

	if l4Minkowski != computedl4Minkowski {
		t.Errorf("\nComputed l4 Minkowski Distance: %f\nActual l4 Minkowski Distance: %f", computedl4Minkowski, l4Minkowski)
	}
}

func TestWeightedMinkowski(t *testing.T) {
	// Weight Vector is all 1.
	// Results should be same when we do not apply any weighting vector
	firstVector := []float64{1., 2., 3., 2}
	secondVector := []float64{3., 4., 5., -1}
	weightVector := []float64{1., 1., 1., 1.}
	l1Minkowski, _ := Minkowski(firstVector, secondVector, 1.)
	l1WeightedMinkowski, _ := WeightedMinkowski(firstVector, secondVector, weightVector, 1.)

	if l1Minkowski != l1WeightedMinkowski {
		t.Errorf("\nComputed l1 Minkowski Distance: %f\nActual Weighted(1., 1., ...) l1 Minkowski Distance: %f", l1Minkowski, l1WeightedMinkowski)
	}

	firstVector = []float64{5., 12.}
	secondVector = []float64{0., 0.}
	thirdVector := []float64{8., 15.}
	fourthVector := []float64{20., 20.}

	firstl2Minkowski, _ := Minkowski(firstVector, secondVector, 2.)
	anotherFirstl2Minkowski, _ := Minkowski(thirdVector, fourthVector, 2.)
	secondl2Minkowski, _ := Minkowski(firstVector, thirdVector, 2.)
	thirdl2Minkowski, _ := Minkowski(thirdVector, secondVector, 2.)

	firstl2WeightedMinkowski, _ := WeightedMinkowski(firstVector, secondVector, weightVector, 2.)
	anotherFirstl2WeightedMinkowski, _ := WeightedMinkowski(thirdVector, fourthVector, weightVector, 2.)
	secondl2WeightedMinkowski, _ := WeightedMinkowski(firstVector, thirdVector, weightVector, 2.)
	thirdl2WeightedMinkowski, _ := WeightedMinkowski(thirdVector, secondVector, weightVector, 2.)

	if firstl2Minkowski != firstl2WeightedMinkowski {
		t.Errorf("\nComputed l2 Minkowski Distance: %f\nActual Weighted(1., 1., ...) l2 Minkowski Distance: %f", firstl2Minkowski, firstl2WeightedMinkowski)
	}
	if anotherFirstl2Minkowski != anotherFirstl2WeightedMinkowski {
		t.Errorf("\nComputed l2 Minkowski Distance: %f\nActual Weighted(1., 1., ...) l2 Minkowski Distance: %f", anotherFirstl2Minkowski, anotherFirstl2WeightedMinkowski)
	}
	if secondl2Minkowski != secondl2WeightedMinkowski {
		t.Errorf("\nComputed l2 Minkowski Distance: %f\nActual Weighted(1., 1., ...) l2 Minkowski Distance: %f", secondl2Minkowski, secondl2WeightedMinkowski)
	}
	if thirdl2Minkowski != thirdl2WeightedMinkowski {
		t.Errorf("\nComputed l2 Minkowski Distance: %f\nActual Weighted(1., 1., ...) l2 Minkowski Distance: %f", thirdl2Minkowski, thirdl2WeightedMinkowski)
	}

}

func TestChebyshev(t *testing.T) {
	firstVector := []float64{1., 2., 3., 4.}
	secondVector := []float64{3., -4., 6., 1.5}
	thirdVector := []float64{4., 3., -2.5, -5.}
	const firstActual, secondActual = 6., 8.5
	firstComputed, _ := Chebyshev(firstVector, secondVector)
	secondComputed, _ := Chebyshev(secondVector, thirdVector)
	if firstComputed != firstActual {
		t.Errorf("\nComputed Chebyshev Distance: %f\nActual Chebyshev Distance: %f", firstComputed, firstActual)
	}
	if secondComputed != secondActual {
		t.Errorf("\nComputed Chebyshev Distance: %f\nActual Chebyshev Distance: %f", secondComputed, secondActual)
	}
}

func TestHamming(t *testing.T) {
	firstVector := []float64{1., 2., 2.5, 3., 4.}
	secondVector := []float64{1., 2.5, 3., 3., 4.}
	thirdVector := []float64{1., 2., 3., 4., 5., 6.}
	fourthVector := []float64{1., 1., 1., 1., 1., 1.}
	const firstActual, secondActual = 2, 5
	firstComputed, _ := Hamming(firstVector, secondVector)
	secondComputed, _ := Hamming(thirdVector, fourthVector)

	if firstComputed != firstActual {
		t.Errorf("\nComputed Hamming Distance: %f\nActual Hamming Distance: %d", firstComputed, firstActual)
	}
	if secondComputed != secondActual {
		t.Errorf("\nComputed Hamming Distance: %f\nActual Hmming Distance: %d", secondComputed, secondActual)
	}
}

func TestBrayCurtis(t *testing.T) {
	firstVector := []float64{1., 2., 3., 4., 5.}
	secondVector := []float64{1.5, 2.5, 5., 5., 6.}

	thirdVector := []float64{3., 2., 4., 6.5, 7}
	fourthVector := []float64{1., 6., 3., 5.5, 4.5}
	const firstActual, secondActual, precision = 0.14285, 0.24705, 100000

	firstComputed, _ := BrayCurtis(firstVector, secondVector)
	secondComputed, _ := BrayCurtis(thirdVector, fourthVector)

	firstComputed = float64(int(firstComputed*precision)) / precision
	secondComputed = float64(int(secondComputed*precision)) / precision

	if firstComputed != firstActual {
		t.Errorf("\nComputed Bray Curtis Distance: %f\nActual Bray Curtis Distance: %f", firstComputed, firstActual)
	}
	if secondComputed != secondActual {
		t.Errorf("\nComputed Bray Curtis Distance: %f\nActual Bray Curtis Distance: %f", secondComputed, secondActual)
	}
}

func TestCanberraDistance(t *testing.T) {
	firstVector := []float64{3., 4., 5., -2., 4.}
	secondVector := []float64{2., 6., 5., 3., -1.}
	const firstActual = 2.4
	firstComputed, _ := Canberra(firstVector, secondVector)
	if firstActual != firstComputed {
		t.Errorf("Computed Canberra Distance: %f\n Actual Canberra Distance: %f", firstComputed, firstActual)
	}
}

func TestHistogramIntersection(t *testing.T) {
	hist1 := []float64{1, 2, 10, 50, 30, 5, 2, 0, 1, 1}
	hist2 := []float64{1, 2, 5, 10, 50, 30, 3, 2, 1, 1}

	{
		_, err := NormalizedIntersection(hist1, []float64{0.0})
		assert.Error(t, err)
	}
	{
		v, err := NormalizedIntersection(hist1, hist1)
		assert.NoError(t, err)
		assert.Equal(t, 1.0, v)
	}
	{
		v, err := NormalizedIntersection(hist1, hist2)
		assert.NoError(t, err)
		assert.Equal(t, 0.5428571428571428, v)
	}
}
