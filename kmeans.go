package kmeans

import (
	"math"
	"math/rand"
)

// Observation represents an data abstraction for an N-dimensional
// observation with a label.
type Observation struct {
	Point []float64 // The multi-dimentional point
	Label string    // The label of the observation
}

// ClusteredObservation abstracts an observation with a cluster number
type ClusteredObservation struct {
	ClusterNumber int // The cluster index
	Observation       // The observation
}

// DistanceFunction used to compute the distanfe between observations
type DistanceFunction func(first, second []float64) (float64, error)

// Add adds two vectors together
func (observation Observation) Add(other Observation) {
	for i, j := range other.Point {
		observation.Point[i] += j
	}
}

// Mul multiplies vector with a scalar
func (observation Observation) Mul(scalar float64) {
	for ii := range observation.Point {
		observation.Point[ii] *= scalar
	}
}

// InnerProduct computes a dot product of two vectors
func (observation Observation) InnerProduct(other Observation) {
	for ii := range observation.Point {
		observation.Point[ii] *= other.Point[ii]
	}
}

// OuterProduct of two arrays
func (observation Observation) OuterProduct(other Observation) [][]float64 {
	result := make([][]float64, len(observation.Point))
	for i := range result {
		result[i] = make([]float64, len(other.Point))
	}
	for i := range result {
		for j := range result[i] {
			result[i][j] = observation.Point[i] * other.Point[j]
		}
	}
	return result
}

// Find the closest observation and return the distance
// Index of observation, distance
func near(p ClusteredObservation, mean []Observation, distanceFunction DistanceFunction) (int, float64) {
	indexOfCluster := 0
	minSquaredDistance, _ := distanceFunction(p.Observation.Point, mean[0].Point)
	for i := 1; i < len(mean); i++ {
		squaredDistance, _ := distanceFunction(p.Observation.Point, mean[i].Point)
		if squaredDistance < minSquaredDistance {
			minSquaredDistance = squaredDistance
			indexOfCluster = i
		}
	}
	return indexOfCluster, math.Sqrt(minSquaredDistance)
}

// Instead of initializing randomly the seeds, make a sound decision of initializing
func seed(data []ClusteredObservation, k int, distanceFunction DistanceFunction) []Observation {
	s := make([]Observation, k)
	s[0] = data[rand.Intn(len(data))].Observation
	d2 := make([]float64, len(data))
	for i := 1; i < k; i++ {
		var sum float64
		for j, p := range data {
			_, dMin := near(p, s[:i], distanceFunction)
			d2[j] = dMin * dMin
			sum += d2[j]
		}
		target := rand.Float64() * sum
		j := 0
		for sum = d2[0]; sum < target; sum += d2[j] {
			j++
		}
		s[i] = data[j].Observation
	}
	return s
}

// K-Means Algorithm
func kmeans(data []ClusteredObservation, mean []Observation, distanceFunction DistanceFunction, threshold int) ([]ClusteredObservation, error) {
	counter := 0
	for i, j := range data {
		closestCluster, _ := near(j, mean, distanceFunction)
		data[i].ClusterNumber = closestCluster
	}

	mLen := make([]int, len(mean))
	for n := len(data[0].Observation.Point); ; {
		for i := range mean {
			mLen[i] = 0
			mean[i] = Observation{
				Point: make([]float64, n),
			}
		}
		for _, p := range data {
			mean[p.ClusterNumber].Add(p.Observation)
			mLen[p.ClusterNumber]++
		}
		for i := range mean {
			mean[i].Mul(1 / float64(mLen[i]))
		}
		var changes int
		for i, p := range data {
			if closestCluster, _ := near(p, mean, distanceFunction); closestCluster != p.ClusterNumber {
				changes++
				data[i].ClusterNumber = closestCluster
			}
		}
		counter++
		if changes == 0 || counter > threshold {
			return data, nil
		}
	}
	return data, nil
}

// Cluster clusters the set of observations using K-Means ++ with a specified distance function
func Cluster(input []Observation, k int, distanceFunction DistanceFunction, threshold int) ([]int, error) {
	data := make([]ClusteredObservation, len(input))
	for i, v := range input {
		data[i].Observation = v
	}

	seeds := seed(data, k, distanceFunction)
	clusteredData, err := kmeans(data, seeds, distanceFunction, threshold)
	labels := make([]int, len(clusteredData))
	for i, j := range clusteredData {
		labels[i] = j.ClusterNumber
	}
	return labels, err
}
