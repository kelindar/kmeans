package kmeans

import (
	"github.com/stretchr/testify/assert"
	"io/ioutil"
	"log"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/kelindar/kmeans/distance"
)

// Test K-Means Algorithm in Iris Dataset
func TestKmeans(t *testing.T) {
	filePath, err := filepath.Abs("fixtures/iris.csv")
	if err != nil {
		log.Fatal(err)
	}
	content, err := ioutil.ReadFile(filePath)
	if err != nil {
		log.Fatal(err)
	}

	lines := strings.Split(string(content), "\n")
	irisData := make([]Observation, 0, len(lines))
	for _, line := range lines {
		vector := strings.Split(line, ",")
		label := vector[len(vector)-1]
		vector = vector[:len(vector)-1]
		floatVector := make([]float64, len(vector))
		for j := range vector {
			floatVector[j], err = strconv.ParseFloat(vector[j], 64)
		}

		irisData = append(irisData, Observation{
			Point: floatVector,
			Label: label,
		})
	}

	threshold := 10

	// Best Distance for Iris is Canberra Distance
	output, err := Cluster(irisData, 3, distance.Canberra, threshold)
	assert.NoError(t, err)

	misclassifiedOnes := 0
	for i, v := range output {
		if i < 50 {
			if v.Cluster != 2 {
				misclassifiedOnes++
			}
		} else if i < 100 {
			if v.Cluster != 1 {
				misclassifiedOnes++
			}
		} else {
			if v.Cluster != 0 {
				misclassifiedOnes++
			}
		}
	}
}
