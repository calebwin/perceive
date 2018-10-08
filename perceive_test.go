package perceive

import (
  "testing"
  "fmt"
)

func TestBasic(t *testing.T) {
  trainingInputs := [][]float64{
    []float64{-0.8, -0.7, -0.5,},
    []float64{-0.5, -0.4, -0.4,},
    []float64{0.0, 0.2, 0.5,},
    []float64{-0.5, -0.5, -0.4,},
    []float64{0.7, 0.8, 0.9,},
  }

  trainingOutputs := []float64{
    0,
    0,
    1,
    0,
    1,
  }

  myInputs := [][]float64{
    []float64{-0.8, -0.7, -0.5,},
    []float64{-0.5, -0.4, -0.4,},
    []float64{0.0, 0.2, 0.5,},
  }

  myOutputs := Perceive(trainingInputs, trainingOutputs, myInputs, Parameters{
    []float64{0, 0, 0},
    0,
    0.1,
    3,
  })

  for _, myOutput := range myOutputs {
    fmt.Println(myOutput)
  }
}
