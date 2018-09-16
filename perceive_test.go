package perceive

import (
  "testing"
  "fmt"
)

func TestBasic(t *testing.T) {
  trainingInputs := [][]float32{
    []float32{-0.8, -0.7, -0.5,},
    []float32{-0.5, -0.4, -0.4,},
    []float32{0.0, 0.2, 0.5,},
    []float32{-0.5, -0.5, -0.4,},
    []float32{0.7, 0.8, 0.9,},
  }

  trainingOutputs := []float32{
    0,
    0,
    1,
    0,
    1,
  }

  myInputs := [][]float32{
    []float32{-0.8, -0.7, -0.5,},
    []float32{-0.5, -0.4, -0.4,},
    []float32{0.0, 0.2, 0.5,},
  }

  myOutputs := Perceive(trainingInputs, trainingOutputs, myInputs, Parameters{
    []float32{0, 0, 0},
    0,
    0.1,
  })

  for _, myOutput := range myOutputs {
    fmt.Println(myOutput)
  }
}
