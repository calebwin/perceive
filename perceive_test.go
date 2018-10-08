package perceive

import (
  "testing"
  "fmt"
)

func TestBasic(t *testing.T) {
  trainingInputsClass1 := [][]float64{
    []float64{-0.8, -0.7, -0.5,},
    []float64{-0.5, -0.4, -0.4,},
    []float64{-0.5, -0.5, -0.4,},
  }

  trainingInputsClass2 := [][]float64{
    []float64{0.0, 0.2, 0.5,},
    []float64{0.7, 0.8, 0.9,},
  }

  myInputs := [][]float64{
    []float64{-0.8, -0.7, -0.5,},
    []float64{-0.5, -0.4, -0.4,},
    []float64{1.7, 1.8, 1.9,},
  }

  myOutputsClass1, myOutputsClass2 := Perceive(trainingInputsClass1, trainingInputsClass2, myInputs, Parameters{
    []float64{0, 0, 0},
    0.1,
    3,
  })

  fmt.Println("Class 1:")
  for _, myOutput := range myOutputsClass1 {
    fmt.Println(myOutput)
  }

  fmt.Println("Class 2:")
  for _, myOutput := range myOutputsClass2 {
    fmt.Println(myOutput)
  }
}
