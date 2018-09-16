package perceive

import (
)

func heaviside(x float32) float32 {
  if x < 0 {
    return 0
  }
  return 1
}

type Parameters struct {
  initialWeights []float32
  initialBias float32
  learningRate float32
}

func Perceive(trainingInputs [][]float32, trainingOutputs []float32, inputs [][]float32, parameters Parameters) []float32 {
  weights := parameters.initialWeights
  bias := parameters.initialBias
  learningRate := parameters.learningRate

  // adjust weights and random values with training data
  for i, trainingInput := range trainingInputs {
    // calculate expected output of training input
    expectedOutput := trainingOutputs[i]

    // calculate actual output of training input
    actualOuput := bias
    for i, input := range trainingInput {
      actualOuput += input * weights[i]
    }
    actualOuput = float32(heaviside(actualOuput))

    // calculate cost
    cost := expectedOutput - actualOuput

    // adjust weights based on cost
    for i, input := range trainingInput {
      weights[i] += input * cost * learningRate
    }

    // adjust bias based on cost
    bias += cost * learningRate
  }

  outputs := []float32{}

  for _, inputInputs := range inputs {
    // calculate sum of weighted inputs
    inputSum := bias
    for i, input := range inputInputs {
      inputSum += input * weights[i]
    }

    // calculate output by applying activation function
    output := heaviside(inputSum)

    // add output to outputs array
    outputs = append(outputs, output)
  }

  // return sum of weighted inputs passed into activation function
  return outputs
}
