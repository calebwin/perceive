package perceive

import (
  "math"
)

type ActivationFunction func(x float64) (result float64)

func Heaviside(x float64) float64 {
  if x < 0 {
    return 0
  }
  return 1
}

func Linear(x float64) float64 {
  return x
}

func Logistic(x float64) float64 {
  return 1 / (1 + math.Pow(math.E, -x))
}

func TanH(x float64) float64 {
  return 2 * Logistic(2 * x) - 1
}

func Rectifier(x float64) float64 {
  return math.Max(0.0, x)
}

func Bipolar(x float64) float64 {
  if x < 0 {
    return -1
  } else if x > 0 {
    return 1
  }
  return 0
}

func BipolarSigmoid(x float64) float64 {
  return (1 - math.Pow(math.E, -x)) / (1 + math.Pow(math.E, -x))
}

func LeCunTahH(x float64) float64 {
  return 1.7159 * TanH(2 * x / 3)
}

func HardTahH(x float64) float64 {
  return math.Max(-1, math.Min(1, x))
}

func Abs(x float64) float64 {
  return math.Abs(x)
}

func SmoothRectifier(x float64) float64 {
  return math.Log(1 + math.Pow(math.E, x))
}

func Logit(x float64) float64 {
  return math.Log(x / (1 - x))
}

func Gaussian(x float64) float64 {
  return math.Pow(math.E, -0.5 * math.Pow(x, 2))
}

type Parameters struct {
  initialWeights []float64
  initialBias float64
  learningRate float64
  activationFunction ActivationFunction
}

func Perceive(trainingInputs [][]float64, trainingOutputs []float64, inputs [][]float64, parameters Parameters) []float64 {
  weights := parameters.initialWeights
  bias := parameters.initialBias
  learningRate := parameters.learningRate
  activationFunction := parameters.activationFunction

  // adjust weights and random values with training data
  for i, trainingInput := range trainingInputs {
    // calculate expected output of training input
    expectedOutput := trainingOutputs[i]

    // calculate actual output of training input
    actualOuput := bias
    for i, input := range trainingInput {
      actualOuput += input * weights[i]
    }
    actualOuput = float64(activationFunction(actualOuput))

    // calculate cost
    cost := expectedOutput - actualOuput

    // adjust weights based on cost
    for i, input := range trainingInput {
      weights[i] += input * cost * learningRate
    }

    // adjust bias based on cost
    bias += cost * learningRate
  }

  outputs := []float64{}

  for _, inputInputs := range inputs {
    // calculate sum of weighted inputs
    inputSum := bias
    for i, input := range inputInputs {
      inputSum += input * weights[i]
    }

    // calculate output by applying activation function
    output := activationFunction(inputSum)

    // add output to outputs array
    outputs = append(outputs, output)
  }

  // return sum of weighted inputs passed into activation function
  return outputs
}
