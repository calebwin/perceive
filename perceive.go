package perceive

//type ActivationFunction func(x float64) (result float64)

type Parameters struct {
  initialWeights []float64
  initialBias float64
  learningRate float64
  numEpochs int
  //activationFunction ActivationFunction
}

func Perceive(trainingInputs [][]float64, trainingOutputs []float64, inputs [][]float64, parameters Parameters) []float64 {
  weights := parameters.initialWeights
  bias := parameters.initialBias
  learningRate := parameters.learningRate
  numEpochs := parameters.numEpochs
  activationFunction := Heaviside//parameters.activationFunction

  for numEpochs > 0 {
    // adjust weights and random values with training data
    for i, trainingInput := range trainingInputs {
      // calculate expected output of training input
      expectedOutput := trainingOutputs[i]

      // calculate actual output of training input
      actualOuput := 0.0
      actualOuput += bias
      for i, input := range trainingInput {
        actualOuput += input * weights[i]
      }
      actualOuput = activationFunction(actualOuput)

      // calculate cost
      cost := expectedOutput - actualOuput

      // adjust weights based on cost
      for i, input := range trainingInput {
        weights[i] += input * cost * learningRate
      }

      // adjust bias based on cost
      bias += cost * learningRate
    }

    numEpochs--
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
