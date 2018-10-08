package perceive

//type ActivationFunction func(x float64) (result float64)

type Parameters struct {
  initialWeights []float64
  learningRate float64
  numEpochs int
  //activationFunction ActivationFunction
}

func Perceive(trainingInputsClass1 [][]float64, trainingInputsClass2 [][]float64, inputs [][]float64, parameters Parameters) ([][]float64, [][]float64) {
  weights := parameters.initialWeights
  bias := 0.0
  learningRate := parameters.learningRate
  numEpochs := parameters.numEpochs
  activationFunction := Heaviside//parameters.activationFunction

  // generate training data from 2 classes of input data
  trainingInputs := make([][]float64, len(trainingInputsClass1) + len(trainingInputsClass2))
  trainingOutputs := make([]float64, len(trainingInputsClass1) + len(trainingInputsClass2))
  nextTrainingInputIndex := 0
  for _, trainingInput := range trainingInputsClass1 {
    trainingInputs[nextTrainingInputIndex] = []float64{}
    trainingInputs[nextTrainingInputIndex] = append(trainingInputs[nextTrainingInputIndex], trainingInput...)
    trainingOutputs = append(trainingOutputs, 0.0)
    nextTrainingInputIndex++
  }
  for _, trainingInput := range trainingInputsClass2 {
    trainingInputs[nextTrainingInputIndex] = []float64{}
    trainingInputs[nextTrainingInputIndex] = append(trainingInputs[nextTrainingInputIndex], trainingInput...)
    trainingOutputs = append(trainingOutputs, 1.0)
    nextTrainingInputIndex++
  }

  for numEpochs > 0 {
    // adjust weights and random values with training data
    for i, trainingInput := range trainingInputs {
      // calculate expected output of training input
      expectedOutput := trainingOutputs[i]

      // calculate actual output of training input
      actualOutput := 0.0
      actualOutput += bias
      for j, input := range trainingInput {
        actualOutput += input * weights[j]
      }
      actualOutput = activationFunction(actualOutput)

      // calculate cost
      cost := expectedOutput - actualOutput

      // adjust weights based on cost
      for j, input := range trainingInput {
        weights[j] += input * cost * learningRate
      }

      // adjust bias based on cost
      bias += cost * learningRate
    }

    numEpochs--
  }

  outputs := [][]float64{}
  outputClassIndices := []float64{}

  for _, inputInputs := range inputs {
    // calculate sum of weighted inputs
    inputSum := bias
    for i, input := range inputInputs {
      inputSum += input * weights[i]
    }

    // calculate output class index by applying activation function
    outputClassIndex := activationFunction(inputSum)

    // add output class index to output class indices array
    outputClassIndices = append(outputClassIndices, outputClassIndex)

    // add output to outputs array
    outputs = append(outputs, []float64{})
    outputs[len(outputs) - 1] = append(outputs[len(outputs) - 1], inputInputs...)
  }

  // generated seperated classes of outputs
  outputsClass1 := [][]float64{}
  outputsClass2 := [][]float64{}
  for i, outputClassIndex := range outputClassIndices {
    if int(outputClassIndex) == 0 {
      outputsClass1 = append(outputsClass1, []float64{})
      outputsClass1[len(outputsClass1) - 1] = append(outputsClass1[len(outputsClass1) - 1], outputs[i]...)
    }
    if int(outputClassIndex) == 1 {
      outputsClass2 = append(outputsClass2, []float64{})
      outputsClass2[len(outputsClass2) - 1] = append(outputsClass2[len(outputsClass2) - 1], outputs[i]...)
    }
  }

  // return sum of weighted inputs passed into activation function
  return outputsClass1, outputsClass2
}
