## What it is
perceive allows for easy perceptron-based binary classification in Go with a single function. perceive aims to be a lightweight solution for basic machine learning use cases.

## How to use it
The perceive function requires just four arguments.
- A list of inputs in the first class for training
- A list of inputs in the second class for training
- A list of inputs to be classified
- A group of parameters

### Basic usage

In the following example, each input has three features.
```golang
import "github.com/calebwin/perceive"

trainingInputsClassA := [][]float64{
  []float64{-0.8, -0.7, -0.5,},
  []float64{-0.5, -0.4, -0.4,},
  []float64{-0.5, -0.5, -0.4,},
}

trainingInputsClassB := [][]float64{
  []float64{0.0, 0.2, 0.5,},
  []float64{0.7, 0.8, 0.9,},
}

myInputs := [][]float64{
  []float64{-0.8, -0.7, -0.5,},
  []float64{-0.5, -0.4, -0.4,},
  []float64{0.0, 0.2, 0.5,},
}

myOutputsClassA, myOutputsClassB := perceive.Perceive(trainingInputsClassA, trainingInputsClassB, myInputs, perceive.Parameters{
  []float64{0, 0, 0},
  0.1,
  32,
})
```

### Parameters
perceive requires four parameters that can be fine-tuned to train the perceptron to yield more accurate outputs.
- a list of initial relative weights of each input
- a learning rate (can be set to 0.01 to start with)
- number of epochs (number of times to update classification model during learning phase)

## Notes
perceive assumes that all inputs are (a) normalized and (b) scaled
perceive uses stochastic gradient descent by default
