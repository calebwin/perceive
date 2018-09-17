## What it is
perceive allows for easy perceptron-based deep learning in Go with a single function. perceive aims to be a lightweight solution to basic machine learning use cases

## How to use it
The perceive function requires just four arguments.
- A list of inputs for training
- A list of outputs for training
- A list of inputs to evaluate
- A group of parameters

### Basic usage

In the following example, each input has three features.
```golang
import "github.com/calebwin/perceive"

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

myOutputs := perceive.Perceive(trainingInputs, trainingOutputs, myInputs, Parameters{
  []float64{0, 0, 0},
  0,
  0.1,
  perceive.Heaviside,
})

// myOutputs == []float64{0, 0, 1,}
```

### Parameters
perceive requires four parameters that can be fine-tuned to train the perceptron to yield more accurate outputs.
- a list of initial relative weights of each feature
- an initial bias (can be set to 0 for most purposes)
- a learning rate (can be set to 0.01 to start with)
- an activation function

perceive includes the following 13 default activation functions - Heaviside, Linear, Logistic, TanH, LeCunTanH, HardTanH, Rectifier, SmoothRectifier, Logit, Gaussian, Abs, Bipolar, BipolarSigmoid.

Custom activation functions can be provided as follows.
```golang
myOutputs := perceive.Perceive(trainingInputs, trainingOutputs, myInputs, Parameters{
  []float64{0, 0, 0},
  0,
  0.1,
  func (x float64) float64 {
    return x
  },
})
```

## Notes
perceive assumes that all inputs are (a) normalized and (b) scaled
perceive's default activation functions assume inputs are within the range (-1, 1)
