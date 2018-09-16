## What it is
perceive allows for easy perceptron-based deep learning in Go with a single function. perceive aims to be a lightweight solution to basic machine learning use cases

## How to use it
The perceive function requires just four arguments.
- A list of inputs for training
- A list of outputs for training
- A list of inputs to evaluate
- A group of parameters

In the following example, each input has three features.
```golang
import "github.com/calebwin/perceive"

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

myOutputs := perceive.Perceive(trainingInputs, trainingOutputs, myInputs, Parameters{
  []float32{0, 0, 0},
  0,
  0.1,
})

// myOutputs == []float32{0, 0, 1,}
```

## Notes
perceive assumes that all inputs are (a) normalized and (b) scaled to the range (-1, 1)
perceive assumes that all outputs are binary and can either be 0 or 1
perceive uses the Heaviside step function to calculate output of 0 or 1
