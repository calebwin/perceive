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
