package main

import (
    "os"
    "image"
    "image/png"
    "math"
    "errors"
    "fmt"
)

const (
    blockSize = 64
)

func main() {
    // first we initialze data set which will be used to train the network
    inputs, outputs, weights, err := initializeDataset()
    if err != nil {
        fmt.Println(err)
        return
    }

    //then we train the network with the dataset for 1500 trails
    train(1500, inputs, outputs, &weights)

    //and last we verify the network by predicting images not part of the training data
    for id, tt := range []struct {
        fileName        string
        expected        bool
    }{
        {
            "testdata/pac3.png", 
            true,
        },
        {
            "testdata/ghost3.png", 
            false,
        },

    } {
        img, err := openImage(tt.fileName)
        if err != nil {
            fmt.Printf("test %v: unexpected error - %v", id, err)
            continue
        }

        mask := createMask(img)

        //p ranges from 0 to 1 and indicates the level of certainty
        p := predict(mask, weights)
        result := true
        if p < 0.5 {
            result = false
        }

        fmt.Printf("test %v: does image '%v' depict pacman?\n", id, tt.fileName)
        fmt.Printf("  prediction: %v, certainty: %.2f\n", result, p)
        fmt.Printf("  prediction correct? %v\n", tt.expected == result)
    }
}

func openImage(fileName string) (image.Image, error) {
    file, err := os.Open(fileName)
    if err != nil {
      return nil, err
    }

    img, err := png.Decode(file)
    if err != nil {
        return nil, err
    }

    b := img.Bounds()
    if b.Dx() * b.Dy() != blockSize {
        return nil, errors.New("incorrect image dimensions")
    }

    return img, nil
}

func createMask(img image.Image) []float64 {
    b := img.Bounds()
    w := b.Dx()
    h := b.Dy()

    sprite := make([]float64, w * h)
    for y := 1; y < h; y++ {
        for x := 1; x < w; x++ {
            r, g, b, _ := img.At(x, y).RGBA()
            if r != 0 || g != 0 || b != 0 {
                sprite[y * w + x] = 1
            }
        }
    }

    return sprite
}

func initializeDataset() ([][]float64, []float64, []float64, error) {
    images := []string {
        "testdata/ghost1.png",
        "testdata/ghost2.png",
        "testdata/pac1.png",
        "testdata/pac2.png",
    }

    outputs := []float64 {
        0,
        0,
        1,
        1,
    }

    weights := make([]float64, blockSize)
    for i := 0; i < len(weights); i++ {
        weights[i] = 0.5
    }

    inputs := make([][]float64, len(images))
    for i := 0; i < len(images); i++ {
        img, err := openImage(images[i])
        if err != nil {
            return nil, nil, nil, err
        }

        mask := createMask(img)
        inputs[i] = mask
    }

    return inputs, outputs, weights, nil
}

func sigmoid(x float64) float64 { 
    return 1 / (1 + math.Exp(-x))
}

func dSigmoid(x float64) float64 { 
    return x * (1 - x)
}

func dot(v1 []float64, v2 []float64) float64 {
    r := float64(0)
    for i := 0; i < len(v1); i++ {
        r += v1[i] * v2[i]
    }
    return r
}

func train(trials int, inputs [][]float64, outputs []float64, weights *[]float64) {
    hidden := make([]float64, len(inputs))
    for i := 0; i < trials; i++ {
        feedForward(inputs, *weights, &hidden)
        backPropagation(inputs, hidden, outputs, weights)
    }
}

func feedForward(inputs [][]float64, weights []float64, hidden *[]float64) {
    for i := 0; i < len(inputs); i ++ {
        (*hidden)[i] = sigmoid(dot(inputs[i], weights))
    }
}

func backPropagation(inputs [][]float64, hidden []float64, outputs []float64, weights *[]float64) {
    deltas := make([]float64, len(inputs))
    for i := 0; i < len(inputs); i++ {
        err := outputs[i] - hidden[i]
        deltas[i] = err * dSigmoid(hidden[i])
    }

    m := transposeMatrix(inputs)
    for i := 0; i < len((*weights)); i++ {
        (*weights)[i] += dot(m[i], deltas)
    }
}

func transposeMatrix(input [][]float64) [][]float64 {
    numColumns := len(input[0])
    numRows := len(input)
    dest := make([][]float64, numColumns)
    for c := 0; c < numColumns; c++ {
        dest[c] = make([]float64, numRows)
        for r := 0; r < numRows; r++ {
            dest[c][r] = input[r][c]
        }
    }
    return dest
}

func predict(input []float64, weights []float64) float64 {
    return sigmoid(dot(input, weights))
}
