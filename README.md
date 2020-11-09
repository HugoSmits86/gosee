[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Go Report Card](https://goreportcard.com/badge/github.com/HugoSmits86/gosee)](https://goreportcard.com/report/github.com/HugoSmits86/gosee)
![Build](https://github.com/HugoSmits86/gosee/workflows/Build/badge.svg)

# Introduction

Toy neural network for image recognizing from scratch in less than 200 lines of code.

This research and development project is written to aid my personal quest to gain an in-depth understanding 
of deep learning. The project contains a simple two-layered neural network utilizing backpropagation 
for weight adjustments. This code should only be used for educational purposes.

# Results

The neural network is able to predict if an image depicts pacman or not.

Image | Prediction | Certainty
------------ | ------------- | -------------
<img src="testdata/pac3.png" width=32px height=32px> | yes | 0.99
<img src="testdata/ghost1.png" width=32px height=32px> | no | 0.05

# Build project

The command-line tool includes a make file that can build the tool for multiple platforms.

```Bash
#compile and install package for Windows
make build-windows
#compile and install package for MacOs
make build-macos
#compile and install package for Linux
make build-linux
