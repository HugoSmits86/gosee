# Go parameters
    GOCMD=go
    GOBUILD=$(GOCMD) build
    GOINSTALL=$(GOCMD) install
    GOCLEAN=$(GOCMD) clean
    GOTEST=$(GOCMD) test
    GOGET=$(GOCMD) get
    BINARY_NAME=nn
    BINARY_MACOS=$(BINARY_NAME)_macos
    BINARY_UNIX=$(BINARY_NAME)_unix
    BINARY_WIN=$(BINARY_NAME)_win.exe
    
    all: test build
    build: 
	$(GOBUILD) -o $(BINARY_NAME) -v
    test: 
	$(GOTEST) -v ./...
    install:
	$(GOINSTALL) -v
    clean: 
	$(GOCLEAN)
	rm -f $(BINARY_NAME)
    run:
	$(GOBUILD) -o $(BINARY_NAME) -v ./...
	./$(BINARY_NAME)
    
    # Cross cleaning
    clean-all: clean-windows clean-macos clean-linux
    clean-windows: 
	CGO_ENABLED=0 GOOS=windows GOARCH=amd64 $(GOCLEAN) -i
	rm -f $(BINARY_NAME)
	rm -f $(BINARY_WIN)
    clean-macos: 
	CGO_ENABLED=0 GOOS=darwin GOARCH=amd64 $(GOCLEAN) -i
	rm -f $(BINARY_NAME)
	rm -f $(BINARY_MACOS)
    clean-linux: 
	CGO_ENABLED=0 GOOS=linux GOARCH=amd64 $(GOCLEAN) -i
	rm -f $(BINARY_NAME)
	rm -f $(BINARY_UNIX)

    # Cross compilation
    build-all: build-windows build-macos build-linux
    build-windows:
	CGO_ENABLED=0 GOOS=windows GOARCH=amd64 $(GOBUILD) -o $(BINARY_WIN) -v
    build-macos:
	CGO_ENABLED=0 GOOS=darwin GOARCH=amd64 $(GOBUILD) -o $(BINARY_MACOS) -v
    build-linux:
	CGO_ENABLED=0 GOOS=linux GOARCH=amd64 $(GOBUILD) -o $(BINARY_UNIX) -v