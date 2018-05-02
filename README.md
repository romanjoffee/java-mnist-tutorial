# java-mnist-tutorial

Neural network to classify MNIST handwritten digits - implemented in Java

## Data model

*See [MNIST](http://yann.lecun.com/exdb/mnist/) for description of the MNIST data*

## Getting Started

```
mvn clean install -U
```

### Prerequisites

Java 8

## Description
Spring-boot app

Configuration is set in `AppConfiguration.java` and `application.yml`

Main model logic (i.e. back-propagation) is in `Processor.java`

Training can be executed in 2 modes - `serial` and `concurrent`

To change between modes edit `application.yml` entry:
```
spring:
  profiles:
    include: general
```
`general` - execute training in serial mode - each batch will processed synchronously

`concurrent` - execute batches in concurrent mode - each batch will processed concurrently in a thread-pool