package com.bellvenue.experimental.mnist.dto;

import lombok.Builder;
import lombok.Getter;
import org.apache.commons.math3.linear.ArrayRealVector;

@Builder
@Getter
public class MnistObservation {

    private ArrayRealVector pixelVector;

    private ArrayRealVector labelVector;
}
