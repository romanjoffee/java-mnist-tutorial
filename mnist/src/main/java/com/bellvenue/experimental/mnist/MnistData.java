package com.bellvenue.experimental.mnist;

import com.bellvenue.experimental.mnist.dto.MnistObservation;
import lombok.Data;

import java.util.List;

@Data
class MnistData {

    private int imageCount;

    private int labelCount;

    private int imageVectorSize;

    private List<MnistObservation> data;
}
