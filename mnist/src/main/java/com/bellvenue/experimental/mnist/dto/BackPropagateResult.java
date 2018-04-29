package com.bellvenue.experimental.mnist.dto;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class BackPropagateResult {

    private LayerMap weightErrors;

    private LayerMap biasErrors;
}
