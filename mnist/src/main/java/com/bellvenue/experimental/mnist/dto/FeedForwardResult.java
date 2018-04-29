package com.bellvenue.experimental.mnist.dto;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class FeedForwardResult {

    private LayerMap activations;

    private LayerMap zs;
}