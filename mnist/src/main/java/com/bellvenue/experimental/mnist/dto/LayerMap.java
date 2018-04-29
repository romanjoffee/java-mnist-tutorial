package com.bellvenue.experimental.mnist.dto;

import com.google.common.collect.Maps;
import lombok.AccessLevel;
import lombok.Getter;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.Map;

public class LayerMap {

    @Getter(value = AccessLevel.NONE)
    private Map<Integer, RealMatrix> items;

    public LayerMap() {
        items = Maps.newConcurrentMap();
    }

    public void put(int layerIndex, RealMatrix matrix) {
        items.put(layerIndex, matrix);
    }

    public void setRow(int layerIndex, int rowIndex, double[] array, int rowCount, int columnCount) {
        RealMatrix layerMatrix = items.computeIfAbsent(layerIndex, key -> new Array2DRowRealMatrix(rowCount, columnCount));
        layerMatrix.setRow(rowIndex, array);
    }

    public RealMatrix get(int layerIndex) {
        return items.get(layerIndex);
    }
}
