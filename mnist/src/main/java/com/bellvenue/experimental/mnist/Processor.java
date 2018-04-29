package com.bellvenue.experimental.mnist;

import com.bellvenue.experimental.mnist.config.AppConfiguration;
import com.bellvenue.experimental.mnist.dto.BackPropagateResult;
import com.bellvenue.experimental.mnist.dto.FeedForwardResult;
import com.bellvenue.experimental.mnist.dto.LayerMap;
import com.bellvenue.experimental.mnist.dto.MnistObservation;
import com.google.common.collect.Lists;
import lombok.Data;
import org.apache.commons.math3.analysis.function.Sigmoid;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.random.RandomDataGenerator;
import org.apache.commons.math3.util.FastMath;
import org.springframework.stereotype.Component;

import javax.inject.Inject;
import java.util.List;

@Component
@Data
public class Processor {

    private LayerMap weights;

    private LayerMap biases;

    private List<Integer> layers;

    private MnistData trainingData;

    private MnistData testingData;

    private final DataLoader dataLoader;

    private final AppConfiguration configuration;

    @Inject
    Processor(DataLoader dataLoader, AppConfiguration configuration) {
        this.dataLoader = dataLoader;
        this.configuration = configuration;
    }

    void initialize() {
        try {
            trainingData = dataLoader.loadData(configuration.getTrainingImgDataFile(), configuration.getTrainingLabelDataFile());
            testingData = dataLoader.loadData(configuration.getTestingImgDataFile(), configuration.getTestingLabelDataFile());
        } catch (Exception e) {
            e.printStackTrace();
        }
        layers = Lists.newArrayList(trainingData.getImageVectorSize(), configuration.getHiddenLayerNeurons(), configuration.getOutputLayerNeurons());
        weights = generateRandomWeights(layers);
        biases = generateRandomBiases(layers);
    }

    FeedForwardResult feedForward(MnistObservation observation) {
        LayerMap activations = new LayerMap();
        activations.put(0, new Array2DRowRealMatrix(observation.getPixelVector().getDataRef()));
        LayerMap zs = new LayerMap();
        for (int layerIndex = 1; layerIndex < layers.size(); layerIndex++) {
            RealMatrix layerWeights = weights.get(layerIndex);
            RealMatrix layerBiases = biases.get(layerIndex);
            int neuronsCurrentLayer = layers.get(layerIndex);
            double[] zsArray = new double[neuronsCurrentLayer];
            double[] activationsArray = new double[neuronsCurrentLayer];

            for (int neuronIndex = 0; neuronIndex < neuronsCurrentLayer; neuronIndex++) {
                double z = layerWeights.getRowVector(neuronIndex).dotProduct(activations.get(layerIndex - 1).getColumnVector(0)) + layerBiases.getEntry(neuronIndex, 0);
                zsArray[neuronIndex] = z;
                double activation = sigmoid(z);
                activationsArray[neuronIndex] = activation;
            }
            activations.put(layerIndex, new Array2DRowRealMatrix(activationsArray));
            zs.put(layerIndex, new Array2DRowRealMatrix(zsArray));
        }
        return FeedForwardResult.builder()
                .activations(activations)
                .zs(zs)
                .build();
    }

    BackPropagateResult backPropagate(FeedForwardResult feedForwardResult, MnistObservation observation) {
        LayerMap weightErrors = new LayerMap();
        LayerMap biasErrors = new LayerMap();

        RealMatrix delta = null;
        for (int layerIndex = layers.size() - 1; layerIndex > 0; layerIndex--) {
            if (layerIndex == layers.size() - 1) {
                RealVector costDerivative = costFuncDerivative(feedForwardResult, observation.getLabelVector(), layerIndex);
                RealVector sigmoidPrime = sigmoidPrime(feedForwardResult.getZs().get(layerIndex).getColumnVector(0));
                RealVector deltaVector = costDerivative.ebeMultiply(sigmoidPrime);
                delta = new Array2DRowRealMatrix(((ArrayRealVector) deltaVector).getDataRef());
                biasErrors.put(layerIndex, delta);
                RealMatrix activationsPrevLayer = feedForwardResult.getActivations().get(layerIndex - 1);
                RealMatrix weightError = delta.multiply(activationsPrevLayer.transpose());
                weightErrors.put(layerIndex, weightError);
                continue;
            }
            RealMatrix w = weights.get(layerIndex + 1);
            RealMatrix z = feedForwardResult.getZs().get(layerIndex);
            RealVector sigmoidPrime = sigmoidPrime(z.getColumnVector(0));
            RealMatrix intermediate = w.transpose().multiply(delta);
            RealVector deltaVector = intermediate.getColumnVector(0).ebeMultiply(sigmoidPrime);
            delta = new Array2DRowRealMatrix((((ArrayRealVector) deltaVector).getDataRef()));

            biasErrors.put(layerIndex, delta);
            RealMatrix activationsPrevLayer = feedForwardResult.getActivations().get(layerIndex - 1);
            RealMatrix weightError = delta.multiply(activationsPrevLayer.transpose());
            weightErrors.put(layerIndex, weightError);
        }
        return BackPropagateResult.builder()
                .biasErrors(biasErrors)
                .weightErrors(weightErrors)
                .build();
    }

    private RealVector costFuncDerivative(FeedForwardResult feedForwardResult, RealVector y, int layerIndex) {
        //quadratic function, partial derivative of C with respect to `a` (activation)
        return feedForwardResult.getActivations().get(layerIndex).getColumnVector(0).subtract(y);
    }

    private double sigmoid(double z) {
        return 1 / (1 + FastMath.exp(-z));
    }

    private RealVector sigmoidPrime(RealVector zs) {
        RealVector resultZ = zs.mapToSelf(new Sigmoid());
        return resultZ.ebeMultiply(resultZ.mapMultiply(-1).mapAdd(1)); // equivalent to sigmoid(z) * (1 - sigmoid(z))
    }

    private LayerMap generateRandomWeights(List<Integer> layers) {
        LayerMap weights = new LayerMap();
        RandomDataGenerator generator = new RandomDataGenerator();
        for (int layerIndex = 1; layerIndex < layers.size(); layerIndex++) {
            int neurons = layers.get(layerIndex);
            int weightsVectorSize = layers.get(layerIndex - 1);
            for (int neuronIndex = 0; neuronIndex < neurons; neuronIndex++) {
                double[] array = new double[weightsVectorSize];
                for (int i = 0; i < weightsVectorSize; i++) {
                    array[i] = generator.nextGaussian(configuration.getGaussianDistributionMean(), configuration.getGaussianDistributionStdDeviation());
                }
                weights.setRow(layerIndex, neuronIndex, array, neurons, weightsVectorSize);
            }
        }
        return weights;
    }

    private LayerMap generateRandomBiases(List<Integer> layers) {
        LayerMap biases = new LayerMap();
        RandomDataGenerator generator = new RandomDataGenerator();
        for (int layerIndex = 1; layerIndex < layers.size(); layerIndex++) {
            int neurons = layers.get(layerIndex);
            for (int neuronIndex = 0; neuronIndex < neurons; neuronIndex++) {
                double[] array = new double[1];
                array[0] = generator.nextGaussian(configuration.getGaussianDistributionMean(), configuration.getGaussianDistributionStdDeviation());
                biases.setRow(layerIndex, neuronIndex, array, neurons, 1);
            }
        }
        return biases;
    }

    LayerMap initWeightErrors() {
        LayerMap layerMap = new LayerMap();
        for (int layerIndex = 1; layerIndex < layers.size(); layerIndex++) {
            int neurons = layers.get(layerIndex);
            int weightsVectorSize = layers.get(layerIndex - 1);
            layerMap.put(layerIndex, new Array2DRowRealMatrix(neurons, weightsVectorSize));
        }
        return layerMap;
    }

    LayerMap initBiasErrors() {
        LayerMap biases = new LayerMap();
        for (int layerIndex = 1; layerIndex < layers.size(); layerIndex++) {
            int neurons = layers.get(layerIndex);
            biases.put(layerIndex, new Array2DRowRealMatrix(neurons, 1));
        }
        return biases;
    }

    int getIndexOfLargest(double[] array) {
        if (array == null || array.length == 0) return -1;
        int idx = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[idx]) idx = i;
        }
        return idx;
    }
}
