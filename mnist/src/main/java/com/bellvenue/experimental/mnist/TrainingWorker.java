package com.bellvenue.experimental.mnist;

import com.bellvenue.experimental.mnist.config.AppConfiguration;
import com.bellvenue.experimental.mnist.dto.BackPropagateResult;
import com.bellvenue.experimental.mnist.dto.FeedForwardResult;
import com.bellvenue.experimental.mnist.dto.LayerMap;
import com.bellvenue.experimental.mnist.dto.MnistObservation;
import lombok.extern.slf4j.Slf4j;

import java.util.List;
import java.util.Set;

@Slf4j
public class TrainingWorker implements Runnable {

    private final AppConfiguration configuration;

    private final Processor processor;

    private final int batchStartIndex;

    private final int batchEndIndex;

    private Set<Integer> executionTracker;

    TrainingWorker(int batchStartIndex,
                   int batchEndIndex,
                   AppConfiguration configuration,
                   Processor processor,
                   Set<Integer> executionTracker) {
        this.batchStartIndex = batchStartIndex;
        this.batchEndIndex = batchEndIndex;
        this.processor = processor;
        this.configuration = configuration;
        this.executionTracker = executionTracker;
    }

    @Override
    public void run() {
        try {
            List<MnistObservation> trainingData = processor.getTrainingData().getData();

            LayerMap weightErrors = processor.initWeightErrors();
            LayerMap biasErrors = processor.initBiasErrors();

            for (int i = batchStartIndex; i < batchEndIndex; i++) {
                final MnistObservation observation = trainingData.get(i);
                FeedForwardResult feedForwardResult = processor.feedForward(observation);
                BackPropagateResult backPropagateResult = processor.backPropagate(feedForwardResult, observation);

                for (int layerIndex = 1; layerIndex < processor.getLayers().size(); layerIndex++) {
                    weightErrors.put(layerIndex, weightErrors.get(layerIndex).add(backPropagateResult.getWeightErrors().get(layerIndex)));
                    biasErrors.put(layerIndex, biasErrors.get(layerIndex).add(backPropagateResult.getBiasErrors().get(layerIndex)));
                }
            }

            double normalizedLearningRate = configuration.getLearningRate() / configuration.getMiniBatchSize();
            for (int layerIndex = 1; layerIndex < processor.getLayers().size(); layerIndex++) {
                processor.getWeights().put(layerIndex,
                        processor.getWeights().get(layerIndex).subtract(weightErrors.get(layerIndex).scalarMultiply(normalizedLearningRate))); //gradient descent
                processor.getBiases().put(layerIndex,
                        processor.getBiases().get(layerIndex).subtract(biasErrors.get(layerIndex).scalarMultiply(normalizedLearningRate)));    //gradient descent
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            executionTracker.remove(batchEndIndex);
        }
    }
}