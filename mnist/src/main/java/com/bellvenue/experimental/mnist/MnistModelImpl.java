package com.bellvenue.experimental.mnist;

import com.bellvenue.experimental.mnist.config.AppConfiguration;
import com.bellvenue.experimental.mnist.dto.BackPropagateResult;
import com.bellvenue.experimental.mnist.dto.FeedForwardResult;
import com.bellvenue.experimental.mnist.dto.LayerMap;
import com.bellvenue.experimental.mnist.dto.MnistObservation;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

import javax.inject.Inject;
import java.time.Instant;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

@Component
@Slf4j
@Profile("general")
class MnistModelImpl extends MnistModel {

    private final AppConfiguration configuration;

    private final Processor processor;

    @Inject
    MnistModelImpl(AppConfiguration configuration, Processor processor) {
        this.configuration = configuration;
        this.processor = processor;
        this.processor.initialize();
    }

    @Override
    void trainModel() throws Exception {
        setStart(Instant.now());
        MnistData trainingData = processor.getTrainingData();
        Collections.shuffle(trainingData.getData());
        if (trainingData.getData().size() % configuration.getMiniBatchSize() != 0) {
            throw new Exception("Training set should be divisible by mini batch size");
        }
        for (int index = 0; index < processor.getTrainingData().getData().size(); index += configuration.getMiniBatchSize()) {
            processBatch(trainingData.getData(), index, index + configuration.getMiniBatchSize());
        }
        setEnd(Instant.now());
        //log.info("Duration of training: " + Duration.between(getStart(), getEnd()));
    }

    @Override
    void testModel(int epoch) {
        MnistData testData = processor.getTestingData();
        AtomicInteger correctGuesses = new AtomicInteger(0);
        testData.getData().forEach(observation -> {
            FeedForwardResult feedForwardResult = processor.feedForward(observation);
            ArrayRealVector activations = (ArrayRealVector) feedForwardResult.getActivations().get(processor.getLayers().size() - 1).getColumnVector(0);
            int i = processor.getIndexOfLargest(activations.getDataRef());
            if (observation.getLabelVector().getDataRef()[i] > 0) {
                correctGuesses.incrementAndGet();
            }
        });
        log.info("Epoch {}: Correct guesses {} of 10000", epoch, correctGuesses.get());
    }

    private void processBatch(List<MnistObservation> trainingData, int batchStartIndex, int batchEndIndex) {
        LayerMap weightErrors = processor.initWeightErrors();
        LayerMap biasErrors = processor.initBiasErrors();

        for (int i = batchStartIndex; i < batchEndIndex; i++) {
            MnistObservation observation = trainingData.get(i);
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
    }
}
