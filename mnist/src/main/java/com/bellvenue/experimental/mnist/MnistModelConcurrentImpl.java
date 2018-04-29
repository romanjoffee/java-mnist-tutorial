package com.bellvenue.experimental.mnist;

import com.bellvenue.experimental.mnist.config.AppConfiguration;
import com.bellvenue.experimental.mnist.dto.FeedForwardResult;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.springframework.context.annotation.Profile;
import org.springframework.core.task.TaskExecutor;
import org.springframework.stereotype.Component;

import javax.inject.Inject;
import java.time.Duration;
import java.time.Instant;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

@Component
@Profile("concurrent")
@Slf4j
public class MnistModelConcurrentImpl extends MnistModel {

    private final TaskExecutor taskExecutor;

    private final AppConfiguration configuration;

    private final Processor processor;

    @Inject
    MnistModelConcurrentImpl(AppConfiguration configuration, TaskExecutor taskExecutor, Processor processor) {
        this.configuration = configuration;
        this.taskExecutor = taskExecutor;
        this.processor = processor;
        this.processor.initialize();
    }

    private Set<Integer> executionTracker;

    @Override
    void trainModel() throws Exception {
        setStart(Instant.now());
        executionTracker = Collections.synchronizedSet(new HashSet<>());

        MnistData trainingData = processor.getTrainingData();
        Collections.shuffle(trainingData.getData());
        if (trainingData.getData().size() % configuration.getMiniBatchSize() != 0) {
            throw new Exception("Training set should be divisible by mini batch size");
        }
        for (int index = 0; index < trainingData.getData().size(); index += configuration.getMiniBatchSize()) {
            executionTracker.add(index + configuration.getMiniBatchSize());
            taskExecutor.execute(new TrainingWorker(index,
                    index + configuration.getMiniBatchSize(),
                    configuration,
                    processor,
                    executionTracker)
            );
        }
    }

    @Override
    void testModel(int epoch) {
        while (executionTracker.size() > 0) {
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                //ignore
            }
        }
        setEnd(Instant.now());
        log.info("Duration of training: " + Duration.between(getStart(), getEnd()));

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
        log.info("Epoch {}: Correct guesses {}", epoch, correctGuesses.get());
        if (correctGuesses.get() >= 9500) {
            log.info("95% accuracy reached");
            System.exit(1);
        }
    }
}
