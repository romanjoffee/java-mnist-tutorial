package com.bellvenue.experimental.mnist;

import com.bellvenue.experimental.mnist.config.AppConfiguration;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;

import javax.inject.Inject;
import java.time.Instant;

@Slf4j
@Data
abstract class MnistModel {

    @Inject
    private AppConfiguration configuration;

    @Inject
    private DataLoader dataLoader;

    @Inject
    private Processor processor;

    private Instant start;

    private Instant end;

    abstract void trainModel() throws Exception;

    abstract void testModel(int epoch);

    void run() throws Exception {
        for (int epoch = 0; epoch < configuration.getEpochCount(); epoch++) {
            trainModel();
            testModel(epoch);
        }
    }
}
