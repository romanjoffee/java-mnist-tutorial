package com.bellvenue.experimental.mnist.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.task.TaskExecutor;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;

@Configuration
@ConfigurationProperties(prefix = "mnist")
@Data
public class AppConfiguration {

    private String trainingImgDataFile;

    private String trainingLabelDataFile;

    private String testingImgDataFile;

    private String testingLabelDataFile;

    private int hiddenLayerNeurons;

    private int outputLayerNeurons;

    private int miniBatchSize;

    private double learningRate;

    private double gaussianDistributionMean;

    private double gaussianDistributionStdDeviation;

    private int epochCount;


    private int workerPoolCoreSize;

    private int workerPoolMaxSize;

    private int workerKeepAliveSeconds;

    private int workerQueueCapacity;

    private String workerThreadNamePrefix;

    @Bean
    public TaskExecutor threadPoolTaskExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(workerPoolCoreSize);
        executor.setMaxPoolSize(workerPoolMaxSize);
        executor.setThreadNamePrefix(workerThreadNamePrefix);
        executor.setQueueCapacity(workerQueueCapacity);
        executor.setKeepAliveSeconds(workerKeepAliveSeconds);
        executor.initialize();
        return executor;
    }
}

