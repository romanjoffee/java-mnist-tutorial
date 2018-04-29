package com.bellvenue.experimental.mnist;

import org.springframework.stereotype.Service;

import javax.inject.Inject;

@Service
public class MnistService {

    @Inject
    MnistService(MnistModel model) {
        try {
            model.run();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
