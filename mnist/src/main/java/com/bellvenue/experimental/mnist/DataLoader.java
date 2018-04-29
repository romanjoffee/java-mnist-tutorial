package com.bellvenue.experimental.mnist;

import com.bellvenue.experimental.mnist.dto.MnistObservation;
import com.google.common.collect.Lists;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Component;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.util.List;
import java.util.zip.GZIPInputStream;

@Component
public class DataLoader {

    MnistData loadData(String imageFileName, String labelFileName) throws Exception {
        MnistData mnistData = new MnistData();
        List<MnistObservation> observations = Lists.newArrayList();

        int numRows;
        int numCols;
        try (DataInputStream imagesInputStream = new DataInputStream(new FileInputStream(new ClassPathResource(imageFileName).getFile()));
             DataInputStream labelsInputStream = new DataInputStream(new FileInputStream(new ClassPathResource(labelFileName).getFile()))) {

            int magicNumber = labelsInputStream.readInt();
            if (magicNumber != 2049) {
                throw new Exception("Label file has wrong magic number: " + magicNumber + " (should be 2049)");
            }
            magicNumber = imagesInputStream.readInt();
            if (magicNumber != 2051) {
                throw new Exception("Image file has wrong magic number: " + magicNumber + " (should be 2051)");
            }
            mnistData.setLabelCount(labelsInputStream.readInt());
            mnistData.setImageCount(imagesInputStream.readInt());

            numRows = imagesInputStream.readInt();
            numCols = imagesInputStream.readInt();

            if (mnistData.getLabelCount() != mnistData.getImageCount()) {
                throw new Exception("mismatch of input data");
            }
            mnistData.setImageVectorSize(numCols * numRows);
            if (mnistData.getLabelCount() % mnistData.getImageVectorSize() == 0) {
                throw new Exception("mismatch of input data");
            }

            byte[] labelsData = new byte[mnistData.getLabelCount()];
            labelsInputStream.read(labelsData);
            byte[] imagesData = new byte[mnistData.getLabelCount() * mnistData.getImageVectorSize()];
            imagesInputStream.read(imagesData);

            int imgStart = 0;
            int imgEnd = mnistData.getImageVectorSize();
            for (int i = 0; i < mnistData.getLabelCount(); i++) {
                double[] pixelArray = new double[numRows * numCols];
                int idx = 0;
                for (int imgIndex = imgStart; imgIndex < imgEnd; imgIndex++) {
                    pixelArray[idx++] = (imagesData[imgIndex] & 0xff) / 255.0;
                }
                imgStart += mnistData.getImageVectorSize();
                imgEnd += mnistData.getImageVectorSize();

                int label = labelsData[i];
                double[] labelArray = new double[10];
                labelArray[label] = 1;

                observations.add(MnistObservation.builder()
                        .pixelVector(new ArrayRealVector(pixelArray))
                        .labelVector(new ArrayRealVector(labelArray))
                        .build());
            }
        }
        mnistData.setData(observations);
        return mnistData;
    }
}
