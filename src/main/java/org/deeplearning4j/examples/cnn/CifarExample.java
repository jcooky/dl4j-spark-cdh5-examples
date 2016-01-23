package org.deeplearning4j.examples.cnn;

import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.DoubleFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.input.PortableDataStream;
import org.canova.image.loader.CifarLoader;
import org.canova.spark.functions.data.FilesAsBytesFunction;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.spark.canova.CanovaByteDataSetFunction;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.Tuple2;

import java.util.*;

/**
* CIFAR-10 is an image dataset created by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. The dataset inculdes 60K
* tiny RGB images sized 32 x 32 pixels covering 10 classes. There are 50K training images and 10K test images.
*
* Use this example to run cifar-10.
*
* Reference: https://www.cs.toronto.edu/~kriz/cifar.html
* Dataset url: https://s3.amazonaws.com/dl4j-distribution/cifar-small.bin
 */

public class CifarExample {
    protected static final Logger log = LoggerFactory.getLogger(CifarExample.class);

    protected static final int HEIGHT = 32;
    protected static final int WIDTH = 32;
    protected static final int CHANNELS = 3;
    protected static final int outputNum = CifarLoader.NUM_LABELS;
    protected static int batchSize = 5;
    protected static int numBatches = 6;
    protected static final int numTrainSamples = batchSize * numBatches;// CifarLoader.NUM_TRAIN_IMAGES;
    protected static final int numTestSamples = numTrainSamples; // CifarLoader.NUM_TEST_IMAGES;
    protected static int iterations = 5;
    protected static int seed = 123;

    public static void main(String[] args) throws Exception {

        Nd4j.dtype = DataBuffer.Type.DOUBLE;

        int listenerFreq = batchSize;
        List<String> labels = new CifarLoader().getLabels();
        int nEpochs = 1;

        // Setup SparkContext
        SparkConf sparkConf = new SparkConf()
                .setMaster("local[6]");
        sparkConf.setAppName("Cifar");
        sparkConf.set(SparkDl4jMultiLayer.AVERAGE_EACH_ITERATION, String.valueOf(true));
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        log.info("Load train data....");
        JavaPairRDD<String,PortableDataStream> sparkData = sc.binaryFiles(CifarLoader.TRAINPATH.toString());
        JavaPairRDD<Text, BytesWritable> filesAsBytes = sparkData.mapToPair(new FilesAsBytesFunction());
        JavaPairRDD<Double, DataSet> trainData = filesAsBytes.mapToPair(new CanovaByteDataSetFunction(0, CifarLoader.NUM_LABELS, batchSize, CifarLoader.BYTEFILELEN));

//        JavaRDD<DataSet> train = filesAsBytes.map(new CanovaByteDataSetFunction(0, CifarLoader.NUM_LABELS, batchSize, numTrainSamples, CifarLoader.BYTEFILELEN));

        JavaRDD<DataSet> train = trainData.map(new Function<Tuple2<Double, DataSet>, DataSet>() {
            @Override
            public DataSet call(Tuple2<Double,DataSet> ds) throws Exception {
                return ds._2();
            }
        });

        train.cache();

        JavaDoubleRDD numExamplesPerRDD = trainData.mapToDouble(new DoubleFunction<Tuple2<Double,DataSet>>(){
            @Override
            public double call(Tuple2<Double,DataSet> ds) throws Exception {
                return ds._1();
            }
        });

        double totalCaptured = numExamplesPerRDD.sum();


        log.info("Build model....");
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .activation("relu")
                .weightInit(WeightInit.XAVIER) // consider standard distribution with std .05
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(0.01)
                .momentum(0.9)
                .regularization(true)
                .l2(0.04)
                .updater(Updater.NESTEROVS)
                .useDropConnect(true)
                .list(10)
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .name("cnn1")
                        .nIn(CHANNELS)
                        .stride(1, 1)
                        .padding(2, 2)
                        .nOut(32)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3})
                        .name("pool1")
                        .build())
                .layer(2, new LocalResponseNormalization.Builder(3, 5e-05, 0.75).build())
                .layer(3, new ConvolutionLayer.Builder(5, 5)
                        .name("cnn2")
                        .stride(1, 1)
                        .padding(2, 2)
                        .nOut(32)
                        .build())
                .layer(4, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3})
                        .name("pool2")
                        .build())
                .layer(5, new LocalResponseNormalization.Builder(3, 5e-05, 0.75).build())
                .layer(6, new ConvolutionLayer.Builder(5, 5)
                        .name("cnn3")
                        .stride(1, 1)
                        .padding(2, 2)
                        .nOut(64)
                        .build())
                .layer(7, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3})
                        .name("pool3")
                        .build())
                .layer(8, new DenseLayer.Builder()
                        .name("ffn1")
                        .nOut(250)
                        .dropOut(0.5)
                        .build())
                .layer(9, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false)

                .cnnInputSize(HEIGHT, WIDTH, CHANNELS);

        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));

        //Create Spark multi layer network from configuration
        SparkDl4jMultiLayer sparkNetwork = new SparkDl4jMultiLayer(sc, network);


        log.info("Train model...");

        for (int i = 0; i < nEpochs; i++) {
            sparkNetwork.fitDataSet(train);
            System.out.println("----- Epoch " + i + " complete -----");
        }

        train.unpersist();

        sparkData = sc.binaryFiles(CifarLoader.TESTPATH.toString());
        filesAsBytes = sparkData.mapToPair(new FilesAsBytesFunction());
        JavaPairRDD<Double, DataSet> testData = filesAsBytes.mapToPair(new CanovaByteDataSetFunction(0, CifarLoader.NUM_LABELS, batchSize, CifarLoader.BYTEFILELEN));

        JavaRDD<DataSet> test = testData.map(new Function<Tuple2<Double, DataSet>, DataSet>() {
            @Override
            public DataSet call(Tuple2<Double,DataSet> ds) throws Exception {
                return ds._2();
            }
        });

        test.cache();

        log.info("Eval model...");
        Evaluation evalActual = sparkNetwork.evaluate(test, labels);
        log.info(evalActual.stats());
        List<DataSet> dst = test.collect();

        test.unpersist();

        log.info("****************Example finished********************");


    }


}
