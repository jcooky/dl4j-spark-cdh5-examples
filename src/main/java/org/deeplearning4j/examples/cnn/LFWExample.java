package org.deeplearning4j.examples.cnn;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.canova.image.loader.LFWLoader;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * Labeled Faces in the Wild is a data set created by Erik Learned-Miller, Gary Huang, Aruni RoyChowdhury,
 * Haoxiang Li, Gang Hua. This is used to study unconstrained face recognition. There are over 13K images.
 * Each face has been labeled with the name of the person pictured.
 *
 * 5749 unique classes (different people)
 * 1680 people have 2+ photos
 *
 * References:
 * General information is at http://vis-www.cs.umass.edu/lfw/. Architecture partially based on DeepFace:
 * http://mmlab.ie.cuhk.edu.hk/pdf/YiSun_CVPR14.pdf
 *
 * Note: this is a sparse dataset with only 1 example for many of the faces; thus, performance is low.
 * Ideally train on a larger dataset like celebs to get params and/or generate variations of the image examples.
 *
 * Currently set to only use the subset images, names starting with A.
 * Switch to NUM_LABELS & NUM_IMAGES and set subset to false to use full dataset.
 */

public class LFWExample {

    protected static final Logger log = LoggerFactory.getLogger(LFWExample.class);

    protected static final int HEIGHT = 40; // original is 250
    protected static final int WIDTH = 40;
    protected static final int CHANNELS = 3;
    protected static final int outputNum = LFWLoader.NUM_LABELS;
    protected static final int numSamples = 50; //LFWLoader.SUB_NUM_IMAGES - 4;
    protected static int batchSize = 10;
    protected static int numBatches = numSamples/batchSize;
    protected static int iterations = 5;
    protected static int seed = 123;
    protected static boolean useSubset = false;

    public static void main(String[] args) throws Exception {
        Nd4j.dtype = DataBuffer.Type.DOUBLE;

        int splitTrainNum =(int) (batchSize * 0.8);
        int nTrain = (int) (numSamples * 0.8);
        int nTest = numSamples - nTrain;
        int listenerFreq = batchSize;
        int nEpochs = 1;

        // Setup SparkContext
        SparkConf sparkConf = new SparkConf()
                .setMaster("local[*]");
        sparkConf.setAppName("LFW");
        sparkConf.set(SparkDl4jMultiLayer.AVERAGE_EACH_ITERATION, String.valueOf(true));
//        conf.set(SparkDl4jMultiLayer.ACCUM_GRADIENT, String.valueOf(true));
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        log.info("Load data....");
        DataSetIterator lfw = new LFWDataSetIterator(batchSize, numSamples, new int[]{HEIGHT, WIDTH, CHANNELS}, outputNum, useSubset, new Random(seed));
        List<String> labels = lfw.getLabels();

        List<DataSet> train = new ArrayList<>(nTrain);
        List<DataSet> test = new ArrayList<>(nTest);

        int c = 0;
        while(lfw.hasNext()){
            DataSet ds = lfw.next();
            SplitTestAndTrain next = ds.splitTestAndTrain(splitTrainNum, new Random(seed));
            train.add(next.getTrain());
            test.add(next.getTest());
        }

        JavaRDD<DataSet> trainData = sc.parallelize(train);
        JavaRDD<DataSet> testData = sc.parallelize(test);


        log.info("Build model....");
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .activation("relu")
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(0.01)
                .momentum(0.9)
                .regularization(true)
                .updater(Updater.ADAGRAD)
                .useDropConnect(true)
                .list(6)
                .layer(0, new ConvolutionLayer.Builder(4, 4)
                        .name("cnn1")
                        .nIn(CHANNELS)
                        .stride(1, 1)
                        .nOut(20)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .name("pool1")
                        .build())
                .layer(2, new ConvolutionLayer.Builder(3, 3)
                        .name("cnn2")
                        .stride(1, 1)
                        .nOut(40)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .name("pool2")
                        .build())
                .layer(2, new ConvolutionLayer.Builder(3, 3)
                        .name("cnn3")
                        .stride(1, 1)
                        .nOut(60)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .name("pool3")
                        .build())
                .layer(2, new ConvolutionLayer.Builder(2, 2)
                        .name("cnn3")
                        .stride(1, 1)
                        .nOut(80)
                        .build())
                .layer(4, new DenseLayer.Builder()
                        .name("ffn1")
                        .nOut(160)
                        .dropOut(0.5)
                        .build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
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
            sparkNetwork.fitDataSet(trainData);
            System.out.println("----- Epoch " + i + " complete -----");
        }

        log.info("Eval model...");
        Evaluation evalActual = sparkNetwork.evaluate(testData, labels, false);
        log.info(evalActual.stats());



        log.info("****************Example finished********************");


    }
}