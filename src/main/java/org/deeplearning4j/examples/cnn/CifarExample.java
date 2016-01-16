package org.deeplearning4j.examples.cnn;

import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.input.PortableDataStream;
import org.canova.image.loader.CifarLoader;
import org.canova.spark.functions.data.FilesAsBytesFunction;
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
import org.deeplearning4j.spark.canova.CanovaByteDataSetFunction;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 */
public class CifarExample {
    protected static final Logger log = LoggerFactory.getLogger(CifarExample.class);

    protected static final int HEIGHT = 32;
    protected static final int WIDTH = 32;
    protected static final int CHANNELS = 3;
    protected static final int outputNum = CifarLoader.NUM_LABELS;
    protected static final int numTrainSamples = 5000; // CifarLoader.NUM_TRAIN_IMAGES;
    protected static final int numTestSamples = 5000; //CifarLoader.NUM_TEST_IMAGES;
    protected static int batchSize = 250;
    protected static int iterations = 5;
    protected static int seed = 123;

    public static void main(String[] args) throws Exception {

        Nd4j.dtype = DataBuffer.Type.DOUBLE;

        int listenerFreq = batchSize;
        List<String> labels = new CifarLoader().getLabels();
        int nEpochs = 1;

        // Setup SparkContext
        SparkConf sparkConf = new SparkConf()
                .setMaster("local[*]");
        sparkConf.setAppName("LFW");
        sparkConf.set(SparkDl4jMultiLayer.AVERAGE_EACH_ITERATION, String.valueOf(true));
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        log.info("Load data....");
        JavaPairRDD<String,PortableDataStream> sparkData = sc.binaryFiles(CifarLoader.TRAINPATH.toString());
        JavaPairRDD<Text, BytesWritable> filesAsBytes = sparkData.mapToPair(new FilesAsBytesFunction());
        JavaRDD<DataSet> train = filesAsBytes.map(new CanovaByteDataSetFunction(0, CifarLoader.NUM_LABELS, batchSize, numTrainSamples, CifarLoader.BYTEFILELEN));

        sparkData = sc.binaryFiles(CifarLoader.TESTPATH.toString());
        filesAsBytes = sparkData.mapToPair(new FilesAsBytesFunction());
        JavaRDD<DataSet> test = filesAsBytes.map(new CanovaByteDataSetFunction(0, CifarLoader.NUM_LABELS, batchSize, numTestSamples, CifarLoader.BYTEFILELEN));

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
            sparkNetwork.fitDataSet(train);
            System.out.println("----- Epoch " + i + " complete -----");
        }

        log.info("Eval model...");
        Evaluation evalActual = sparkNetwork.evaluate(test, labels, false);
        log.info(evalActual.stats());



        log.info("****************Example finished********************");


    }


}
