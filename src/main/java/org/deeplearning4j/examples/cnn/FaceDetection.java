package org.deeplearning4j.examples.cnn;

import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.python.SerDeUtil;
import org.apache.spark.input.PortableDataStream;
import org.canova.image.loader.BaseImageLoader;
import org.canova.image.loader.ImageByteBuffer;
import org.canova.image.recordreader.ImageRecordReader;
import org.canova.spark.functions.data.FilesAsBytesFunction;
import org.canova.spark.functions.data.RecordReaderBytesFunction;
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
import org.deeplearning4j.spark.canova.CanovaDataSetFunction;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.canova.api.writable.Writable;
import scala.Tuple2;

import java.io.File;
import java.io.FilenameFilter;
import java.util.*;

/**
 * MSRA-CFW Dataset of Celebrity Faces on the Web is a data set created by MicrosoftResearch.
 * This is based of of the thumbnails data set which is a smaller subset. It includes 2215 images
 * and 10 classifications with each image only including one face.
 *
 * More information and the data set can be found at: http://research.microsoft.com/en-us/projects/msra-cfw/
 *
 */

public class FaceDetection {
    protected static final Logger log = LoggerFactory.getLogger(FaceDetection.class);

    public final static int NUM_IMAGES = 2215; // some are 50 and others 700
    public final static int NUM_LABELS = 2;// 10;
    public final static int WIDTH = 50; // size varies
    public final static int HEIGHT = 50;
    public final static int CHANNELS = 3;

    public static void main(String[] args) {
        Nd4j.dtype = DataBuffer.Type.DOUBLE;

        boolean appendLabels = true;
        int numExamples = 50;
        int batchSize = 10;
        int numBatches = NUM_IMAGES/batchSize;

        int iterations = 1;
        int seed = 123;
        double[] splitTrainNum = new double[]{ .8, .2};
        int listenerFreq = batchSize;

        // Setup SparkContext
        SparkConf sparkConf = new SparkConf()
                .setMaster("local[*]");
        sparkConf.setAppName("FaceDetection");
        sparkConf.set("spak.executor.memory", "4g");
        sparkConf.set("spak.driver.memory", "4g");

        sparkConf.set(SparkDl4jMultiLayer.AVERAGE_EACH_ITERATION, String.valueOf(true));
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        log.info("Load data....");
        // man / woman classification
        File mainPath = new File(BaseImageLoader.BASE_DIR, "gender_class/*");
        List<String> labels = Arrays.asList(new String[]{"man", "woman"});

        // classification by name
//        File mainPath = new File(BaseImageLoader.BASE_DIR, "ms_sample/*");
//        List<String> labels = Arrays.asList(new String[]{"aaron_carter", "martina_hingis", "michelle_obama", "adam_brody"});

//        File mainPath = new File(BaseImageLoader.BASE_DIR, "thumbnails_features_deduped_sample/*");
//        String[] tags = new File(BaseImageLoader.BASE_DIR, "thumbnails_features_deduped_sample").list(new FilenameFilter() {
//            @Override
//            public boolean accept(File dir, String name) {
//                return dir.isDirectory();
//            }
//        });
//        List<String> labels = Arrays.asList(tags);

        JavaPairRDD<String,PortableDataStream> sparkData = sc.binaryFiles(mainPath.toString());
        JavaPairRDD<Text,BytesWritable> filesAsBytes = sparkData.mapToPair(new FilesAsBytesFunction());
        RecordReaderBytesFunction recordReader = new RecordReaderBytesFunction(new ImageRecordReader(WIDTH, HEIGHT, CHANNELS, appendLabels, labels));

        JavaRDD<Collection<Writable>> data = filesAsBytes.map(recordReader);
        JavaRDD<DataSet> fullData = data.map(new CanovaDataSetFunction(-1, NUM_LABELS, false));

        fullData.cache();
        JavaRDD<DataSet>[] trainTestSplit = fullData.randomSplit(splitTrainNum);


        // TODO coordinate readers and functions to load multiple types of datasets


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
                .list(11)
                .layer(0, new ConvolutionLayer.Builder(7, 7)
                        .name("cnn1")
                        .nIn(CHANNELS)
                        .stride(1, 1)
                        .nOut(48)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .name("pool1")
                        .build())
                .layer(2, new LocalResponseNormalization.Builder().build())
                .layer(3, new ConvolutionLayer.Builder(5, 5)
                        .name("cnn2")
                        .stride(1, 1)
                        .nOut(128)
                        .build())
                .layer(4, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .name("pool2")
                        .build())
                .layer(5, new LocalResponseNormalization.Builder().build())
                .layer(6, new ConvolutionLayer.Builder(3, 3)
                        .name("cnn3")
                        .stride(1, 1)
                        .nOut(192)
                        .build())
                .layer(7, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .name("pool3")
                        .build())
                .layer(8, new DenseLayer.Builder()
                        .name("ffn1")
                        .nOut(256)
                        .dropOut(0.5)
                        .build())
                .layer(9, new DenseLayer.Builder()
                        .name("ffn1")
                        .nOut(256)
                        .dropOut(0.5)
                        .build())
                .layer(10, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(NUM_LABELS)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false)
                .cnnInputSize(HEIGHT, WIDTH, CHANNELS);

        MultiLayerNetwork model = new MultiLayerNetwork(builder.build());
        model.init();
        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));

        //Create Spark multi layer network from configuration
        SparkDl4jMultiLayer sparkNetwork = new SparkDl4jMultiLayer(sc, model);


        log.info("Train model....");
//        sparkNetwork.fitDataSet(fullData);
        sparkNetwork.fitDataSet(trainTestSplit[0].coalesce(5));

        log.info("Evaluate model....");
        Evaluation evalActual = sparkNetwork.evaluate(trainTestSplit[1].coalesce(5), labels);
        log.info(evalActual.stats());

        fullData.unpersist();

        log.info("****************Example finished********************");


    }


}
