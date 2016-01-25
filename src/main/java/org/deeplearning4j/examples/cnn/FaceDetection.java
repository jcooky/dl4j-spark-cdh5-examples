package org.deeplearning4j.examples.cnn;

import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.input.PortableDataStream;
import org.apache.spark.storage.StorageLevel;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.split.LimitFileSplit;
import org.canova.image.loader.BaseImageLoader;
import org.canova.image.recordreader.ImageRecordReader;
import org.canova.spark.functions.data.FilesAsBytesFunction;
import org.canova.spark.functions.data.RecordReaderBytesFunction;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingModelSaver;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
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
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.canova.api.writable.Writable;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;

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
    public final static int NUM_LABELS = 10;
    public final static int WIDTH = 80; // size varies
    public final static int HEIGHT = 80;
    public final static int CHANNELS = 3;

    public static void main(String[] args) {
        Nd4j.dtype = DataBuffer.Type.DOUBLE;

        boolean appendLabels = true;
        int numExamples = 100;
        int batchSize = 1;
        int numBatches = NUM_IMAGES/batchSize;

        int iterations = 5;
        int epochs = 3;
        int seed = 123;
        double[] splitTrainTest = new double[]{ .8, .2};
        int listenerFreq = batchSize;
        int nTrain = (int) (numExamples * 0.8);
        int nTest = numExamples - nTrain;

        // Setup SparkContext
        SparkConf sparkConf = new SparkConf()
                .setMaster("local[*]");
        sparkConf.setAppName("FaceDetection");
        sparkConf.set("spak.executor.memory", "4g");
        sparkConf.set("spak.driver.memory", "4g");
        sparkConf.set(SparkDl4jMultiLayer.AVERAGE_EACH_ITERATION, String.valueOf(true));
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        log.info("Load data....");

//        String[] tags = new File(BaseImageLoader.BASE_DIR, "thumbnails_features_deduped_sample").list(new FilenameFilter() {
//            @Override
//            public boolean accept(File dir, String name) {
//                return dir.isDirectory();
//            }
//        });
//        List<String> labels = Arrays.asList(tags);

        // Load as sequence files onto RDD
        // man / woman classification
//        File mainPath = new File(BaseImageLoader.BASE_DIR, "gender_class/*");
//        List<String> labels = Arrays.asList(new String[]{"man", "woman"});

        // classification by name
        File mainPath = new File(BaseImageLoader.BASE_DIR, "ms_sample");
        List<String> labels = Arrays.asList(new String[]{"aaron_carter", "martina_hingis", "michelle_obama", "adam_brody"});

//        File mainPath = new File(BaseImageLoader.BASE_DIR, "thumbnails_features_deduped_sample/*");
//        JavaPairRDD<String,PortableDataStream> sparkData = sc.binaryFiles(mainPath.toString());
//        JavaPairRDD<Text,BytesWritable> filesAsBytes = sparkData.mapToPair(new FilesAsBytesFunction());
//        RecordReaderBytesFunction rrFunc = new RecordReaderBytesFunction(recordReader);
//        JavaRDD<Collection<Writable>> data = filesAsBytes.map(rrFunc);
//        JavaRDD<DataSet> fullData = data.map(new CanovaDataSetFunction(-1, NUM_LABELS, false));
//        fullData.cache();
//        JavaRDD<DataSet>[] trainTestSplit = fullData.randomSplit(splitTrainTest);


        // Alternative load
//        File mainPath = new File(BaseImageLoader.BASE_DIR, "thumbnails_features_deduped_sample");

        RecordReader recordReader = new ImageRecordReader(WIDTH, HEIGHT, CHANNELS, appendLabels, labels);
        try {
            recordReader.initialize(new FileSplit(mainPath, BaseImageLoader.ALLOWED_FORMATS, new Random(123)));
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, -1, NUM_LABELS);
        List<DataSet> allData = new ArrayList<>(numExamples);
        while(dataIter.hasNext()){
            allData.add(dataIter.next());
        }

        Collections.shuffle(allData, new Random(seed));
        Iterator<DataSet> iter = allData.iterator();
        List<DataSet> train = new ArrayList<>(nTrain);
        List<DataSet> test = new ArrayList<>(nTest);

        int c = 0;
        while(iter.hasNext()){
            if(c++ < nTrain) train.add(iter.next());
            test.add(iter.next());
        }

        JavaRDD<DataSet> sparkDataTrain = sc.parallelize(train);
        sparkDataTrain.persist(StorageLevel.MEMORY_ONLY());


        log.info("Build model....");
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .activation("relu")
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(5e-2)
                .momentum(0.9)
                .regularization(true)
                .l2(1e-3)
                .updater(Updater.NESTEROVS)
                .useDropConnect(true)
//                .list(9)
//                .layer(0, new ConvolutionLayer.Builder(3, 3)
//                        .name("cnn1")
//                        .nIn(CHANNELS)
//                        .stride(1, 1)
//                        .padding(1, 1)
//                        .nOut(64)
//                        .build())
//                .layer(1, new ConvolutionLayer.Builder(3, 3)
//                        .name("cnn2")
//                        .stride(1, 1)
//                        .padding(1, 1)
//                        .nOut(64)
//                        .build())
//                .layer(2, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
//                        .name("pool1")
//                        .stride(1, 1)
//                        .build())
//                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
//                        .name("pool2")
//                        .stride(1, 1)
//                        .build())
//                .layer(4, new ConvolutionLayer.Builder(3, 3)
//                        .name("cnn3")
//                        .stride(1, 1)
//                        .padding(1, 1)
//                        .nOut(64)
//                        .build())
//                .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
//                        .name("pool3")
//                        .stride(1, 1)
//                        .build())
//                .layer(6, new DenseLayer.Builder()
//                        .name("ffn1")
//                        .nOut(400)
//                        .dropOut(0.5)
//                        .build())
//                .layer(7, new DenseLayer.Builder()
//                        .name("ffn1")
//                        .nOut(200)
//                        .dropOut(0.5)
//                        .build())
//                .layer(8, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                        .nOut(NUM_LABELS)
//                        .activation("softmax")
//                        .build())
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
//        sparkNetwork.fitDataSet(trainTestSplit[0].coalesce(5));
        for(int i = 0; i < epochs; i++){
            sparkNetwork.fitDataSet(sparkDataTrain);
        }
        sparkDataTrain.unpersist();

        log.info("Evaluate model....");
//        Evaluation evalActual = sparkNetwork.evaluate(trainTestSplit[1].coalesce(5), labels);

        JavaRDD<DataSet> sparkDataTest = sc.parallelize(test);
        sparkDataTest.persist(StorageLevel.MEMORY_ONLY());
        Evaluation evalActual = sparkNetwork.evaluate(sparkDataTest, labels);
        log.info(evalActual.stats());

        sparkDataTest.unpersist();
//        fullData.unpersist();

        log.info("****************Example finished********************");
    }

    // Starting code to rename and move files
//    public fixFileName(File path){
//        Files.walkFileTree(path, new SimpleFileVisitor<path>()
//        {
//            @Override
//            public FileVisitResult visitFile(Path filePath, BasicFileAttributes attrs) throws IOException
//            {
//                def fileName = filePath.fileName;
//                def dirPath = filePath.parent;
//                def newFileName = fileName.toString().trim().toLowerCase().replace(" ","_");
//                def newFilePath = Paths.get(dirPath.toString(), newFileName);
//                Files.move(filePath, newFilePath)
//                return FileVisitResult.CONTINUE;
//            }
//        });
//    }


}
