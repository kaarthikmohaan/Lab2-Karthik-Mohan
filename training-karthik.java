import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class TrainWineQualityModel {
    public static void main(String[] args) {
        SparkConf configuration = new SparkConf().setAppName("TrainWineQualityModel");
        try (JavaSparkContext context = new JavaSparkContext(configuration)) {
            SparkSession session = SparkSession.builder()
                                               .appName("TrainWineQualityModel")
                                               .getOrCreate();

            String csvPath = "s3a://your-s3-bucket/TrainingDataset.csv";
            Dataset<Row> rawData = session.read()
                                          .format("csv")
                                          .option("header", "true")
                                          .option("inferSchema", "true")
                                          .load(csvPath);

            String[] inputFeatures = new String[]{
                "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
                "pH", "sulphates", "alcohol"
            };

            VectorAssembler featureAssembler = new VectorAssembler()
                                                   .setInputCols(inputFeatures)
                                                   .setOutputCol("assembledFeatures");

            Dataset<Row>[] splitData = rawData.randomSplit(new double[]{0.8, 0.2}, 42);
            Dataset<Row> trainingDataset = splitData[0];
            Dataset<Row> validationDataset = splitData[1];

            LogisticRegression logisticRegression = new LogisticRegression()
                                                        .setMaxIter(10)
                                                        .setRegParam(0.3)
                                                        .setElasticNetParam(0.8)
                                                        .setLabelCol("quality")
                                                        .setFamily("multinomial");

            Pipeline trainingPipeline = new Pipeline()
                                            .setStages(new PipelineStage[]{featureAssembler, logisticRegression});

            PipelineModel trainedModel = trainingPipeline.fit(trainingDataset);

            Dataset<Row> validationResults = trainedModel.transform(validationDataset);
            MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                                                              .setLabelCol("quality")
                                                              .setPredictionCol("prediction")
                                                              .setMetricName("f1");
            double validationF1Score = evaluator.evaluate(validationResults);
            System.out.println("Validation F1 Score: " + validationF1Score);

            trainedModel.write().overwrite().save("s3a://your-s3-bucket/wineQualityModel");

            session.stop();
        }
    }
}
