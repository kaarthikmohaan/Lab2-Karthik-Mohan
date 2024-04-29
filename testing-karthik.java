import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class PredictWineQuality {
    public static void main(String[] args) {
        SparkConf config = new SparkConf().setAppName("PredictWineQuality");
        try (JavaSparkContext context = new JavaSparkContext(config)) {
            SparkSession session = SparkSession.builder()
                                               .appName("PredictWineQuality")
                                               .getOrCreate();

            String modelPath = "s3a://your-s3-bucket/wineQualityModel";
            PipelineModel loadedModel = PipelineModel.load(modelPath);

            String testDataPath = "s3a://your-s3-bucket/TestDataset.csv";
            Dataset<Row> testDataset = session.read()
                                              .format("csv")
                                              .option("header", "true")
                                              .option("inferSchema", "true")
                                              .load(testDataPath);

            Dataset<Row> resultPredictions = loadedModel.transform(testDataset);

            MulticlassClassificationEvaluator scoreEvaluator = new MulticlassClassificationEvaluator()
                                                                  .setLabelCol("quality")
                                                                  .setPredictionCol("prediction")
                                                                  .setMetricName("f1");
            double testF1Score = scoreEvaluator.evaluate(resultPredictions);
            System.out.println("Test F1 Score: " + testF1Score);

            session.stop();
        }
    }
}
