using Microsoft.ML;
using System;
using static Microsoft.ML.DataOperationsCatalog;
using DTO;
using System.IO;
using Microsoft.ML.Data;
using System.Collections.Generic;
using System.Text;
using System.Reflection;

namespace MLAccessLayer
{
    public class MLAccess : IMLAccess
    {
        //static readonly string textDirectory = Path.Combine(Environment.CurrentDirectory + "/Data");
        static readonly string textDirectory = Path.Combine(Directory.GetCurrentDirectory() + @"\Data");
        static readonly string textFullPath = Path.GetFullPath(textDirectory);
        //static readonly string fullPath = Path.GetFullPath(Path.Combine(System.AppContext.BaseDirectory, @"..\..\..\..\"+ System.Reflection.Assembly.GetExecutingAssembly().GetName().Name + @"\Data"));
        //static readonly string _dataPath = Path.Combine(fullPath, "yelp_labelled.txt");
        static readonly string _dataPath = Path.Combine(textFullPath, "yelp_labelled.txt");

        public TrainTestData LoadData(MLContext mlContext)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);
            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            return splitDataView;
        }

        public ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
        {
            var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
                                                     .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

            #region Comments#1
            //The FeaturizeText() method converts the text column(SentimentText) into a 
            //numeric key type Features column used by the machine learning algorithm 
            //and adds it as a new dataset column

            //The SdcaLogisticRegressionBinaryTrainer is your classification training algorithm. 
            //This is appended to the estimator and accepts the featurized 
            //SentimentText (Features) and the Label input parameters to learn from the historic data.
            #endregion

            //Console.WriteLine("=============== Create and Train the Model ===============");
            var model = estimator.Fit(splitTrainSet);
            //The Fit() method trains your model by transforming the dataset and applying the training
            return model;
        }

        public StringBuilder Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            StringBuilder builder = new StringBuilder();
            builder.AppendLine("=============== Evaluating Model accuracy with Test data===============");
            IDataView predictions = model.Transform(splitTestSet);
            //Transform() method makes predictions for multiple provided input rows of a test dataset.

            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
            #region Comments#2
            //Once you have the prediction set(predictions), the Evaluate() method assesses the model, 
            //which compares the predicted values with the actual Labels in the test dataset and 
            //returns a CalibratedBinaryClassificationMetrics object on how the model is performing.
            #endregion


            builder.AppendLine("Model quality metrics evaluation");
            builder.AppendLine("--------------------------------");
            builder.AppendLine($"Accuracy: {metrics.Accuracy:P2}");
            builder.AppendLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
            builder.AppendLine($"F1Score: {metrics.F1Score:P2}");
            builder.AppendLine("=============== End of model evaluation ===============");

            #region Comments#3
            //The Accuracy metric gets the accuracy of a model, which is the proportion of correct 
            //predictions in the test set.

            //The AreaUnderRocCurve metric indicates how confident the model is correctly 
            //classifying the positive and negative classes. You want the AreaUnderRocCurve to 
            //be as close to one as possible.

            //The F1Score metric gets the model's F1 score, which is a measure of balance 
            //between precision and recall. You want the F1Score to be as close to one as possible.
            #endregion
            return builder;
        }

        public StringBuilder UseModelWithSingleItem(MLContext mlContext, ITransformer model)
        {
            StringBuilder builder = new StringBuilder();

            PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

            //The PredictionEngine allows you to perform a prediction on a single instance of data

            SentimentData sampleStatement = new SentimentData
            {
                SentimentText = "This place is very good"
            };
            var resultPrediction = predictionFunction.Predict(sampleStatement);
            builder.AppendLine("=============== Prediction Test of model with a single sample and test dataset ===============");
            builder.AppendLine($"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");
            builder.AppendLine("=============== End of Predictions ===============");
            return builder;
        }

        public StringBuilder UseModelWithBatchItems(MLContext mlContext, ITransformer model)
        {
            StringBuilder builder = new StringBuilder();
            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText = "This was a horrible meal"
                },
                new SentimentData
                {
                    SentimentText = "I love this spaghetti."
                }
            };

            IDataView batchComments = mlContext.Data.LoadFromEnumerable(sentiments);

            IDataView predictions = model.Transform(batchComments);

            // Use model to predict whether comment data is Positive (1) or Negative (0).
            IEnumerable<SentimentPrediction> predictedResults = mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);

            builder.AppendLine("=============== Prediction Test of loaded model with multiple samples ===============");

            foreach (SentimentPrediction prediction in predictedResults)
                builder.AppendLine($"Sentiment: {prediction.SentimentText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");

            builder.AppendLine("=============== End of predictions ===============");
            return builder;
        }

        public string UseModelFromView(MLContext mlContext, ITransformer model, string sentimentText)
        {
            PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

            SentimentData sentiment = new SentimentData()
            {
                SentimentText = sentimentText
            };

            var resultPrediction = predictionFunction.Predict(sentiment);
            
            var viewResultString = $"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ";
            return viewResultString;
        }
    }
}