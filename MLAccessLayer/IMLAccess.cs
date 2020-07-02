using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Text;
using static Microsoft.ML.DataOperationsCatalog;

namespace MLAccessLayer
{
    public interface IMLAccess
    {
        public  TrainTestData LoadData(MLContext mlContext);
        public  ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet);
        public StringBuilder Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet);
        public StringBuilder UseModelWithSingleItem(MLContext mlContext, ITransformer model);
        public StringBuilder UseModelWithBatchItems(MLContext mlContext, ITransformer model);
        public string UseModelFromView(MLContext mlContext, ITransformer model, string sentimentText);
    }
}
