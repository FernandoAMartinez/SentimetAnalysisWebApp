using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using Microsoft.ML;
using MLAccessLayer;
using SentimentAnalysisWebApp.Models;
using System.Diagnostics;
using System.Text;
using SentimentAnalysisWebApp.Helpers;
using static Microsoft.ML.DataOperationsCatalog;

namespace SentimentAnalysisWebApp.Controllers
{
    public class HomeController : Controller
    {
        ConsoleViewModel viewModel = new ConsoleViewModel()
        {
            StringBuilder = new StringBuilder()
        };

        MLContext mlContext = new MLContext();

        private readonly ILogger<HomeController> _logger;

        private readonly IMLAccess MLAccess;
        //private static MLAccess mlAccess;
        //public static MLAccess MLAccess
        //{
            //get
            //{
                //if (mlAccess == null)
                    //mlAccess = new MLAccess();

                //return mlAccess;
            //}
        //}

        public HomeController(ILogger<HomeController> logger, IMLAccess access)
        {
            _logger = logger;
            MLAccess = access;
        }

        #region Views
        //public IActionResult Index() => View(); 
        public ActionResult<string> Privacy(string id) => GetPredictionFromModel(id);
        #endregion

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error() =>  View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });

        public ActionResult<ConsoleViewModel> Index()
        {

            TrainTestData splitDataView = MLAccess.LoadData(mlContext);
            viewModel.StringBuilder.Append("=============== Create and Train the Model ===============");
            ITransformer model = MLAccess.BuildAndTrainModel(mlContext, splitDataView.TrainSet);
            viewModel.StringBuilder.AppendLine("=============== End of training ===============");

            viewModel.StringBuilder.AppendLine(MLAccess.Evaluate(mlContext, model, splitDataView.TestSet).ToString());
            viewModel.StringBuilder.AppendLine(MLAccess.UseModelWithSingleItem(mlContext, model).ToString());
            //HttpContext.Session.SetComplexData("model", model);
            //HttpContext.Session.SetComplexData("mlContext", mlContext);
            return View(viewModel);
        }

        [HttpGet]
        [Route("Home/GetPredictionFromModel/{inputSentiment}")]
        public ActionResult<string> GetPredictionFromModel(string inputSentiment) 
        {
            TrainTestData splitDataView = MLAccess.LoadData(mlContext);
            ITransformer model = MLAccess.BuildAndTrainModel(mlContext, splitDataView.TrainSet);
            //ITransformer model = HttpContext.Session.GetComplexData<ITransformer>("model");
            //MLContext context = HttpContext.Session.GetComplexData<MLContext>("mlContext");
            return MLAccess.UseModelFromView(mlContext, model, inputSentiment);
        }
    }
}
