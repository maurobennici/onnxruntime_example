using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace onnx_iris
{
    public class Run
    {
        public int Evaluate()
        {
            // Official link
            // https://onnxruntime.ai/docs/get-started/with-csharp.html#create-method-for-inference

            // OpenVINO Linux and Windows
            // https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html#install

            // Get path to model to create inference session.
            //var modelPath = @"E:\onnx_iris\iris.onnx";
            var modelPath = @"E:\onnx_iris\sequential.onnx";

            // Create an InferenceSession from the Model Path.
            var session = new InferenceSession(modelPath);

            // Create tensor
            Tensor<float> inputTensor = new DenseTensor<float>(new[] { 1, 4 });

            // put the value
            inputTensor[0, 0] = 1.0F; // sepal_length			
            inputTensor[0, 1] = 4.0F; // sepal_width
            inputTensor[0, 2] = 1.3F; // petal_length
            inputTensor[0, 3] = 3.0F; // petal_width

            // Create input data for session.
            var input = new List<NamedOnnxValue>{
                NamedOnnxValue.CreateFromTensor<float>("input", inputTensor)
            };

            // Run session and send input data in to get inference output.
            var output = session.Run(input);

            // Get the input 0
            var inferenceResult = output.First().AsTensor<float>().ToArray();

            // Get the index
            var result = Array.IndexOf(inferenceResult, inferenceResult.Max());
            return result;
        }
    }
}
