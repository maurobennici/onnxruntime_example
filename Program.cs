
// See https://aka.ms/new-console-template for more information
var r = new onnx_iris.Run();
var tag = r.Evaluate();
Console.Out.WriteLine($"result = {tag}");
Console.In.ReadLine();
