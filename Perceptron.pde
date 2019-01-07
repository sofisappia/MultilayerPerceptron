Layer layer = new Layer(2, 3);


class MLP{
  int inputLayerUnits;
  int hiddenLayerUnits;
  int hiddenLayers;
  int outputLayers;
  
  // learning parameters
  float learningRate;
  int epochs;
  
  MLP(int tempInputLayerUnits, int tempHiddenLayerUnits, int tempHiddenLayers, int tempOutputLayers){
  inputLayerUnits = tempInputLayerUnits;
  hiddenLayerUnits = tempHiddenLayerUnits;
  hiddenLayers = tempHiddenLayers;
  outputLayers = tempOutputLayers;
  }
  
  void compile(float tempLearningRate, int tempEpochs){
    learningRate = tempLearningRate;
    epochs = tempEpochs;
    // weights initialization
    
    
    
    
  }
 
}

class Layer {
  float[][] weights;
  
  Layer(int neuronsPreviousLayer, int neuronsLayer){
    randomSeed(2);
    for(int i=0; i < neuronsLayer; i++){
      for(int j=0; i < neuronsPreviousLayer; i++){
        weights[j][i] = randomGaussian();
      }
    }
  }

}
