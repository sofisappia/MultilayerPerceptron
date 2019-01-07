Layer layer = new Layer(3,5);

void setup() {
  size(1200,400);
  
}

void draw() {
  background(0, 100);
  layer.printWeights();
}

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
    this.weights = new float[neuronsPreviousLayer][neuronsLayer];
    randomSeed(2); // busco arreglo jxi
    for(int i=0; i < neuronsPreviousLayer; i++){ // columnas
      for(int j=0; j < neuronsLayer; j++){ // filas
        this.weights[i][j] = randomGaussian();
        //print(this.weights[j][i]);
      }
    }
   }
    void printWeights(){
      for(int i=0; i < 3; i++){
        for(int j=0; j < 5; j++){
          println(this.weights[i][j]);
        }
      }
    }
  }
