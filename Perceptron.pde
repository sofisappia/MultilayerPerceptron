import Jama.*;
import papaya.*;

MLP mlp = new MLP(2,3,2,1);

void setup() {
  size(1200,400);
  mlp.compile(0.01, 10);
  
}

/*void draw() {
  background(0, 100);

  
}*/

class MLP{
  int inputLayerUnits;
  int hiddenLayerUnits;
  int hiddenLayers;
  int outputLayerUnits;
  
  // learning parameters
  float learningRate;
  int epochs;
  
  MLP(int tempInputLayerUnits, int tempHiddenLayerUnits, int tempHiddenLayers, int tempOutputLayerUnits){
    inputLayerUnits = tempInputLayerUnits;
    hiddenLayerUnits = tempHiddenLayerUnits;
    hiddenLayers = tempHiddenLayers;
    outputLayerUnits = tempOutputLayerUnits;
  }
  
  void compile(float tempLearningRate, int tempEpochs){
    learningRate = tempLearningRate;
    epochs = tempEpochs;
    
    // Crea arreglo de las conexiones entre cada capa
    ArrayList<LayerConnection> layers = new ArrayList<LayerConnection>();
    
    layers.add(new LayerConnection(inputLayerUnits,hiddenLayerUnits)); //1st layer
    for(int i=0; i<hiddenLayers; i++){ // Hidden layers
      layers.add(new LayerConnection(hiddenLayerUnits,hiddenLayerUnits));
    }
    // output layer
    layers.add(new LayerConnection(hiddenLayerUnits,outputLayerUnits));
    
    print(layers.size()); // Tamaño de las capas ocultas
  //  layers.get(0).printWeights();
   
  }
  
  float[][] sigmoid(float[][] z, boolean derv){
    // Función que calcula la función de activación sigmoidea
    // y su derivada
    float[][] y = new float[z.length][z[1].length];
    if(derv==true){ // cálculo de la derivada
      for(int i=0; i<z.length; i++){
        for(int j=0; j<z[1].length; j++){
          y[i][j] = z[i][j] * (1 - z[i][j]);
        }
      }
    }
    else{ // cálculo de la función sigmoidea
      for(int i=0; i<z.length; i++){
        for(int j=0; j<z[1].length; j++){      
          y[i][j] = 1 / (1 + exp(-z[i][j]));
        }
      }
    }
    return y;
   }
   
   void train(double[][] X, int[] y){
     this.forward(X);
   }
   
   void forward(double[][] X){
     
   }
   void backPropagation(){
   }
   void predict(){
   }
}

class LayerConnection {
  // Crea las conexiones entre las diferentes capas de la red
  //double[][] weights;
  float[][] weights;
  int nPrevious;
  int nLayer;
  //Matrix W;
  
  LayerConnection(int neuronsInPreviousLayer, int neuronsInLayer){
    nPrevious = neuronsInPreviousLayer;
    nLayer = neuronsInLayer;
    // weights initialization
    this.weights = new float[nPrevious][nLayer];
    randomSeed(2); // busco arreglo jxi
    for(int i=0; i < nPrevious; i++){ // columnas
      for(int j=0; j < nLayer; j++){ // filas
        this.weights[i][j] = randomGaussian();
        //print(this.weights[j][i]);
      }
    }
    // convert to Matrix
   // W = new Matrix(weights);
   }
   float[][] getWeights(){ 
     return weights; 
   }
   /* void printWeights(){
      for(int i=0; i < this.nPrevious; i++){
        for(int j=0; j < this.nLayer; j++){
          println(this.weights[i][j]);
        }
      }
      println(this.weights.length);
    }*/
  }
  
float[] dotProduct(float[][] A, float[][] B){
      int ARowLength = A.length; // m1 rows length
      int BRowLength = B.length;    // m2 rows length
      int AColLength = A[1].length;
      int BColLength = A[1].length;
      if( (ARowLength != BRowLength) || (AColLength != BColLength)){
        println("Error de tamanio");
        return null; // matrix multiplication is not possible
      }
      else{
        float[][] C = Mat.dotMultiply(A,B);
        float[] D = new float[ARowLength];
        for(int i=0; i<ARowLength; i++){
          for(int j=0; j<AColLength; j++){
            D[i] += C[i][j];
          }
        }
        return D;
      }  
}
