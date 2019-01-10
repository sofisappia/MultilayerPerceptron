import Jama.*;
import papaya.*;

//https://www.bogotobogo.com/python/python_Neural_Networks_Backpropagation_for_XOR_using_one_hidden_layer.php
//https://github.com/Frixoe/xor-neural-network/blob/master/XOR-Net-Notebook.ipynb
//https://www.uow.edu.au/~markus/teaching/CSCI323/Lecture_MLP.pdf
//https://aimatters.wordpress.com/2016/01/11/solving-xor-with-a-neural-network-in-python/ (?

MLP mlp = new MLP(2,3,2,1);
float[][] X = {{1,2},{2,3},{3,4}};
float[][] y = {{0},{1},{0}};

void setup() {
  size(1200,400);
  mlp.compile(0.01, 10);
  mlp.train(X,y);
  //mlp.forward(X); 
  
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
  ArrayList<LayerConnection> layers = new ArrayList<LayerConnection>();
  
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
    layers.add(new LayerConnection(inputLayerUnits,hiddenLayerUnits)); //1st layer
    for(int i=1; i<hiddenLayers; i++){ // Hidden layers
      layers.add(new LayerConnection(hiddenLayerUnits,hiddenLayerUnits));
    }
    // output layer
    layers.add(new LayerConnection(hiddenLayerUnits, outputLayerUnits));
  //  print(layers.size()); // Tamaño de las capas ocultas  
  }
  
   
   void train(float[][] X, float[][] y){
     ArrayList<float[][]> activations = new ArrayList<float[][]>();
     activations = this.forward(X);
     mlp.backPropagation(activations, y);
     //Mat.print(activations.get(activations.size()-1),3);
   }
   
   ArrayList<float[][]> forward(float[][] X){
     float[][] hi = Mat.transpose(X); // potenciales postsinápticos temporales
     float[][] Vi = new float[0][0]; // salida después de aplicar función de activación
     float[][] W;
     ArrayList<float[][]> activationOutputs = new ArrayList<float[][]>(); 
     
     for(int i=0; i<layers.size(); i++){
       W = layers.get(i).getWeights();
       hi = Mat.multiply(W, hi);
       Vi = sigmoid(hi, false);
       activationOutputs.add(Vi);
      // Mat.print(Vi,2);
       //println();
     }
       return activationOutputs; 
   }
   void backPropagation(ArrayList<float[][]> a, float[][] y){
     //https://medium.com/@erikhallstrm/backpropagation-from-the-beginning-77356edf427d
     
     //https://www.uow.edu.au/~markus/teaching/CSCI323/Lecture_MLP.pdf
     
     ArrayList<float[][]> deltas = new ArrayList<float[][]>();
     float[][] di = subtract(Mat.transpose(y), a.get(a.size()-1)); // delta of the output layer
     float[][] dWij = scalarMultiply(learningRate, Mat.dotMultiply(di, a.get(a.size()-1))); 
     
 /*    Mat.print(dWij,5);
     println();
     deltas.add(Mat.dotMultiply(di, sigmoid(a.get(a.size()-1),true)));
    

     float[][] Wjk = layers.get(layers.size()-1).getWeights();
     float[][] dj = deltas.get(deltas.size()-1);

     //Mat.print(matrixDotProduct(Wjk, dj),2);
     println();
     println();
     Mat.print(sigmoid(a.get(a.size()-2),true),2);
    // Mat.dotMultiply(Mat.dotMultiply(layers.get(layers.size()-1).getWeights(), deltas.get(deltas.size()-1)), sigmoid(a.get(a.size()-2),true));
     
     
*/
     
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
    this.weights = new float[nLayer][nPrevious];
    randomSeed(2); // busco arreglo jxi
    for(int i=0; i < nLayer; i++){ // columnas
      for(int j=0; j < nPrevious; j++){ // filas
        this.weights[i][j] = randomGaussian();
        //print(this.weights[j][i]);
      }
    }
   // Mat.print(weights,0);
   // println();
   }
   float[][] getWeights(){ 
     return weights; 
   }
   
   void adjustWeights(float[][] W){
     weights = W;
   }
  }
//--------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------
// Métodos auxiliares
  
float[] matrixDotProduct(float[][] A, float[][] B){ 
      int ARowLength = A.length; // m1 rows length
      int BRowLength = B.length;    // m2 rows length
      int AColLength = A[0].length;
      int BColLength = A[0].length;
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

float[][] sigmoid(float[][] z, boolean derv){
  // Función que calcula la función de activación sigmoidea
  // y su derivada
  float[][] y = new float[z.length][z[0].length];
  if(derv==true){ // cálculo de la derivada
    for(int i=0; i<z.length; i++){
      for(int j=0; j<z[0].length; j++){
        y[i][j] = z[i][j] * (1 - z[i][j]);
      }
    }
  }
  else{ // cálculo de la función sigmoidea
    for(int i=0; i<z.length; i++){
      for(int j=0; j<z[0].length; j++){      
        y[i][j] = 1 / (1 + exp(-z[i][j]));
      }
    }
  }
  return y;
 }
 
float[][] subtract(float[][] A, float[][] B){
  int rowsA = A.length;
  int colsA = A[0].length;
  int rowsB = B.length;
  int colsB = B[0].length;
  float[][] C = new float[rowsA][colsA];
  
 /* print("rows A: ", rowsA);
  print(" rows B: ", rowsB);
  print(" cols A: ", colsA);
  println(" cols B: ", colsB);*/
  if((rowsA != rowsB) || (colsA != colsB)){
    print("Dimensiones no coinciden");
    return null;
  }else{
    for(int i=0; i<rowsA; i++){
      for(int j=0; j<colsA; j++){
        C[i][j] = A[i][j] - B[i][j];  
      }
    }
    return C;
  }
}

float[][] scalarMultiply(float a, float[][] A){
  int rowsA = A.length;
  int colsA = A[0].length;
  float[][] B = new float[rowsA][colsA];
  for(int i=0; i<rowsA; i++){
    for(int j=0; j<colsA; j++){
      B[i][j] = a * A[i][j];
    }
  }
  return B;
}
