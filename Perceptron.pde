import Jama.*;
import papaya.*;

//https://www.bogotobogo.com/python/python_Neural_Networks_Backpropagation_for_XOR_using_one_hidden_layer.php
//https://github.com/Frixoe/xor-neural-network/blob/master/XOR-Net-Notebook.ipynb
//https://www.uow.edu.au/~markus/teaching/CSCI323/Lecture_MLP.pdf
//https://aimatters.wordpress.com/2016/01/11/solving-xor-with-a-neural-network-in-python/ (?

// MLP(int tempInputLayerUnits, int tempHiddenLayerUnits, int tempHiddenLayers, int tempOutputLayerUnits)
MLP mlp = new MLP(2,5,2,1);
float[][] X = {{0,0},{0,1},{1,0},{1,1}};
float[][] y = {{0},{1},{1},{0}};

void setup() {
  size(1200,400);
  mlp.compile(0.01, 1000);
  mlp.train(X,y);
  float[][] pred = mlp.predict(X);
  Mat.print(pred,2);
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
     ArrayList<float[][]> dW;
     for(int i =0; i<epochs; i++){
       activations = this.forward(X);
       dW = this.backPropagation(activations, y);
       this.updateWeights(dW);
     }
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
     }
       return activationOutputs; 
   }
   ArrayList<float[][]> backPropagation(ArrayList<float[][]> activations, float[][] y){
     // https://medium.com/@erikhallstrm/backpropagation-from-the-beginning-77356edf427d  
     
     //https://www.uow.edu.au/~markus/teaching/CSCI323/Lecture_MLP.pdf
     
     ArrayList<float[][]> deltas = new ArrayList<float[][]>();
     ArrayList<float[][]> deltaW = new ArrayList<float[][]>();
     // output layer
     float[][] di;
     float[][] dWij;
     // hidden layers
     float[][] dj;
     float[][] dWjk;
     
     // output layer  
     di = subtract(Mat.transpose(y), activations.get(activations.size()-1)); 
     di = Mat.dotMultiply(di, sigmoid(activations.get(activations.size()-1),true));// delta of the output layer
     deltas.add(di);
     
     dWij = scalarMultiply(learningRate, di); 
     dWij = Mat.multiply(activations.get(activations.size()-2), Mat.transpose(di));
     deltaW.add(Mat.transpose(dWij));   
    
     // hidden layer 
     for(int i=layers.size()-1; i>0; i--){
       // dj = (Wij.T * di) .* f'(aj)
       dj = Mat.multiply(Mat.transpose(layers.get(i).getWeights()), deltas.get(deltas.size()-1));
       dj = Mat.dotMultiply(dj, sigmoid(activations.get(i-1), true));
       deltas.add(dj);
       
       dWjk = scalarMultiply(learningRate, dj);
       if(i == 1){         
         dWjk = Mat.multiply(dWjk, X);
         deltaW.add(dWjk);
       }else{ 
         dWjk = Mat.multiply(dWjk, Mat.transpose(activations.get(i-2)));
         deltaW.add(dWjk);
       }
     }
     return deltaW;  
   }
   
   void updateWeights(ArrayList<float[][]> dW){
     for(int i=0; i<layers.size(); i++){
       layers.get(i).adjustWeights(dW.get(dW.size()-1-i));
     }
   
   }
   float[][] predict(float[][] X){
     ArrayList<float[][]> outputs = new ArrayList<float[][]>();
     outputs = this.forward(X);
     return outputs.get(outputs.size()-1); 
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
      }
    }
   }
   float[][] getWeights(){ 
     return weights; 
   }
   
   void adjustWeights(float[][] dW){
     this.weights = Mat.sum(this.weights, dW);
   }
  }
//--------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------
// Métodos auxiliares
  
float[][] dotProduct(float[][] A, float[][] B, boolean columns){ 
      // columns: boolean to know if data is arranged in columns or not
      int ARows = A.length; // m1 rows length
      int BRows = B.length;    // m2 rows length
      int ACols = A[0].length;
      int BCols = A[0].length;

      if((ACols != BCols) || (ARows != BRows)){
          println("Error de tamaño");
          return null; // matrix multiplication is not possible
      }else{
        if(columns){
          float[][] D = new float[ARows][1];
          for(int i=0; i<ARows; i++){
            D[i][0] = dotProduct(A[i], B[i]);
          } return D;
        }else{
          float[][] D = new float[1][ACols];
          for(int i=0; i<ACols; i++){
            D[0][i] = dotProduct(Mat.transpose(A)[i], Mat.transpose(B)[i]);
          }  return D;          
          }
      }
}


float dotProduct(float[] A, float[] B){ 
      // columns: boolean to know if data is arranged in columns or not
      int Alen = A.length; // m1 rows length
      int Blen = B.length;    // m2 rows length
      if(Alen != Blen){
          println("Error de tamaño");
          return 0; 
      }else{
        float D = 0;
          for(int i=0; i<Alen; i++){
              D += A[i] * B[i];
          }
          return D;          
      }  
}

float[][] sigmoid(float[][] z, boolean derv){
  // Función que calcula la función de activación sigmoidea
  // y su derivada
  float[][] y = new float[z.length][z[0].length];
  if(derv==true){ // cálculo de la derivada
      float[][] sigm = sigmoid(z, false);
      y = Mat.dotMultiply(sigm, subtract(1, sigm));
   // }
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

float[][] subtract(float a, float[][] B){
  int rowsB = B.length;
  int colsB = B[0].length;
  float[][] C = new float[rowsB][colsB];
  for(int i=0; i<rowsB; i++){
    for(int j=0; j<colsB; j++){
        C[i][j] = a - B[i][j];  
      }
   }
   return C;
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
