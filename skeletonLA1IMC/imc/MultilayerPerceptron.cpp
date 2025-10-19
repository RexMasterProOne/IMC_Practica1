/*********************************************************************
* File  : MultilayerPerceptron.cpp
* Date  : 2020
*********************************************************************/

#include "MultilayerPerceptron.h"

#include "util.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <limits>
#include <math.h>


using namespace imc;
using namespace std;
using namespace util;

// ------------------------------
// Constructor: Default values for all the parameters
MultilayerPerceptron::MultilayerPerceptron()
{
    nOfLayers = 0;
    layers = nullptr;
    eta = 0.1;
    mu = 0.9;
}

// ------------------------------
// Allocate memory for the data structures
// nl is the number of layers and npl is a vector containing the number of neurons in every layer
// Give values to Layer* layers
int MultilayerPerceptron::initialize(int nl, int npl[]) {
    // -------------------------------------------------------------
    // Paso 0: Liberamos cualquier memoria previa
    // -------------------------------------------------------------
    // Esto garantiza que si se llama initialize() varias veces,
    // no queden residuos de estructuras anteriores.
    freeMemory();

    // -------------------------------------------------------------
    // Paso 1: Asignar número de capas y reservar array principal
    // -------------------------------------------------------------
    nOfLayers = nl;
    layers = new Layer[nOfLayers];

    // -------------------------------------------------------------
    // Paso 2: Crear cada capa
    // -------------------------------------------------------------
    for (int h = 0; h < nOfLayers; h++) {
        layers[h].nOfNeurons = npl[h];
        layers[h].neurons = new Neuron[npl[h]];

        // Inicializamos cada neurona con punteros nulos y valores base
        for (int j = 0; j < npl[h]; j++) {
            layers[h].neurons[j].w = nullptr;
            layers[h].neurons[j].deltaW = nullptr;
            layers[h].neurons[j].lastDeltaW = nullptr;
            layers[h].neurons[j].wCopy = nullptr;
            layers[h].neurons[j].out = 0.0;
            layers[h].neurons[j].delta = 0.0;
        }

        // -------------------------------------------------------------
        // Paso 3: Reservar pesos solo para las capas > 0 (no entrada)
        // -------------------------------------------------------------
        if (h > 0) {
            int prevNeurons = npl[h - 1] + 1; // +1 para el bias

            for (int j = 0; j < npl[h]; j++) {
                // Reserva de memoria protegida (inicializada a 0.0)
                layers[h].neurons[j].w          = new double[prevNeurons]();
                layers[h].neurons[j].deltaW     = new double[prevNeurons]();
                layers[h].neurons[j].lastDeltaW = new double[prevNeurons]();
                layers[h].neurons[j].wCopy      = new double[prevNeurons]();

                // Aunque el operador () ya inicializa a 0.0,
                // dejamos el bucle por claridad.
                for (int k = 0; k < prevNeurons; k++) {
                    layers[h].neurons[j].w[k] = 0.0;
                    layers[h].neurons[j].deltaW[k] = 0.0;
                    layers[h].neurons[j].lastDeltaW[k] = 0.0;
                    layers[h].neurons[j].wCopy[k] = 0.0;
                }
            }
        }
    }

    // -------------------------------------------------------------
    // Paso 4: Comprobación opcional de consistencia
    // -------------------------------------------------------------
    // Aseguramos que la capa 0 (entrada) no tenga pesos asignados.
    for (int j = 0; j < layers[0].nOfNeurons; j++) {
        layers[0].neurons[j].w = nullptr;
        layers[0].neurons[j].deltaW = nullptr;
        layers[0].neurons[j].lastDeltaW = nullptr;
        layers[0].neurons[j].wCopy = nullptr;
    }

    return 0; // Todo correcto
}


// ------------------------------
// DESTRUCTOR: free memory
MultilayerPerceptron::~MultilayerPerceptron() {
	freeMemory();
}


// ------------------------------
// Free memory for the data structures
void MultilayerPerceptron::freeMemory() {
   if (layers != nullptr) {
        for (int h = 1; h < nOfLayers; h++) {
            for (int j = 0; j < layers[h].nOfNeurons; j++) {
                delete[] layers[h].neurons[j].w;
                delete[] layers[h].neurons[j].deltaW;
                delete[] layers[h].neurons[j].lastDeltaW;
                delete[] layers[h].neurons[j].wCopy;
            }
            delete[] layers[h].neurons;
        }
        delete[] layers[0].neurons;
        delete[] layers;
        layers = nullptr;
    }
    nOfLayers = 0;
}

// ------------------------------
// Feel all the weights (w) with random numbers between -1 and +1
void MultilayerPerceptron::randomWeights() {
    for (int h = 1; h < nOfLayers; h++) {
        for (int j = 0; j < layers[h].nOfNeurons; j++) {
            int prevNeurons = layers[h-1].nOfNeurons + 1;
            for (int k = 0; k < prevNeurons; k++) {
                layers[h].neurons[j].w[k] = util::randomDouble(-1.0, 1.0);
            }
        }
    }
}

// ------------------------------
// Feed the input neurons of the network with a vector passed as an argument
void MultilayerPerceptron::feedInputs(double* input) {
    for (int i = 0; i < layers[0].nOfNeurons; i++) {
        layers[0].neurons[i].out = input[i];
    }
}

// ------------------------------
// Get the outputs predicted by the network (out vector the output layer) and save them in the vector passed as an argument
void MultilayerPerceptron::getOutputs(double* output)
{
    Layer& outputLayer = layers[nOfLayers - 1];
    for (int j = 0; j < outputLayer.nOfNeurons; j++) {
        output[j] = outputLayer.neurons[j].out;
    }
}

// ------------------------------
// Make a copy of all the weights (copy w in wCopy)
void MultilayerPerceptron::copyWeights() {
    for (int h = 1; h < nOfLayers; h++) {
        for (int j = 0; j < layers[h].nOfNeurons; j++) {
            int prevNeurons = layers[h-1].nOfNeurons + 1;
            for (int k = 0; k < prevNeurons; k++) {
                layers[h].neurons[j].wCopy[k] = layers[h].neurons[j].w[k];
            }
        }
    }
}

// ------------------------------
// Restore a copy of all the weights (copy wCopy in w)
void MultilayerPerceptron::restoreWeights() {
	for (int h = 1; h < nOfLayers; h++) {
		for (int j = 0; j < layers[h].nOfNeurons; j++) {
			int prevNeurons = layers[h-1].nOfNeurons + 1;
			for (int k = 0; k < prevNeurons; k++) {
				layers[h].neurons[j].w[k] = layers[h].neurons[j].wCopy[k];
			}
		}
	}
}

// ------------------------------
// Calculate and propagate the outputs of the neurons, from the first layer until the last one -->-->
void MultilayerPerceptron::forwardPropagate() {
    double net = 0.0;
    for (int h = 1; h < nOfLayers; h++) {
        int prevNeurons = layers[h-1].nOfNeurons; 
        for (int j = 0; j < layers[h].nOfNeurons; j++) {
            net = layers[h].neurons[j].w[0]; 
            for (int k = 1; k <= prevNeurons; k++) {
                net += layers[h].neurons[j].w[k] * layers[h-1].neurons[k-1].out;
            }
            layers[h].neurons[j].out = 1.0 / (1.0 + exp(-net));
        }
    }
}

// ------------------------------
// Obtain the output error (MSE) of the out vector of the output layer wrt a target vector and return it
double MultilayerPerceptron::obtainError(double* target) {
    double patternError = 0.0;
    Layer& outputLayer = layers[nOfLayers - 1];
    for (int j = 0; j < outputLayer.nOfNeurons; j++) {
        double error = target[j] - outputLayer.neurons[j].out;
        patternError += error * error;
    }
    return patternError / outputLayer.nOfNeurons;
}


// ------------------------------
// Backpropagate the output error wrt a vector passed as an argument, from the last layer to the first one <--<--
void MultilayerPerceptron::backpropagateError(double* target) {
    // -------------------------------------------------------------
    // Paso 1: Calcular los deltas (δ) para la capa de salida
    // -------------------------------------------------------------
    // δ_Hj = g'(net_Hj) * (out_Hj - target_j)
    // donde g'(net) = out * (1 - out) si g es la sigmoide.
    //
    // -------------------------------------------------------------
    int h = nOfLayers - 1;
    Layer& outputLayer = layers[h];
    for (int j = 0; j < outputLayer.nOfNeurons; j++) {
        double out = outputLayer.neurons[j].out; // Salida real de la neurona j
        double diff = out - target[j];           // Diferencia salida - objetivo
        // Derivada de la sigmoide: out * (1 - out)
        outputLayer.neurons[j].delta = diff * out * (1.0 - out);
    }

    // -------------------------------------------------------------
    // Paso 2: Retropropagar los deltas hacia atrás (capas ocultas)
    // -------------------------------------------------------------
    // δ_hj = g'(net_hj) * Σ_i( w_(h+1)ij * δ_(h+1)i )
    //
    // Esto significa que el error de una neurona oculta depende del
    // error de todas las neuronas a las que está conectada en la capa siguiente.
    // -------------------------------------------------------------
    for (h = nOfLayers - 2; h >= 1; h--) {
        Layer& currentLayer = layers[h];     // Capa actual (oculta)
        Layer& nextLayer = layers[h + 1];    // Capa siguiente
        for (int j = 0; j < currentLayer.nOfNeurons; j++) {
            double out = currentLayer.neurons[j].out;
            double sumDeltaW = 0.0;

            // Suma ponderada de los deltas de la siguiente capa
            // (nota: +1 porque el índice 0 del vector w es el bias)
            for (int i = 0; i < nextLayer.nOfNeurons; i++) {
                sumDeltaW += nextLayer.neurons[i].w[j + 1] * nextLayer.neurons[i].delta;
            }

            // Derivada de la sigmoide aplicada al valor de salida
            currentLayer.neurons[j].delta = out * (1.0 - out) * sumDeltaW;
        }
    }
}



// ------------------------------
// Accumulate the changes produced by one pattern and save them in deltaW
void MultilayerPerceptron::accumulateChange() {
	for (int h = 1; h < nOfLayers; h++) {
		for (int j = 0; j < layers[h].nOfNeurons; j++) {
			int prevNeurons = layers[h-1].nOfNeurons + 1;
			for (int k = 0; k < prevNeurons; k++) {
				double out_i = (k == 0) ? 1.0 : layers[h-1].neurons[k-1].out;
				layers[h].neurons[j].deltaW[k] += layers[h].neurons[j].delta * out_i;
			}
		}
	}
}

// ------------------------------
// Update the network weights, from the first layer to the last one
void MultilayerPerceptron::weightAdjustment() {
    // Recorremos todas las capas ocultas y de salida (empezando en 1 porque la capa 0 es de entrada)
    for (int h = 1; h < nOfLayers; h++) {
        int prevNeuronsCount = layers[h - 1].nOfNeurons + 1; // +1 por el sesgo (bias)

        // Recorremos todas las neuronas de la capa actual
        for (int j = 0; j < layers[h].nOfNeurons; j++) {
            double delta_j = layers[h].neurons[j].delta; // Error local (δ_j) de la neurona j

            // Recorremos todos los pesos que conectan la capa anterior con esta neurona
            for (int k = 0; k < prevNeuronsCount; k++) {

                // out_i representa la salida del nodo previo (o 1.0 si es el bias)
                double out_i = (k == 0) ? 1.0 : layers[h - 1].neurons[k - 1].out;

                // ------------------------------
                // GRADIENT TERM: η * δ_j * out_i
                // ------------------------------
                // Este término representa la contribución del gradiente (descenso del error)
                // Cuanto mayor es δ_j o out_i, más se ajusta este peso.
                double gradientTerm = eta * delta_j * out_i;

                // ------------------------------
                // MOMENT TERM: μ * lastDeltaW
                // ------------------------------
                // El término de momento (μ) añade una fracción del cambio anterior al nuevo ajuste.
                // Esto suaviza la trayectoria del aprendizaje y evita oscilaciones.
                double momentTerm = mu * layers[h].neurons[j].lastDeltaW[k];

                // ------------------------------
                // TOTAL DELTAW: combinación de ambos
                // ------------------------------
                double deltaW = gradientTerm + momentTerm;

                // ------------------------------
                // ACTUALIZACIÓN DE PESO:
                // w = w - Δw
                // ------------------------------
                // Restamos porque queremos minimizar el error (descenso del gradiente).
                layers[h].neurons[j].w[k] -= deltaW;

                // ------------------------------
                // GUARDAMOS EL ÚLTIMO CAMBIO (para usar en la próxima iteración con el momento)
                // ------------------------------
                layers[h].neurons[j].lastDeltaW[k] = deltaW;
            }
        }
    }
}


// ------------------------------
// Print the network, i.e. all the weight matrices
void MultilayerPerceptron::printNetwork() {
    // Imprime pesos de cada capa (excepto capa 0 de entrada)
	for (int h = 1; h < nOfLayers; h++) {
		cout << "Layer " << h << " weights:" << endl;
		for (int j = 0; j < layers[h].nOfNeurons; j++) {
			cout << " Neuron " << j << ": ";
			int prevNeurons = layers[h-1].nOfNeurons + 1; // +1 por el sesgo
			for (int k = 0; k < prevNeurons; k++) {
				cout << layers[h].neurons[j].w[k] << " ";
			}
			cout << endl;
		}
		cout << endl;
	}
}

// ------------------------------
// Perform an epoch: forward propagate the inputs, backpropagate the error and adjust the weights
// input is the input vector of the pattern and target is the desired output vector of the pattern
void MultilayerPerceptron::performEpochOnline(double* input, double* target) {
    // Online mode: process a single pattern (forward, backprop, update weights)
    feedInputs(input);
    forwardPropagate();

    // Calcular deltas
    backpropagateError(target);

    // Ajuste de pesos (en este diseño weightAdjustment usa delta y lastDeltaW)
    weightAdjustment();
}

// ------------------------------
// Perform an online training for a specific trainDataset
void MultilayerPerceptron::trainOnline(Dataset* trainDataset) {
	int i;
	for(i=0; i<trainDataset->nOfPatterns; i++){
		performEpochOnline(trainDataset->inputs[i], trainDataset->outputs[i]);
	}
}

// ------------------------------
// Test the network with a dataset and return the MSE
double MultilayerPerceptron::test(Dataset* testDataset) {
    double totalError = 0.0;
    int nOfPatterns = testDataset->nOfPatterns;
    int nOfOutputs = testDataset->nOfOutputs;
    
    // Vector temporal para almacenar las salidas de la red
    double* obtainedOutputs = new double[nOfOutputs];

    // Recorrer patrones
    for (int i = 0; i < nOfPatterns; i++) {
        feedInputs(testDataset->inputs[i]);
        forwardPropagate();
        getOutputs(obtainedOutputs);

        for (int j = 0; j < nOfOutputs; j++) {
            double error = testDataset->outputs[i][j] - obtainedOutputs[j];
            totalError += error * error;
        }
    }

    delete[] obtainedOutputs;

    // Devolver MSE: totalError dividido entre (N_patrones * N_salidas)
    return totalError / (nOfPatterns * nOfOutputs);
}


// Optional - KAGGLE
// Test the network with a dataset and return the MSE
// Your have to use the format from Kaggle: two columns (Id y predictied)
void MultilayerPerceptron::predict(Dataset* pDatosTest)
{
	int i;
	int j;
	int numSalidas = layers[nOfLayers-1].nOfNeurons;
	double * obtained = new double[numSalidas];
	
	cout << "Id,Predicted" << endl;
	
	for (i=0; i<pDatosTest->nOfPatterns; i++){

		feedInputs(pDatosTest->inputs[i]);
		forwardPropagate();
		getOutputs(obtained);
		
		cout << i;

		for (j = 0; j < numSalidas; j++)
			cout << "," << obtained[j];
		cout << endl;

	}
}

// ------------------------------
// Run the traning algorithm for a given number of epochs, using trainDataset
// Once finished, check the performance of the network in testDataset
// Both training and test MSEs should be obtained and stored in errorTrain and errorTest
void MultilayerPerceptron::runOnlineBackPropagation(Dataset * trainDataset, Dataset * pDatosTest, int maxiter, double *errorTrain, double *errorTest)
{
    int countTrain = 0;
    // Random assignment of weights (starting point)
    randomWeights();
    double minTrainError = 0;
    int iterWithoutImproving = 0;
    double testError = 0;
    Dataset * newTrainDataset = NULL;
    // Learning
    do {
        trainOnline(trainDataset);
        double trainError = test(trainDataset);
        if(countTrain==0 || trainError < minTrainError){
            if( (minTrainError-trainError) > 0.00001)
                iterWithoutImproving = 0;
            else
                iterWithoutImproving++;
            minTrainError = trainError;
            copyWeights();
        }
        else
            iterWithoutImproving++;
        if(iterWithoutImproving==50){
            cout << "We exit because the training is not improving!!"<< endl;
            restoreWeights();
            countTrain = maxiter;
        }
        countTrain++;
        cout << "Iteration " << countTrain << "\t Training error: " << trainError  << endl;
    } while ( countTrain<maxiter );
    cout << "NETWORK WEIGHTS" << endl;
    cout << "===============" << endl;
    printNetwork();
    cout << "Desired output Vs Obtained output (test)" << endl;
    cout << "=========================================" << endl;
    for(int i=0; i<pDatosTest->nOfPatterns; i++){
        double* prediction = new double[pDatosTest->nOfOutputs];
        // Feed the inputs and propagate the values
        feedInputs(pDatosTest->inputs[i]);
        forwardPropagate();
        getOutputs(prediction);
        for(int j=0; j<pDatosTest->nOfOutputs; j++)
            cout << pDatosTest->outputs[i][j] << " -- " << prediction[j] << " ";
        cout << endl;
        delete[] prediction;
    }
    testError = test(pDatosTest);
    *errorTest=testError;
    *errorTrain=minTrainError;
    if(newTrainDataset != NULL) {
        // We delete the row vector of inputs and outputs but not every row (because
        // the belong to the original training dataset that we want to keep)
        delete[] newTrainDataset->inputs;
        delete[] newTrainDataset->outputs;
        delete newTrainDataset;
    }
}

// Optional Kaggle: Save the model weights in a textfile
bool MultilayerPerceptron::saveWeights(const char * archivo)
{
	// Object for writing the file
	ofstream f(archivo);

	if(!f.is_open())
		return false;

	// Write the number of layers and the number of layers in every layer
	f << nOfLayers;

	for(int i = 0; i < nOfLayers; i++)
		f << " " << layers[i].nOfNeurons;
	f << endl;

	// Write the weight matrix of every layer
	for(int i = 1; i < nOfLayers; i++)
		for(int j = 0; j < layers[i].nOfNeurons; j++)
			for(int k = 0; k < layers[i-1].nOfNeurons + 1; k++)
				f << layers[i].neurons[j].w[k] << " ";

	f.close();

	return true;

}


// Optional Kaggle: Load the model weights from a textfile
bool MultilayerPerceptron::readWeights(const char * archivo)
{
	// Object for reading a file
	ifstream f(archivo);

	if(!f.is_open())
		return false;

	// Number of layers and number of neurons in every layer
	int nl;
	int *npl;

	// Read number of layers
	f >> nl;

	npl = new int[nl];

	// Read number of neurons in every layer
	for(int i = 0; i < nl; i++)
		f >> npl[i];

	// Initialize vectors and data structures
	initialize(nl, npl);

	// Read weights
	for(int i = 1; i < nOfLayers; i++)
		for(int j = 0; j < layers[i].nOfNeurons; j++)
			for(int k = 0; k < layers[i-1].nOfNeurons + 1; k++)
				f >> layers[i].neurons[j].w[k];

	f.close();
	delete[] npl;

	return true;
}
