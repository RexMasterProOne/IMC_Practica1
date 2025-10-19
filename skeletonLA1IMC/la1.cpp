//============================================================================
// Introduction to computational models
// Name        : la1.cpp
// Author      : Pedro A. Gutiérrez
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <ctime>    
#include <cstdlib>  
#include <string.h>
#include <math.h>
#include <float.h>
#include <sstream>
#include <vector>
#include <fstream>

#include "imc/MultilayerPerceptron.h"
#include "imc/util.h"

using namespace imc;
using namespace std;
using namespace util;

int main(int argc, char **argv) {
    // Flags y valores por defecto
    bool pflag = false, wflag = false, normalizate = false, trainMode = false, predictMode = false;
    bool cflag = false;                          // <-- ADD: flag para CSV
    char *trainFile = nullptr, *testFile = nullptr, *weightsFile = nullptr;
    char *csvFile = nullptr;                    // <-- ADD: archivo CSV
    int iterations = 0, hiddenLayers = 0, hiddenNeurons = 0;
    double eta = 0.1, mu = 0.9;
    int c;

    opterr = 0;

    // Parser de argumentos
    while ((c = getopt(argc, argv, "+t:T:i:l:h:e:m:w:c:sp")) != -1) { // <-- ADD c:
        switch(c) {
            case 't': trainFile = optarg; break;
            case 'T': testFile = optarg; break;
            case 'i': iterations = atoi(optarg); break;
            case 'l': hiddenLayers = atoi(optarg); break;
            case 'h': hiddenNeurons = atoi(optarg); break;
            case 'e': eta = atof(optarg); break;
            case 'm': mu  = atof(optarg); break;
            case 's': normalizate = true; break;
            case 'w': wflag = true; weightsFile = optarg; break;
            case 'p': pflag = true; break;
            case 'c': cflag = true; csvFile = optarg; break; // <-- ADD
            case '?':
                cerr << "Unknown option or missing argument" << endl;
                return EXIT_FAILURE;
        }
    }

    // Validación de argumentos
    if (!pflag) {
        //  Modo entrenamiento
        if (!trainFile || !testFile || iterations <= 0 || hiddenLayers < 0) {
            cerr << "Usage (training): " << argv[0]
                << " -t <train_file> -T <test_file> -i <iterations> -l <hidden_layers> -h <neurons_hidden_layer> "
                << "[-e <eta>] [-m <mu>] [-w <weights_file>] [-s] [-c <csv_file>]" << endl; // <-- ADD
            return EXIT_FAILURE;
        }
        // Si el usuario pasó -c pero no puso nombre ==> error
        if (cflag && !csvFile) {               // <-- ADD
            cerr << "Error: -c flag requiere nombre de archivo CSV" << endl;
            return EXIT_FAILURE;
        }
        trainMode = true;
    } else {
        // Modo predicción
        if (!testFile || !weightsFile) {
            cerr << "Usage (predict): " << argv[0]
                << " -T <test_file> -w <weights_file> -p" << endl;
            return EXIT_FAILURE;
        }
        predictMode = true;
    }

    if (trainMode) {
        // --------- ENTRENAMIENTO Y TEST ----------
        Dataset * trainDataset = readData(trainFile);
        Dataset * testDataset  = readData(testFile);
        if (!trainDataset || !testDataset) {
            cerr << "Error al leer datasets" << endl;
            return EXIT_FAILURE;
        }
        int nOfLayers_total = hiddenLayers + 2; // entrada + ocultas + salida
        int *topology = new int[nOfLayers_total];
        topology[0] = trainDataset->nOfInputs;
        for (int i = 0; i < hiddenLayers; i++)
            topology[i+1] = hiddenNeurons;
        topology[nOfLayers_total - 1] = trainDataset->nOfOutputs;
        if (normalizate) {
            // Obtener min y max de las entradas
            double *minInput = minDatasetInputs(trainDataset);
            double *maxInput = maxDatasetInputs(trainDataset);

            // Normalizar entradas en [-1, 1]
            minMaxScalerDataSetInputs(trainDataset, -1, 1, minInput, maxInput);
            minMaxScalerDataSetInputs(testDataset, -1, 1, minInput, maxInput);

            // Obtener min y max de las salidas (solo para regresión)
            double minOutput = minDatasetOutputs(trainDataset);
            double maxOutput = maxDatasetOutputs(trainDataset);

            // Normalizar salidas en [0, 1]
            minMaxScalerDataSetOutputs(trainDataset, 0, 1, minOutput, maxOutput);
            minMaxScalerDataSetOutputs(testDataset, 0, 1, minOutput, maxOutput);

            // Liberar memoria auxiliar
            delete[] minInput;
            delete[] maxInput;
        }

        MultilayerPerceptron mlp;
        mlp.initialize(nOfLayers_total, topology);
        mlp.eta = eta;
        mlp.mu  = mu;

        // <-- ADD: si hay CSV, lo inicializamos con cabecera
        if (cflag && csvFile) {
            ofstream f(csvFile);
            if (f.is_open()) {
                f << "seed,epoch,trainError,testError\n";
                f.close();
            }
        }

        int seeds[] = {1,2,3,4,5};
        const int N = sizeof(seeds) / sizeof(seeds[0]);

        double *testErrors = new double[N];
        double *trainErrors = new double[N];
        for (int i = 0; i < N; ++i) { testErrors[i] = 0.0; trainErrors[i] = 0.0; }

        double bestTestError = DBL_MAX;
        for (int i = 0; i < N; i++) {
            cout << "**********" << endl;
            cout << "SEED " << seeds[i] << endl;
            cout << "**********" << endl;
            srand(seeds[i]);

            // <-- ADD: si hay CSV, pasar nombre a runOnlineBackPropagation extendido
            if (cflag && csvFile) {
                mlp.runOnlineBackPropagation(trainDataset, testDataset, iterations,
                                             &(trainErrors[i]), &(testErrors[i]),
                                             csvFile, seeds[i]); // <-- extensión
            } else {
                mlp.runOnlineBackPropagation(trainDataset, testDataset, iterations,
                                             &(trainErrors[i]), &(testErrors[i]));
            }

            cout << "We end!! => Final test error: " << testErrors[i] << endl;
            if (wflag && testErrors[i] <= bestTestError) {
                mlp.saveWeights(weightsFile);
                bestTestError = testErrors[i];
            }
        }

        double averageTestError = 0, stdTestError = 0;
        double averageTrainError = 0, stdTrainError = 0;
        for (int i = 0; i < N; i++) {
            averageTestError += testErrors[i];
            averageTrainError += trainErrors[i];
        }
        averageTestError /= N;
        averageTrainError /= N;
        double sumSqDiffTest = 0.0, sumSqDiffTrain = 0.0;
        for (int i = 0; i < N; i++) {
            sumSqDiffTest  += pow(testErrors[i] - averageTestError, 2);
            sumSqDiffTrain += pow(trainErrors[i] - averageTrainError, 2);
        }
        stdTestError  = sqrt(sumSqDiffTest  / (N > 1 ? N-1 : 1));
        stdTrainError = sqrt(sumSqDiffTrain / (N > 1 ? N-1 : 1));
        cout << "FINAL REPORT" << endl;
        cout << "************" << endl;
        cout << "Train error (Mean +- SD): " << averageTrainError << " +- " << stdTrainError << endl;
        cout << "Test error  (Mean +- SD): " << averageTestError << " +- " << stdTestError << endl;

        delete[] topology;
        delete[] testErrors;
        delete[] trainErrors;
        if (trainDataset) {
            for (int i = 0; i < trainDataset->nOfPatterns; i++) {
                delete[] trainDataset->inputs[i];
                delete[] trainDataset->outputs[i];
            }
            delete[] trainDataset->inputs;
            delete[] trainDataset->outputs;
            delete trainDataset;
        }
        if (testDataset) {
            for (int i = 0; i < testDataset->nOfPatterns; i++) {
                delete[] testDataset->inputs[i];
                delete[] testDataset->outputs[i];
            }
            delete[] testDataset->inputs;
            delete[] testDataset->outputs;
            delete testDataset;
        }
    } 
    else if (predictMode) {
        // --------- PREDICCIÓN (Kaggle) ----------
        Dataset * testDataset  = readData(testFile);
        if (!testDataset) {
            cerr << "Error al leer dataset de test" << endl;
            return EXIT_FAILURE;
        }
        MultilayerPerceptron mlp;
        if (!wflag || !mlp.readWeights(weightsFile)) {
            cerr << "Error: no se pudieron leer los pesos" << endl;
            return EXIT_FAILURE;
        }
        mlp.predict(testDataset);
        for (int i = 0; i < testDataset->nOfPatterns; i++) {
            delete[] testDataset->inputs[i];
            delete[] testDataset->outputs[i];
        }
        delete[] testDataset->inputs;
        delete[] testDataset->outputs;
        delete testDataset;
    }
    return EXIT_SUCCESS;
}
