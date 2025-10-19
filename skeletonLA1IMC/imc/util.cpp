/*********************************************************************
* File  : util.cpp
* Date  : 2020
* Autor : Pedro A. Gutiérrez + Modificado para normalización
*********************************************************************/

#include "util.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <limits>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;
using namespace util;

// ------------------------------
// Obtain an integer random number in the range [Low,High]
int util::randomInt(int Low, int High)
{
	return rand() % (High-Low+1) + Low;
}

// ------------------------------
// Obtain a real random number in the range [Low,High]
double util::randomDouble(double Low, double High)
{
	return ((double) rand() / RAND_MAX) * (High-Low) + Low;
}

// ------------------------------
// Read a dataset from a file name and return it
Dataset *util::readData(const char *fileName)
{

    ifstream myFile(fileName); // Create an input stream

    if (!myFile.is_open())
    {
        cout << "ERROR: I cannot open the file " << fileName << endl;
        return NULL;
    }

    Dataset *dataset = new Dataset;
    if (dataset == NULL)
        return NULL;

    string line;
    int i, j;

    if (myFile.good())
    {
        getline(myFile, line); // Read a line
        istringstream iss(line);
        iss >> dataset->nOfInputs;
        iss >> dataset->nOfOutputs;
        iss >> dataset->nOfPatterns;
    }
    dataset->inputs = new double *[dataset->nOfPatterns];
    dataset->outputs = new double *[dataset->nOfPatterns];

    for (i = 0; i < dataset->nOfPatterns; i++)
    {
        dataset->inputs[i] = new double[dataset->nOfInputs];
        dataset->outputs[i] = new double[dataset->nOfOutputs];
    }

    i = 0;
    while (myFile.good())
    {
        getline(myFile, line); // Read a line
        if (!line.empty())
        {
            istringstream iss(line);
            for (j = 0; j < dataset->nOfInputs; j++)
            {
                double value;
                iss >> value;
                if (!iss)
                    return NULL;
                dataset->inputs[i][j] = value;
            }
            for (j = 0; j < dataset->nOfOutputs; j++)
            {
                double value;
                iss >> value;
                if (!iss)
                    return NULL;
                dataset->outputs[i][j] = value;
            }
            i++;
        }
    }

    myFile.close();

    return dataset;
}

// ------------------------------
// Print the dataset
void util::printDataset(Dataset *dataset, int len)
{
    if (len == 0)
        len = dataset->nOfPatterns;

    for (int i = 0; i < len; i++)
    {
        cout << "P" << i << ":" << endl;
        for (int j = 0; j < dataset->nOfInputs; j++)
        {
            cout << dataset->inputs[i][j] << ",";
        }

        for (int j = 0; j < dataset->nOfOutputs; j++)
        {
            cout << dataset->outputs[i][j] << ",";
        }
        cout << endl;
    }
}

// ------------------------------
// ------------------------------
// Transform an scalar x by scaling it to a given range [minAllowed, maxAllowed] considering the min
// and max values of the feature in the dataset (minData and maxData). 
double util::minMaxScaler(double x, double minAllowed, double maxAllowed, double minData, double maxData)
{
    if (maxData == minData)
        return (minAllowed + maxAllowed) / 2.0;

    double scaled = ( (x - minData) / (maxData - minData) ) * (maxAllowed - minAllowed) + minAllowed;
    return scaled;
}
// ------------------------------
// Scale the dataset inputs to a given range [minAllowed, maxAllowed] considering the min
// and max values of the feature in the dataset (minData and maxData). 
void util::minMaxScalerDataSetInputs(Dataset *dataset, double minAllowed, double maxAllowed,
                                     double *minData, double *maxData)
{
    if (dataset == NULL || minData == NULL || maxData == NULL)
        return;

    int nP = dataset->nOfPatterns;
    int nI = dataset->nOfInputs;
    for (int p = 0; p < nP; ++p){
        for (int j = 0; j < nI; ++j){
            dataset->inputs[p][j] = minMaxScaler(dataset->inputs[p][j], minAllowed, maxAllowed, minData[j], maxData[j]);
        }
    }
}

// ------------------------------
// Scale the dataset output vector to a given range [minAllowed, maxAllowed] considering the min
// and max values of the feature in the dataset (minData and maxData). Only for regression problems. 
void util::minMaxScalerDataSetOutputs(Dataset *dataset, double minAllowed, double maxAllowed,
                                      double minData, double maxData)
{

    if (dataset == NULL) return;
    int nP = dataset->nOfPatterns;
    int nO = dataset->nOfOutputs;
    for (int p = 0; p < nP; ++p){
        for (int j = 0; j < nO; ++j){
            dataset->outputs[p][j] = minMaxScaler(dataset->outputs[p][j], minAllowed, maxAllowed, minData, maxData);
        }
    }
}
// ------------------------------
// Get a vector of minimum values of the dataset inputs
double *util::minDatasetInputs(Dataset *dataset)
{
    if (dataset == NULL) return NULL;
    int nI = dataset->nOfInputs;
    int nP = dataset->nOfPatterns;
    double *mins = new double[nI];
    for (int j = 0; j < nI; ++j) mins[j] = numeric_limits<double>::infinity();

    for (int p = 0; p < nP; ++p){
        for (int j = 0; j < nI; ++j){
            if (dataset->inputs[p][j] < mins[j]) mins[j] = dataset->inputs[p][j];
        }
    }
    return mins;
}

// ------------------------------
// Get a vector of maximum values of the dataset inputs
double *util::maxDatasetInputs(Dataset *dataset)
{
    if (dataset == NULL) return NULL;
    int nI = dataset->nOfInputs;
    int nP = dataset->nOfPatterns;
    double *maxs = new double[nI];
    for (int j = 0; j < nI; ++j) maxs[j] = -numeric_limits<double>::infinity();

    for (int p = 0; p < nP; ++p){
        for (int j = 0; j < nI; ++j){
            if (dataset->inputs[p][j] > maxs[j]) maxs[j] = dataset->inputs[p][j];
        }
    }
    return maxs;
}

// ------------------------------
// Get the minimum value of the dataset outputs
double util::minDatasetOutputs(Dataset *dataset)
{
    if (dataset == NULL) return 0.0;
    int nP = dataset->nOfPatterns;
    int nO = dataset->nOfOutputs;
    double minv = numeric_limits<double>::infinity();
    for (int p = 0; p < nP; ++p){
        for (int j = 0; j < nO; ++j){
            if (dataset->outputs[p][j] < minv) minv = dataset->outputs[p][j];
        }
    }
    return minv;
}

// ------------------------------
 // Get the maximum value of the dataset outputs
double util::maxDatasetOutputs(Dataset *dataset)
{
    if (dataset == NULL) return 0.0;
    int nP = dataset->nOfPatterns;
    int nO = dataset->nOfOutputs;
    double maxv = -numeric_limits<double>::infinity();
    for (int p = 0; p < nP; ++p){
        for (int j = 0; j < nO; ++j){
            if (dataset->outputs[p][j] > maxv) maxv = dataset->outputs[p][j];
        }
    }
    return maxv;

}
// ------------------------------