//
//  main.cpp
//
//
//  Created by koreanhole on 29/09/2019.
//  Copyright © 2019 koreanhole. All rights reserved.
//

#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <ctime>
#include <fstream>
#include <math.h>

using namespace std;

class init {
public:
    vector<vector<float>> gateInput(string gatename);
    vector<vector<float>> gateTarget(string gatename);
    int setInputDim(string gatename);
    int setOutputDim(string gatename);
};


class multiLayerPerceptron {
           struct inputLayer{
               int outputDim;
               std::vector<vector<float>> output;
               inputLayer(int outputDim_){
                   output.clear();
                   outputDim = outputDim_;
                   output.assign(outputDim, vector<float>(1,0));
               }
           };
           struct hiddenLayer{
               int inputDim;
               int numNodes;
               std::vector<vector<float>> delta;
               std::vector<vector<float>> net;
               std::vector<vector<float>> weight;
               std::vector<vector<float>> bias;
               std::vector<vector<float>> output;
               
               hiddenLayer(int inputDim_, int numNodes_){
                   srand((unsigned int)time(0));
                   inputDim = inputDim_;
                   numNodes = numNodes_;

                   delta.assign(numNodes, vector<float>(1,0));
                   net.assign(numNodes, vector<float>(1,0));
                   output.assign(numNodes, vector<float>(1,0));
                   bias.assign(numNodes, vector<float>(1,0));
                   weight.assign(numNodes, vector<float>(inputDim, 0)); //initializing hidden layer's delta, net, output, bias, weight
                   for (int i = 0; i < numNodes; i++) {
                       for (int j = 0; j < inputDim; j++) {
                           weight[i][j] = static_cast <float> (rand()%2000-1000) / static_cast <float> (1000);
                           bias[i][0] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                       }//initialize weight and bias with random number between -1...1
                   }
               }
           };
           struct outputLayer{
               int inputDim;
               int numNodes;
               float global_Err = 0.;
               
               std::vector<vector<float>> net;
               std::vector<vector<float>> bias;
               std::vector<vector<float>> output;
               std::vector<vector<float>> target;
               std::vector<vector<float>> weight;
               std::vector<vector<float>> delta;
               
               outputLayer(int inputDim_, int numNodes_){
                   srand((unsigned int)time(0));
                   inputDim = inputDim_;
                   numNodes = numNodes_;
                   
                   net.assign(numNodes, vector<float>(1,0));
                   bias.assign(numNodes, vector<float>(1,0));
                   output.assign(numNodes, vector<float>(1,0));
                   target.assign(numNodes, vector<float>(1,0));
                   delta.assign(numNodes, vector<float>(1,0));
                   weight.assign(numNodes, vector<float>(inputDim, 0)); //set 0 value in net, bias, output, target, delta, weight
                   for (int i = 0; i < numNodes; i++) {
                       for (int j = 0; j < inputDim; j++) {
                           weight[i][j] = static_cast <float> (rand()%2000-1000) / static_cast <float> (1000);
                           bias[i][0] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                       }//initialize weight and bias with random number between -1 and 1
                   }
               }
           };
private:
    int inputDim;
    int outputDim;
    int numHiddenLayer;
    int numHiddenLayerNodes;
    float learningRate = 0.7; //learningrate
    
    std::vector<inputLayer> input;
    std::vector<hiddenLayer> firstHidden;
    std::vector<hiddenLayer> otherHidden;
    std::vector<outputLayer> output;
    
    float activationFunc(float net);
    float def_activationFunc(float net);
    void weightToTxt(vector<hiddenLayer> frsthidn, vector<hiddenLayer> otherhidn, string filename);
    vector<vector<float>> mulMatrix(vector<vector<float>> x, vector<vector<float>> y);
    vector<vector<float>> netCalc(vector<vector<float>> input, vector<vector<float>> weight, vector<vector<float>> bias);
    vector<vector<float>> outputCalc(vector<vector<float>>);
    float errCalc(vector<vector<float>> target, vector<vector<float>> output);
    vector<vector<float>> firstDeltaCalc(vector<vector<float>> target, vector<vector<float>> output);
    vector<vector<float>> deltaCalc(vector<vector<float>> net, vector<vector<float>> delta, vector<vector<float>> weight);
    vector<vector<float>> updateWeight(vector<vector<float>> weigth, vector<vector<float>> former_output, vector<vector<float>> delta);
    vector<vector<float>> updateBias(vector<vector<float>> bias, vector<vector<float>> delta);
    void feedForward(int numHidnLyr);
    void backProp(int numHidnLyr);

public:
    void learning(vector<vector<float>> inputValue, vector<vector<float>> targetValue, int inputDim, int hidnDim, int outputDim, int numHiddnlyr, float tolerance);
};

int main(int argc, const char * argv[]) {
    multiLayerPerceptron obj;
    init in;
    
    int inputDim; //number of nodes in inputLayer
    int outputDim; //number of nodes in outputLayer
    int numHiddenLayer; //number of hidden layer
    int hiddenDim; //number of nodes in hidden layer
    string gateName;
    
    vector<vector<float>> input; // vector that saves inputValue
    vector<vector<float>> target; // vector that saves targetValue
    
    std::cout << "select gate" << endl << "andgate/orgate/xorgate/donut" << endl;
    std::cin >> gateName;
    inputDim = in.setInputDim(gateName);
    outputDim = in.setOutputDim(gateName);
    input = in.gateInput(gateName);
    target = in.gateTarget(gateName);
    std::cout << "set number of hidden layer" << endl;
    std::cin >> numHiddenLayer;
    std::cout << "set number of nodes in hidden layer" << endl;
    std::cin >> hiddenDim;
    std::cout << "learning" << endl;

    obj.learning(input, target, inputDim, hiddenDim, outputDim, numHiddenLayer, 0.00001);
    
    return 0;
}
vector<vector<float>> init::gateInput(string gateName){
    if (gateName == "andgate" || gateName == "orgate" || gateName == "xorgate") {
        return vector<vector<float>> ({
            vector<float>({0.00,0.00}),
            vector<float>({0.00,1.00}),
            vector<float>({1.00,0.00}),
            vector<float>({1.00,1.00})
        });
    }
    else if (gateName == "donut"){
        return vector<vector<float>> ({
            vector<float>({0.,0.}),
            vector<float>({0.,1.}),
            vector<float>({1.,0.}),
            vector<float>({1.,1.}),
            vector<float>({0.5,1.}),
            vector<float>({1.,0.5}),
            vector<float>({0.,0.5}),
            vector<float>({0.5,0.}),
            vector<float>({0.5,0.5})
        });
    }
    else {
        std::cout << "it must be in <andgate, orgate, xorgate, donut>" << endl;
        exit(0);
    }
}
vector<vector<float>> init::gateTarget(string gateName) {
    if (gateName == "andgate"){
        return vector<vector<float>> ({
            vector<float>({0.00}),
            vector<float>({0.00}),
            vector<float>({0.00}),
            vector<float>({1.00})
        });
    }
    else if (gateName == "orgate"){
        return vector<vector<float>> ({
            vector<float>({0.00}),
            vector<float>({1.00}),
            vector<float>({1.00}),
            vector<float>({1.00})
        });
    }
    else if (gateName == "xorgate"){
        return vector<vector<float>> ({
            vector<float>({0.00}),
            vector<float>({1.00}),
            vector<float>({1.00}),
            vector<float>({0.00})
        });
    }
    else if (gateName == "donut"){
        return vector<vector<float>>({
            vector<float>({0.}),
            vector<float>({0.}),
            vector<float>({0.}),
            vector<float>({0.}),
            vector<float>({0.}),
            vector<float>({0.}),
            vector<float>({0.}),
            vector<float>({0.}),
            vector<float>({1.})
        });
    }
    else {
        std::cout << "it must be in <andgate, orgate, xorgate, donut>" << endl;
        exit(0);
    }
}
int init::setInputDim(string gateName){
    if (gateName == "andgate" || gateName == "orgate" || gateName == "xorgate" || gateName == "donut") {
        return 2;
    }
    else{
        std::cout << "andgate, orgate, xorgate, donut 중 하나를 입력해주세요" << endl;
        exit(0);
    }
}
int init::setOutputDim(string gateName){
    if (gateName == "andgate" || gateName == "orgate" || gateName == "xorgate" || gateName == "donut") {
        return 1;
    }
    else{
        std::cout << "andgate, orgate, xorgate, donut 중 하나를 입력해주세요" << endl;
        exit(0);
    }
}
void multiLayerPerceptron::weightToTxt(vector<hiddenLayer> firsthidn, vector<hiddenLayer> otherhidn, string filename){
    //function that saves weight in txt file
    /* <example>
          row (w1) (w2) ... (wn) (bias) (output)
     column
     nodes
     */
    int numotherLyr = static_cast<int>(otherhidn.size());
    ofstream weightTxt;
    int firstRow = static_cast<int>(firsthidn[0].weight.size());
    int firstCol = static_cast<int>(firsthidn[0].weight[0].size());
    weightTxt.open("1번째 히든레이어의 weight,output" + filename);
    for (int i = 0; i < firstRow; i++) {
        for (int j = 0; j < firstCol; j++) {
            weightTxt << firsthidn[0].weight[i][j] << " ";
        }
        weightTxt << firsthidn[0].bias[i][0] << " " <<firsthidn[0].output[i][0] << endl;
    }
    weightTxt.close();
    //runs only numLyr > 2
    if (numotherLyr != 0) {
        for (int i = 0; i < numotherLyr; i++) {
            weightTxt.open(to_string(i+2) + "번째 히든레이어의 weight,output" + filename);
            int otherRow = static_cast<int>(otherhidn[i].weight.size());
            int otherCol = static_cast<int>(otherhidn[i].weight[0].size());
            for (int j = 0; j < otherRow; j++) {
                for (int k = 0; k < otherCol; k++) {
                    weightTxt << otherhidn[i].weight[j][k] << " ";
                }
                weightTxt << otherhidn[i].bias[j][0]<< " " << otherhidn[i].output[j][0] << endl;
            }
            weightTxt.close();
        }
    }
}
float multiLayerPerceptron::activationFunc(float net) {
    return 1.0f / (1+exp((-1)*net));
}//sigmoid function
float multiLayerPerceptron::def_activationFunc(float net) {
    return activationFunc(net) * (1-activationFunc(net));
}//derivates of sigmoid
vector<vector<float>> multiLayerPerceptron::mulMatrix(vector<vector<float>> x, vector<vector<float>> y){
     //implementing matrix multiplication
    int x_row = static_cast<int>(x.size());
    int x_col = static_cast<int>(x[0].size());
    int y_row = static_cast<int>(y.size());
    int y_col = static_cast<int>(y[0].size());

    std::vector<vector<float>> result;
    result.assign(x_row, vector<float>(y_col,0));
    
    if (x_col != y_row) {
        std::cout << "unable to matrix multiplication." << endl;
    }
    
    else {
        for (int i = 0; i < x_row ; i++) {
            for (int j = 0; j < y_col; j++) {
                for (int k = 0; k < x_col; k++) {
                    result[i][j] += x[i][k] * y[k][j];
                }
            }
        }
    }
    return result;
}
vector<vector<float>> multiLayerPerceptron::netCalc(vector<vector<float>> input, vector<vector<float>> weight, vector<vector<float>> bias){
     //function that calculation net value
    int row = static_cast<float>(weight.size());
    std::vector<vector<float>> result;
    result = mulMatrix(weight, input); //net' = weight * input
    
    for (int i = 0; i < row; i++) {
        result[i][0] += bias[i][0]; //net = net' + bias
    }
    return result;
}
vector<vector<float>> multiLayerPerceptron::outputCalc(vector<vector<float>> net){
    //get output
    int net_row = static_cast<int>(net.size());
    std::vector<vector<float>> result;
    result.assign(net_row, vector<float>(1,0));
    for (int i = 0; i < net_row; i++) {
        result[i][0] = activationFunc(net[i][0]);
    } // output = f(net)
    return result;
}
float multiLayerPerceptron::errCalc(vector<vector<float>> target, vector<vector<float>> output){
    //error calculating with mean squared error function

    int target_row = static_cast<int>(target.size());
    int output_row = static_cast<int>(output.size());
    float err = 0.0;
    
    if (target_row != output_row) {
        std::cout << "number of rows in target and output are different" << endl;
    }
    else {
        for (int i = 0; i < target_row; i++) {
            float temp = target[i][0] - output[i][0];
            err += powf(temp, 2);
        }
        err *= 0.5;
    }
    return err;
}
vector<vector<float>> multiLayerPerceptron::firstDeltaCalc(vector<vector<float>> target_, vector<vector<float>> output_){
    //calculating delta in output layer only
    int numNodes = static_cast<int>(output.size());
    std::vector<vector<float>> delta;
    delta.assign(numNodes, vector<float>(1,0));
    
    for (int i = 0; i < numNodes; i++) {
        delta[i][0] = -(target_[i][0] - output_[i][0])*def_activationFunc(output[0].net[i][0]);
    }
    return delta;
}
vector<vector<float>> multiLayerPerceptron::deltaCalc(vector<vector<float>> present_net, vector<vector<float>> next_delta, std::vector<vector<float>> next_weight){
    //calculating delta in hidden layer
    int next_row = static_cast<int>(next_weight.size());
    int next_col = static_cast<int>(next_weight[0].size());
    std::vector<vector<float>> new_delta;
    new_delta.assign(next_col, vector<float>(1,0));
    
    for (int i = 0; i < next_col; i++) {
        //i = number of nodes in persent layer
        for (int j = 0; j < next_row; j++) {
            //j = number of nodes in next layer
            new_delta[i][0] += next_delta[j][0]*next_weight[j][i]*def_activationFunc(present_net[i][0]);
        }
    }
    return new_delta;
}
vector<vector<float>> multiLayerPerceptron::updateWeight(vector<vector<float>> present_weigth, vector<vector<float>> previous_output, vector<vector<float>> present_delta){
    //updating weights
    int weightRow = static_cast<int>(present_weigth.size());
    int weightCol = static_cast<int>(present_weigth[0].size());
    std::vector<vector<float>> new_weight;
    new_weight.assign(weightRow, vector<float>(weightCol,0));
    for (int i = 0; i < weightRow; i++) {
        //i = number of nodes
        for (int j = 0; j < weightCol; j++) {
            //j = number of weights per node
            float delta_weight = 0;
            delta_weight = (-1)*learningRate*present_delta[i][0]*previous_output[j][0];
            new_weight[i][j] = present_weigth[i][j] + delta_weight;
        }
    }
    return new_weight;
}
vector<vector<float>> multiLayerPerceptron::updateBias(vector<vector<float>> present_bias, vector<vector<float>> present_delta){
    //updating bias
    int biasRow = static_cast<int>(present_bias.size());
    std::vector<vector<float>> new_bias;
    new_bias.assign(biasRow, vector<float>(1,0));
    
    for (int i = 0; i < biasRow; i++) {
        //i = number of nodes(bias)
        float delta_bias = 0;
        delta_bias += (-1)*learningRate*present_delta[i][0];
        new_bias[i][0] = present_bias[i][0] + delta_bias;
    }
    return new_bias;
}
void multiLayerPerceptron::feedForward(int numHydnLyr){
    //feedforwarding
    firstHidden[0].net = netCalc(input[0].output, firstHidden[0].weight, firstHidden[0].bias);
    firstHidden[0].output = outputCalc(firstHidden[0].net);
    //calculating net and output in first hidden layer

    if (numHydnLyr == 1) {
        //in case of number of hidden layer = 1
        output[0].net = netCalc(firstHidden[0].output, output[0].weight, output[0].bias);
    }
    else {
        // number of hidden layer > 2
        for (int i = 0; i < otherHidden.size(); i++) {
            if (i == 0) {
                otherHidden[i].net = netCalc(firstHidden[0].output, otherHidden[i].weight, otherHidden[i].bias);
                otherHidden[i].output = outputCalc(otherHidden[i].net);
            }
            else {
                otherHidden[i].net = netCalc(otherHidden[i-1].output, otherHidden[i].weight, otherHidden[i].bias);
                otherHidden[i].output = outputCalc(otherHidden[i].net);
            }
        }
        //calculating net in output layer
        output[0].net = netCalc(otherHidden[numHydnLyr - 2].output, output[0].weight, output[0].bias);
    }
    //calculating output and error in output layer
    output[0].output = outputCalc(output[0].net);
    output[0].global_Err = errCalc(output[0].target, output[0].output);
}
void multiLayerPerceptron::backProp(int numHidnLyr) {
    //back propagation
    int numOtherHidn = static_cast<int>(otherHidden.size());
    //calculating delta in output layer
    output[0].delta = firstDeltaCalc(output[0].target, output[0].output);

    if (numHidnLyr == 1) {
        //only one hidden layer
        firstHidden[0].delta = deltaCalc(firstHidden[0].net, output[0].delta, output[0].weight);

        output[0].weight = updateWeight(output[0].weight, firstHidden[0].output, output[0].delta);
        output[0].bias = updateBias(output[0].bias, output[0].delta);
        firstHidden[0].weight = updateWeight(firstHidden[0].weight, input[0].output, firstHidden[0].delta);
        firstHidden[0].bias = updateBias(firstHidden[0].bias, firstHidden[0].delta);
    }
    else if (numHidnLyr == 2){
        //number of hidden layer = 2
        //calculating delta in second hidden layer
        otherHidden[0].delta = deltaCalc(otherHidden[0].net, output[0].delta, output[0].weight);
        //calculating delta in first hidden layer
        firstHidden[0].delta = deltaCalc(firstHidden[0].net, otherHidden[0].delta, otherHidden[0].weight);
        //updating weight and bias in all layer
        output[0].weight = updateWeight(output[0].weight, otherHidden[0].output, output[0].delta);
        output[0].bias = updateBias(output[0].bias, output[0].delta);
        otherHidden[0].weight = updateWeight(otherHidden[0].weight, firstHidden[0].output, otherHidden[0].delta);
        otherHidden[0].bias = updateBias(otherHidden[0].bias, otherHidden[0].delta);
        firstHidden[0].weight = updateWeight(firstHidden[0].weight, input[0].output, firstHidden[0].delta);
        firstHidden[0].bias = updateBias(firstHidden[0].bias, firstHidden[0].delta);
    }
    else if (numHidnLyr > 2){
        //there is more than 3 hidden layer
        for (int i = numOtherHidn - 1; i >= 0 ; i--) {
            if (i == numOtherHidn - 1) {
                //calculating delta in last hidden layer
                otherHidden[i].delta = deltaCalc(otherHidden[i].net, output[0].delta, output[0].weight);
            }
            else if (i > 0 && i < numOtherHidn - 1){
                //calculating delta except for first, second and last hidden layer
                otherHidden[i].delta = deltaCalc(otherHidden[i].net, otherHidden[i+1].delta, otherHidden[i+1].weight);
            }
            else {
                //calculating delta in second hidden layer
                otherHidden[i].delta = deltaCalc(otherHidden[i].net, otherHidden[i+1].delta, otherHidden[i+1].weight);
            }
        }
        //calculating delta in first hidden layer
        firstHidden[0].delta = deltaCalc(firstHidden[0].net, otherHidden[0].delta, otherHidden[0].weight);
        
        
        for (int i = numOtherHidn - 1; i >= 0 ; i--) {
            //updating weight and bias
            if (i == numOtherHidn - 1) {
                otherHidden[i].weight = updateWeight(otherHidden[i].weight, otherHidden[i-1].output, otherHidden[i].delta);
                otherHidden[i].bias = updateBias(otherHidden[i].bias, otherHidden[i].delta);
            }
            else if (i > 0 && i < numOtherHidn - 1){
                otherHidden[i].weight = updateWeight(otherHidden[i].weight, otherHidden[i-1].output, otherHidden[i].delta);
                otherHidden[i].bias = updateBias(otherHidden[i].bias, otherHidden[i].delta);
            }
            else {
                otherHidden[i].weight = updateWeight(otherHidden[i].weight, firstHidden[0].output, otherHidden[i].delta);
                otherHidden[i].bias = updateBias(otherHidden[i].bias, otherHidden[i].delta);
            }
            firstHidden[0].weight = updateWeight(firstHidden[0].weight, input[0].output, firstHidden[0].delta);
            firstHidden[0].bias = updateBias(firstHidden[0].bias, firstHidden[0].delta);
            output[0].weight = updateWeight(output[0].weight, otherHidden[numOtherHidn-1].output, output[0].delta);
            output[0].bias = updateBias(output[0].bias, output[0].delta);
        }

    }
    //reset all delta
    if (numOtherHidn >= 1) {
        int numOtherHidnRow = static_cast<int>(otherHidden[0].delta.size());
        for (int i = 0; i < numOtherHidn - 1; i++) {
            otherHidden[i].delta.assign(numOtherHidnRow, vector<float>(1,0));
        }
    }
    int numFirstHidnRow = static_cast<int>(firstHidden[0].delta.size());
    firstHidden[0].delta.assign(numFirstHidnRow, vector<float>(1,0));
    int numOutputRow = static_cast<int>(output[0].delta.size());
    output[0].delta.assign(numOutputRow, vector<float>(1,0));
}
void multiLayerPerceptron::learning(vector<vector<float>> inputValue, vector<vector<float>> targetValue, int inputDim_, int hidnDim_, int outputDim_, int numHiddnlyr, float tolerance){
    //multilayer perceptron learning
    
    //size of input value
    int numIteration = static_cast<int>(inputValue.size());
    //creating inputLayer, hiddenLayer, outputLayer Vector
    input.push_back(inputLayer(inputDim_));
    output.push_back(outputLayer(hidnDim_,outputDim_));
    firstHidden.push_back(hiddenLayer(inputDim_, hidnDim_));
    for (int i = 0; i < numHiddnlyr - 1; i++) {
        otherHidden.push_back(hiddenLayer(hidnDim_,hidnDim_));
    }
    
    //save initial weight in txt file
    weightToTxt(firstHidden, otherHidden, "_before learning");
    
    ofstream err;
    err.open("error.txt");
    
    int num = 0;

    bool stop = false;
    //stop when learning meet tolerance
        while (!stop) {
            float tempErr = 0.;
            num += 1;
            if (num % 5000 == 0) {
                std::cout << endl <<"<<<"<< num <<"번째 실행중>>>" << endl;
            }
            stop = true;
            //set each input and target value in inputLayer and outputLayer
            for (int i = 0; i < numIteration; i++) {
                for (int j = 0; j < inputDim_; j++) {
                    input[0].output[j][0] = inputValue[i][j];
                }
                for (int j = 0; j < outputDim_; j++) {
                    output[0].target[j][0] = targetValue[i][j];
                }
                feedForward(numHiddnlyr);
                tempErr += output[0].global_Err;
                //stop iteration when error meet tolerance
                if (output[0].global_Err > tolerance) {
                    stop = false;
                    backProp(numHiddnlyr);
                }
                //print target, output, error when epoch runs 5000 times
                if (num % 5000 == 0) {
                    std::cout << "target: " << output[0].target[0][0] << " output: " << output[0].output[0][0] << endl;
                    std::cout << "global error: "<<output[0].global_Err <<endl;
                }
            }
            //save weights in txt file when epoch is 500
            if (num == 500) {
                weightToTxt(firstHidden, otherHidden, "_학습중");
            }
            //save means of error in txt file
            err << tempErr/numIteration << endl;
    }
    std:: cout << endl <<"총 실행횟수: " << num << endl;
    
    //save weights and bias in txt file when learning is done
    weightToTxt(firstHidden, otherHidden, "_학습후");

    input.clear();
    firstHidden.clear();
    otherHidden.clear();
    output.clear();
    
    std::cout << "gate completed" << endl;
}

