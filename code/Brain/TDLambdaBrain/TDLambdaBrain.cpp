//  MABE is a product of The Hintze Lab @ MSU
//     for general research information:
//         hintzelab.msu.edu
//     for MABE documentation:
//         github.com/Hintzelab/MABE/wiki
//
//  Copyright (c) 2015 Michigan State University. All rights reserved.
//     to view the full license, visit:
//         github.com/Hintzelab/MABE/wiki/License

#include "TDLambdaBrain.h"
#include <Brain/TDLambdaBrain/TDUtils.h>
#include <math.h> // round, truc

std::shared_ptr<ParameterLink<std::string>> TDLambdaBrain::dimensionsPL = Parameters::register_parameter( "BRAIN_TDLAMBDA-dimensions", std::string("0"), "csv list of salient input dimensions, such as \"3,4\" (3 colors, 4 actions)"); // string parameter for
std::shared_ptr<ParameterLink<bool>> TDLambdaBrain::use_confidencePL = Parameters::register_parameter( "BRAIN_TDLAMBDA-useConfidence", true, "'confidence' is decision-annealing, allowing convergence on perfect behavior, and is reset upon TD surprise");
std::shared_ptr<ParameterLink<double>> TDLambdaBrain::no_confidence_random_actionPL = Parameters::register_parameter( "BRAIN_TDLAMBDA-pRandomAction", 0.01, "when not using 'confidence', how often to choose a random action");

// ctor, constructor
TDLambdaBrain::TDLambdaBrain(int ins, int outs, std::shared_ptr<ParametersTable> PT_)
    : AbstractBrain(ins, outs, PT_) {
    std::vector<int> intdimensions;
    convertCSVListToVector<int>(std::string(dimensionsPL->get()), intdimensions);
    dimensions = std::valarray<int>( intdimensions.data(), intdimensions.size() );
    
    use_confidence = use_confidencePL->get();
    no_confidence_random_action = no_confidence_random_actionPL->get();

    nrInputValues = ins;
    nrOutputValues = outs;
    recordActivity = false;

    inputValues.resize(nrInputValues);
    outputValues.resize(nrOutputValues);

    int n_actions(end(dimensions)[-1]);

    tdlambda = TD::Lambda ({.dims=dimensions,
                            .n_features=TD::prod(dimensions),
                            .n_actions=n_actions,
                            .alpha=0.1,
                            .gamma=0.95,
                            .epsilon=0.1,
                            .lmbdaG=0.96, .lmbdaB=0.94, .lmbdaN=0.42});
    tdlambda.use_confidence = use_confidence;
    tdlambda.no_confidence_random_action = no_confidence_random_action;
}

void TDLambdaBrain::update(){
  tdlambda.reward = inputValues[0]; // reward always at pos 0
  for (int i(1); i<nrInputValues; ++i) {
    //tdlambda.sensoryState[i-1] = reinterpret_cast<int*>(&inputValues[i])[0]; // old cast version
    tdlambda.sensoryState[i-1] = int(std::trunc(std::round(inputValues[i]))); // non-cast version
  }
  tdlambda.plasticUpdate();
  //outputValues[0] = reinterpret_cast<double*>(&tdlambda.action)[0]; // old cast version
  outputValues[0] = double(tdlambda.action); // non-cast version
}

// make a copy of the brain that called this
std::shared_ptr<AbstractBrain> TDLambdaBrain::makeCopy(std::shared_ptr<ParametersTable> PT_) {
  std::shared_ptr<TDLambdaBrain> newBrain = std::make_shared<TDLambdaBrain>(nrInputValues, nrOutputValues, PT);
  newBrain->tdlambda.weights = this->tdlambda.weights;
    return(std::make_shared<TDLambdaBrain>(nrInputValues, nrOutputValues, PT));
}

// Make a brain like the brain that called this function, using genomes and initalizing other elements.
std::shared_ptr<AbstractBrain> TDLambdaBrain::makeBrain(std::unordered_map<std::string, std::shared_ptr<AbstractGenome>> & _genomes) {
    
  std::shared_ptr<TDLambdaBrain> newBrain = std::make_shared<TDLambdaBrain>(nrInputValues, nrOutputValues, PT);

  auto gh = _genomes["root::"]->newHandler(_genomes["root::"], true);
  int num_sites(tdlambda.params.n_features);
  const double minval(-8.0), maxval(8.0);

  /* Debugging with evolved genome, assumes alphabet size 1k */ 
  //std::valarray<double> perfectWeights {-3.489, 0.0, 0.0, -2.836, -3.115, -1.825, 0.0, 0.0, 0.0, -3.36, 0.0, 0.0, -3.035, -2.541, -1.947, 0.0, 0.0, 0.0, -6.643, 0.0, 0.0, -2.024, -3.041, -1.407, 0.0, 0.0, 0.0, -7.181, 0.0, 0.0, -4.884, -3.197, -0.414, 0.0, 0.0, 0.0};
  //for (int i=0; i<num_sites; i++) {
  //  gh->writeDouble(perfectWeights[i],minval,maxval);
  //  newBrain->tdlambda.weights[i] = perfectWeights[i];
  //}

  for (int i=0; i<num_sites; i++) {
    newBrain->tdlambda.weights[i] = gh->readDouble(minval,maxval);
    //newBrain->tdlambda.weights[i] *= gh->readDouble(minval,maxval);
  }
  std::copy( begin(newBrain->tdlambda.weights), end(newBrain->tdlambda.weights), begin(newBrain->tdlambda.originalWeights) );
  return(newBrain);
}

std::string TDLambdaBrain::description() {
    return "(no description)";
}

DataMap TDLambdaBrain::getStats(std::string& prefix) {
    // return a vector of DataMap of stats from this brain, this is called just after the brain is constructed
    // values in this datamap are added to the datamap on the organism that "owns" this brain
    // all data names must have prefix prepended (i.e. connections would be prefix + "connections"
    
    DataMap dataMap;
    // datamap example:
    //dataMap.append(prefix + "someStatName", someStat);
    return dataMap;
}

std::string TDLambdaBrain::getType() {
    // return the type of this brain
    return "TDLambda";
}

void TDLambdaBrain::setInput(const int& inputAddress, const double& value) {
    if (inputAddress < nrInputValues) {
        inputValues[inputAddress] = value;
    }
    else {
        std::cout << "in Brain::setInput() : Writing to invalid input ("
            << inputAddress << ") - this brain needs more inputs!\nExiting"
            << std::endl;
        exit(1);
    }
}

double TDLambdaBrain::readInput(const int& inputAddress) {
    if (inputAddress < nrInputValues) {
        return inputValues[inputAddress];
    }
    else {
        std::cout << "in Brain::readInput() : Reading from invalid input ("
            << inputAddress << ") - this brain needs more inputs!\nExiting"
            << std::endl;
        exit(1);
    }
}

void TDLambdaBrain::setOutput(const int& outputAddress, const double& value) {
    if (outputAddress < nrOutputValues) {
        outputValues[outputAddress] = value;
    }
    else {
        std::cout << "in Brain::setOutput() : Writing to invalid output ("
            << outputAddress << ") - this brain needs more outputs!\nExiting"
            << std::endl;
        exit(1);
    }
}

double TDLambdaBrain::readOutput(const int& outputAddress) {
    if (outputAddress < nrOutputValues) {
        return outputValues[outputAddress];
    }
    else {
        std::cout << "in Brain::readOutput() : Reading from invalid output ("
            << outputAddress << ") - this brain needs more outputs!\nExiting"
            << std::endl;
        exit(1);
    }
}

void TDLambdaBrain::resetOutputs() {
    for (int i = 0; i < nrOutputValues; i++) {
        outputValues[i] = 0.0;
    }
}

void TDLambdaBrain::resetInputs() {
    for (int i = 0; i < nrInputValues; i++) {
        inputValues[i] = 0.0;
    }
}

void TDLambdaBrain::resetBrain() {
    resetInputs();
    resetOutputs();
    tdlambda.reset();
    inputValues[0] = -1.0;
}

///////////////////////////////////////////////////////////////////////////////////////////
// these functions need to be filled in if genomes are being used in this brain
///////////////////////////////////////////////////////////////////////////////////////////

// return a set of namespaces, MABE will insure that genomes with these names are created
// on organisms with these brains.
std::unordered_set<std::string> TDLambdaBrain::requiredGenomes() {
    return {"root::"};
}

void TDLambdaBrain::initializeGenomes(std::unordered_map<std::string, std::shared_ptr<AbstractGenome>> & _genomes) {
    _genomes["root::"]->fillRandom();
}

///////////////////////////////////////////////////////////////////////////////////////////
// these functions need to be filled in if this brain is direct encoded (in part or whole)
///////////////////////////////////////////////////////////////////////////////////////////

// Make a brain like the brain that called this function, using genomes and
// inheriting other elements from parent.
std::shared_ptr<AbstractBrain> TDLambdaBrain::makeBrainFrom(
    std::shared_ptr<AbstractBrain> parent,
    std::unordered_map<std::string, 
    std::shared_ptr<AbstractGenome>> & _genomes) {
   
    // in the default case, we assume geneticly encoded brains, so this just calls
    // the no parent version (i.e. makeBrain which builds from genomes)

    return makeBrain(_genomes);
}

// see makeBrainFrom, same thing, but for more then one parent
std::shared_ptr<AbstractBrain> TDLambdaBrain::makeBrainFromMany(
    std::vector<std::shared_ptr<AbstractBrain>> parents,
    std::unordered_map<std::string,
    std::shared_ptr<AbstractGenome>> & _genomes) {

    return makeBrain(_genomes);
}

