//  MABE is a product of The Hintze Lab @ MSU
//     for general research information:
//         hintzelab.msu.edu
//     for MABE documentation:
//         github.com/Hintzelab/MABE/wiki
//
//  Copyright (c) 2015 Michigan State University. All rights reserved.
//     to view the full license, visit:
//         github.com/Hintzelab/MABE/wiki/License

#pragma once					// directive to insure that this .h file is only included one time

// AbstractBrain defines all the basic function templates for brains
#include <Brain/AbstractBrain.h>
#include <valarray>

// If your brain is (or maybe) constructed using a genome, you must include AbstractGenome.h
#include <Genome/AbstractGenome.h>
#include <Brain/TDLambdaBrain/TDLambdaBrain.h>
#include <Brain/TDLambdaBrain/TDLambda.h> // raw td-lambda functionality ported from our py implementation

class TDLambdaBrain : public AbstractBrain {

public:
    static std::shared_ptr<ParameterLink<std::string>> dimensionsPL;
    static std::shared_ptr<ParameterLink<bool>> use_confidencePL;
    static std::shared_ptr<ParameterLink<double>> no_confidence_random_actionPL;

    TDLambdaBrain() = delete;

    TDLambdaBrain(int ins, int outs, std::shared_ptr<ParametersTable> PT_);

    virtual ~TDLambdaBrain() = default;

    virtual void update();

    // make a copy of the brain that called this
    virtual std::shared_ptr<AbstractBrain> makeCopy(std::shared_ptr<ParametersTable> PT_);

    // Make a brain like the brain that called this function, using genomes and initalizing other elements.
    virtual std::shared_ptr<AbstractBrain> makeBrain(std::unordered_map<std::string, std::shared_ptr<AbstractGenome>>& _genomes);

    virtual std::string description(); // returns a desription of this brain in it's current state

    virtual DataMap getStats(std::string& prefix); // return a vector of DataMap of stats from this brain

    virtual std::string getType(); // return the type of this brain

    virtual void setInput(const int& inputAddress, const double& value);

    virtual double readInput(const int& inputAddress);

    virtual void setOutput(const int& outputAddress, const double& value);

    virtual double readOutput(const int& outputAddress);

    virtual void resetOutputs();

    virtual void resetInputs();

    virtual void resetBrain();

    // setRecordActivity and setRecordFileName provide a standard way to set up brain
    // activity recoding. How and when the brain records activity is up to the brain developer
    virtual void inline setRecordActivity(bool _recordActivity) {
        recordActivity = _recordActivity;
    }

    virtual void inline setRecordFileName(std::string _recordActivityFileName) {
        recordActivityFileName = _recordActivityFileName;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // these functions need to be filled in if genomes are being used in this brain
    ///////////////////////////////////////////////////////////////////////////////////////////

    virtual std::unordered_set<std::string> requiredGenomes(); // does this brain use any genomes
    
    // initializeGenomes can be used to randomize the genome and/or insert start codons
    virtual void initializeGenomes(std::unordered_map<std::string, std::shared_ptr<AbstractGenome>>& _genomes);

    ///////////////////////////////////////////////////////////////////////////////////////////
    // these functions need to be filled in if this brain is direct encoded (in part or whole)
    ///////////////////////////////////////////////////////////////////////////////////////////

    // Make a brain like the brain that called this function, using genomes and
    // inheriting other elements from parent.
    // in the default case, we assume geneticly encoded brains, so this just calls
    // the no parent version (i.e. makeBrain which builds from genomes)
    virtual std::shared_ptr<AbstractBrain> makeBrainFrom(
        std::shared_ptr<AbstractBrain> parent,
        std::unordered_map<std::string,
        std::shared_ptr<AbstractGenome>>& _genomes);

    // see makeBrainFrom, this can take more then one parent
    virtual std::shared_ptr<AbstractBrain> makeBrainFromMany(
        std::vector<std::shared_ptr<AbstractBrain>> parents,
        std::unordered_map<std::string,
        std::shared_ptr<AbstractGenome>>&_genomes);

    // CUSTOM PROPERTIES & METHODS
    std::valarray<int> dimensions;
    TD::Lambda tdlambda;

    bool use_confidence;
    double no_confidence_random_action;
};

inline std::shared_ptr<AbstractBrain> TDLambdaBrain_brainFactory(int ins, int outs, std::shared_ptr<ParametersTable> PT) {
    return std::make_shared<TDLambdaBrain>(ins, outs, PT);
}
