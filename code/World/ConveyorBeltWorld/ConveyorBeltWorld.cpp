//  MABE is a product of The Hintze Lab @ MSU
//     for general research information:
//         hintzelab.msu.edu
//     for MABE documentation:
//         github.com/Hintzelab/MABE/wiki
//
//  Copyright (c) 2015 Michigan State University. All rights reserved.
//     to view the full license, visit:
//         github.com/Hintzelab/MABE/wiki/License

// Evaluates agents on how many '1's they can output. This is a purely fixed
// task
// that requires to reactivity to stimuli.
// Each correct '1' confers 1.0 point to score, or the decimal output determined
// by 'mode'.

#include <World/ConveyorBeltWorld/ConveyorBeltWorld.h>
#include <World/ConveyorBeltWorld/ConveyorBeltWorld.h>
#include <Utilities/Utilities.h>
#include <math.h> // round, trunc
//#include <Brain/TDLambdaBrain/TDUtils.h>

std::shared_ptr<ParameterLink<int>> ConveyorBeltWorld::modePL = Parameters::register_parameter( "WORLD_CONVEYORBELT-mode", 0, "0 = bit outputs before adding, 1 = add outputs");
std::shared_ptr<ParameterLink<int>> ConveyorBeltWorld::numThreadsPL = Parameters::register_parameter("WORLD_CONVEYORBELT-numThreads", 0, "Number of threads to use (each member of the population is evaluated on a single thread) 0 implies max physical cores available assuming hyperthreading or equivalent enabled (logical cores / 2).");
std::shared_ptr<ParameterLink<int>> ConveyorBeltWorld::trialsPL = Parameters::register_parameter("WORLD_CONVEYORBELT-trials", 200, "Number of trials.");
std::shared_ptr<ParameterLink<int>> ConveyorBeltWorld::trialLengthPL = Parameters::register_parameter("WORLD_CONVEYORBELT-trialLength", 2'000, "Length of 1 trial.");
std::shared_ptr<ParameterLink<std::string>> ConveyorBeltWorld::groupNamePL = Parameters::register_parameter("WORLD_CONVEYORBELT_NAMES-groupNameSpace", (std::string) "root::", "namespace of group to be evaluated");
std::shared_ptr<ParameterLink<std::string>> ConveyorBeltWorld::brainNamePL = Parameters::register_parameter( "WORLD_CONVEYORBELT_NAMES-brainNameSpace", (std::string) "root::", "namespace for parameters used to define brain");
std::shared_ptr<ParameterLink<std::string>> ConveyorBeltWorld::dimensionsPL = Parameters::register_parameter( "WORLD_CONVEYORBELT-dimensions", (std::string) "3,3,4", "csv list of maximum dimensions of freedom for this world (salient inputs, and their ranges) for example, \"3,3,4\" for a 3x3 grid where agent knows x,y location and can have 4 actions or \"2,3\" for a world that presents binary inputs (2) and allows 3 actions.");
std::shared_ptr<ParameterLink<std::string>> ConveyorBeltWorld::goalRewardsPL = Parameters::register_parameter( "WORLD_CONVEYORBELT-goalRewards", (std::string) "0", "csv list of rewarding values (typically 0, but may include positive values)");
std::shared_ptr<ParameterLink<bool>> ConveyorBeltWorld::randomizePL = Parameters::register_parameter( "WORLD_CONVEYORBELT-randomize", true, "controls order of items on the belt.");
std::shared_ptr<ParameterLink<bool>> ConveyorBeltWorld::logActionsPL = Parameters::register_parameter( "WORLD_CONVEYORBELT-logActions", false, "controls logging of actions taken during lifetime.");
std::shared_ptr<ParameterLink<bool>> ConveyorBeltWorld::logRewardsPL = Parameters::register_parameter( "WORLD_CONVEYORBELT-logRewards", false, "controls logging of rewards received during lifetime.");
std::shared_ptr<ParameterLink<bool>> ConveyorBeltWorld::logSensorsPL = Parameters::register_parameter( "WORLD_CONVEYORBELT-logSensors", false, "controls logging of sensor states during lifetime.");

std::valarray<int> ConveyorBeltWorld::dimensions;
ConveyorBelt::ConveyorBeltParams* ConveyorBeltWorld::pParameters = nullptr;
int ConveyorBeltWorld::num_trials = 0;
int ConveyorBeltWorld::trial_length = 0;

template <class T>
void printCSV(std::ostream& os, const T& arr){
  os << '[';
  for(int i(0); i<arr.size()-1; ++i)
    os << arr[i] << ',';
  os << arr[arr.size()-1] << ']';
}
template<class T>
auto operator<<( std::ostream& os, const std::valarray<T>& arr ) -> std::ostream& {
  printCSV(os, arr);
  return os;
}
template<class T,std::size_t N>
auto operator<<( std::ostream& os, const std::array<T, N>& arr ) -> std::ostream& {
  printCSV(os, arr);
  return os;
}
template<class T, class A>
auto operator<<( std::ostream& os, const std::vector<T, A>& arr ) -> std::ostream& {
  printCSV(os, arr);
  return os;
}

std::vector<Puzzle> ConveyorBeltWorld::puzzles {
  {.features={{0,1},{0,1},{2,1},{0,0}},
   .rewards={-1,-1,-1,0},
   .solvable=true,
   .startState=0,
   .transitions=
     {{0,0,2,3},
      {0,0,0,0},
      {2,0,2,3},
      {3,3,3,3}} }
  ,
  {.features={{0,1},{0,1},{1,1},{0,0}},
   .rewards={-1,-1,-1,-3},
   .solvable=false,
   .startState=0,
   .transitions=
     {{0,0,2,3},
      {0,0,0,0},
      {2,0,2,3},
      {3,3,3,3}} }
  ,
  {.features={{0,1},{0,1},{2,1},{0,0}},
   .rewards={-1,-1,-1,0},
   .solvable=true,
   .startState=0,
   .transitions=
     {{0,0,2,3},
      {0,0,0,0},
      {2,0,2,3},
      {3,3,3,3}} }
};

// ctor, constructor
ConveyorBeltWorld::ConveyorBeltWorld(std::shared_ptr<ParametersTable> PT_)
    : AbstractWorld(PT_) {

  std::vector<int> intdimensions;
  convertCSVListToVector<int>(std::string(dimensionsPL->get()), intdimensions);
  ConveyorBeltWorld::dimensions = std::valarray<int>( intdimensions.data(), intdimensions.size() );

  // determine if working with an RL brain and set dimensions accordingly, if so.
  std::string brainType = capitalize(Parameters::getStringLink("BRAIN-brainType",PT)->get());
  //std::cout << "type: " << std::string(brainType) << std::endl;
  std::string brainDimensionsParamString = std::string("BRAIN_") + brainType + std::string("-dimensions");
  bool isRLBrain = (std::string)PT->getParameterType(brainDimensionsParamString) != "FAIL";
  //std::cout << "is RL brain: " << isRLBrain << std::endl;

  if (isRLBrain) {
    Parameters::getStringLink(brainDimensionsParamString,PT)->set(std::string(dimensionsPL->get()));
  }

  // get goalRewards and randomizing
  std::vector<double> goalRewards;
  convertCSVListToVector<double>(std::string(goalRewardsPL->get()), goalRewards);
  bool randomize = randomizePL->get();
  if (pParameters == nullptr)
    pParameters = new ConveyorBelt::ConveyorBeltParams{.goalRewards=goalRewards, .randomize=randomize};

  ConveyorBeltWorld::num_trials = trialsPL->get();
  ConveyorBeltWorld::trial_length = trialLengthPL->get();
  
  logActions = logActionsPL->get();
  logRewards = logRewardsPL->get();
  logSensors = logSensorsPL->get();

  // columns to be added to ave file
  popFileColumns.clear();
  popFileColumns.push_back("score");
  popFileColumns.push_back("score_VAR"); // specifies to also record the
                                         // variance (performed automatically
                                         // because _VAR)
  popFileColumns.push_back("eaten");
}

auto ConveyorBeltWorld::evaluate_single_thread(int analyze, int visualize, int debug) -> void {
  /* keep looping and evaluating 1 organism per loop
   * and finish when there are no more orgs to evaluate
   * Workflow:
   * * get a new org ID from org_ids_to_evaluate queue
   * * if no orgs left, then finish thread and return
   * * otherwise process organism at index ID we got
   * * repeat
   */
  while(true) {
    int org_to_evaluate; // -1=finish thread, otherwise = ID of org to evaluate

    // respecting multithreading data race conditions,
    // get the next organism that has yet to be evaluated
    org_ids_guard.lock();
    if (org_ids_to_evaluate.size() == 0) org_to_evaluate = -1;
    else {
      org_to_evaluate = org_ids_to_evaluate.front();
      org_ids_to_evaluate.pop();
    }
    org_ids_guard.unlock();

    // if no orgs to evaluate found, then finish thread
    if (org_to_evaluate == -1) return;

    // get reference to the organism to evaluate
    std::vector<std::shared_ptr<Organism>>& population = *population_ptr;
    std::shared_ptr<Organism> org = population[org_to_evaluate];

    auto& brain = *org->brains[brainNamePL->get(PT)]; // convenience and readability
    auto& world = experiences[org_to_evaluate]; // convenience and readability
    state_t nextState;
    double
      reward,
      output;
    int
      action,
      time,
      i; // for loop iterator
    bool goal_achieved;

    // we use the OpenAI Gym RL format (state=env.reset(), (state,reward,goal)=env.step())
    world.numsteps.resize(ConveyorBeltWorld::num_trials);
    world.numsolved.resize(ConveyorBeltWorld::num_trials);
    world.reset(); // clean up after last agent used this memory space
    for (int trialn(0); trialn<ConveyorBeltWorld::num_trials; trialn++) {
      brain.resetBrain();
      brain.setInput(0, -1.0); // initial reward
      nextState = world.env.reset();
      time = 0;
      while (true) {
        ++time;
        // set input(s)
        for (i=1; i<brain.nrInputValues; ++i) { // also, i <= nextState.size()
          //brain.setInput(i, reinterpret_cast<double*>(&nextState[i-1])[0]); // old cast version
          brain.setInput(i, nextState[i-1]);  // non-cast version
        }
        // allow brain to compute action
        brain.update();
        // build action from output(s)
        output = brain.readOutput(0);
        //action = reinterpret_cast<int*>(&output)[0]; // old cast version
        action = int(std::trunc(std::round(output))); // non-cast version
        // compute change in environment due to agent action
        tie(nextState, reward, goal_achieved) = world.env.step(action);
        if (logActions)
          world.actions += std::to_string(action);
        if (logRewards)
          world.rewards += " "+std::to_string(int(reward));
        if (logSensors)
          world.sensors += " "+std::to_string(nextState[0])+std::to_string(nextState[1]);
        // provide reward for previous action
        brain.setInput(0,reward);
        if (reward == 0.0) ++world.eaten;
        if (reward < -1.0) --world.eaten;
        if (goal_achieved or time == ConveyorBeltWorld::trial_length) break; // end episode
        //else if (reward <= -10) { // if dying and will reset...
        //  for (auto& puzzle : world.env.puzzles)
        //    puzzle.solved = false; // maximum penalty
        //  time = ConveyorBeltWorld::trial_length; // maximum penalty
        //  world.numsolved = 0; // maximum penalty
        //  world.numsteps = ConveyorBeltWorld::trial_length; // maximum penalty
        //  trialn = ConveyorBeltWorld::num_trials; // finish all trials
        //  break;
        //  //// allow brain to process death before resetting
        //  //for (i=1; i<brain.nrInputValues; ++i) // also, i <= nextState.size()
        //  //  brain.setInput(i, reinterpret_cast<double*>(&nextState[i])[0]); // TODO
        //  //brain.update();
        //  //// now reset the world
        //  //brain.reset();
        //  //nextState = world.env.reset();
        //}
      }
      for (auto& puzzle : world.env.puzzles) {
        if (puzzle.solved)
          ++world.numsolved[trialn];
      }
      world.numsteps[trialn] = time;
    }

    int stepsTaken = world.numsteps.sum();
    int maxStepsPossible = ConveyorBeltWorld::trial_length * ConveyorBeltWorld::num_trials;
    int steps_score = maxStepsPossible - stepsTaken;
    double scaled_score =  double(steps_score) / double(maxStepsPossible) + 1.0;

    int numsolved_all_trials = world.numsolved.sum();
    // count how many were maximally solvable in a single trial
    int numsolvable_one_trial(0);
    for (auto& puzzle : world.env.puzzles)
      if (puzzle.solvable) ++numsolvable_one_trial;
    // then calculate maximum possible accumulated solved score given num_trials
    int numsolvable_all_trials = numsolvable_one_trial * ConveyorBeltWorld::num_trials ;
    double scaled_solved_score = double(numsolved_all_trials) / double(numsolvable_all_trials) + 1.0;

    double score = scaled_solved_score * scaled_score;

    //org->dataMap.set("score", scaled_score);
    //org->dataMap.set("score", scaled_solved_score);
    //org->dataMap.set("score", scaled_score * scaled_solved_score);
    //org->dataMap.set("score", score);
    //org->dataMap.set("score", score - (world.actions.size() * 0.20));
    //org->dataMap.set("score", numsolved_all_trials);
    org->dataMap.set("score", world.eaten);
    org->dataMap.set("eaten", world.eaten);

    if (logActions) {
      org->dataMap.set("actions", world.actions);
      org->dataMap.setOutputBehavior("actions",DataMap::FIRST);
    }
    if (logRewards) {
      org->dataMap.set("rewards", world.rewards);
      org->dataMap.setOutputBehavior("rewards",DataMap::FIRST);
    }
    if (logSensors) {
      org->dataMap.set("sensors", world.sensors);
      org->dataMap.setOutputBehavior("sensors",DataMap::FIRST);
    }
  }
}


void ConveyorBeltWorld::evaluate(std::map<std::string, std::shared_ptr<Group>> &groups, int analyze, int visualize, int debug) {
  // evaluate() is a MABE API function called automatically for each generation
  // We invoke an 'evaluate_single_thread(org)' for each organism
  // allowing parallel computation.
  // All data local to a single organism is contained in an Experience class instance
  // such as position, state, reward, etc.
  // The visualize and debug flags are up for you to decide
  // what they mean, and you can set in settings.

  //if (Global::update == 3000) {
  //  ConveyorBeltWorld::puzzles.pop_back();
  //  ConveyorBeltWorld::puzzles.push_back( 
  //    {.features={{0,2},{0,2},{1,2},{0,0}},
  //     .rewards={-1,-1,-1,0},
  //     .solvable=true,
  //     .startState=0,
  //     .transitions=
  //       {{0,0,2,3},
  //        {0,0,0,0},
  //        {2,0,2,3},
  //        {3,3,3,3}} }
  //      );
  //}

  int popSize = groups[groupNamePL->get(PT)]->population.size();
  // save population to World-wide pointer so all threads have access to it
  population_ptr = &groups[groupNamePL->get(PT)]->population;
  // populate list of ids with linear increasing list of indices
  // each thread will claim and remove one id and evaluate that organism
  for (int i=0; i<popSize; i++) org_ids_to_evaluate.push(i);
  // create pool of Experience instances, 1 for each organism
  // but we can reuse the pool if it already exists.
  // ensure we have enough experience instances for all agents
  while (experiences.size() < popSize) {
    experiences.emplace_back(Experience(*pParameters));
    auto& env = end(experiences)[-1].env;
    // create/initialize puzzles
    for (Puzzle& puzzle : ConveyorBeltWorld::puzzles)
      env.push_back(puzzle);
  }
  // create and start the thread pool...
  // if no threads set, then set to max cores found
  int num_threads = numThreadsPL->get(PT);
  if (num_threads == 0) num_threads = std::thread::hardware_concurrency()/2;
  // create/start each thread to run evaluate_single_thread()
  for (int i=0; i<num_threads; i++) threads.push_back(std::thread(&ConveyorBeltWorld::evaluate_single_thread, this, analyze, visualize, debug));
  // wait for all threads to finish
  for (auto& thread:threads) thread.join();
  // we can't reuse threads in a good way, so just deconstruct them
  threads.clear();
}

std::unordered_map<std::string, std::unordered_set<std::string>>
ConveyorBeltWorld::requiredGroups() {
  return {{groupNamePL->get(PT),
        {"B:" + brainNamePL->get(PT) + "," +
            std::to_string(2+1) + "," + // +1 because reward from t-1, always at pos 0
            std::to_string(1)}}};
  // requires a root group and a brain (in root namespace) and no addtional
  // genome,
  // the brain must have 1 input, and the variable numberOfOutputs outputs
}

