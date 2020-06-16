#include <Utilities/gitversion.h>
#include <Utilities/Random.h>
#include <chrono>
#include <Global.h>
#include <module_factories.h>
#include <Genome/CircularGenome/CircularGenome.h>
#include <Brain/AbstractBrain.h>
#include <Brain/TDLambdaBrain/TDLambdaBrain.h>

#include <Brain/TDLambdaBrain/TDLambda.h>
#include <Brain/TDLambdaBrain/Grid.h>
#include <Brain/TDLambdaBrain/ConvBelt.h>

#include <unordered_map>
#include <string>
#include "../TDUtils.h"

#include <valarray>

using namespace std;
using namespace TD;

void testLambda() {
  valarray<int> dims = {3,3,4};
  Lambda brain({.dims=dims, .n_features=prod(dims), .n_actions=4, .alpha=0.1, .gamma=0.95, .epsilon=0.1, .lmbdaG=0.96, .lmbdaB=0.1, .lmbdaN=0.42});
  cout << brain.bin[valarray<int>({2,2,3})] << endl;
  auto qvalues = brain.predictPayoffsForAllActions({2,2});
  brain.plasticUpdate();
  brain.staticUpdate();
  cout << str(qvalues) << endl;
}

void testGrid() {
  vector<state_t> goals, holes;
  goals.push_back({3,11});
  for (int i(1); i<11; i++)
    holes.push_back({3,i});
  Grid grid({4,12}, goals, holes, {0,0});

  vector<state_t> finalpath;
  vector<int> numsteps;
  int num_episodes(600);
  int n_actions(4);
  valarray<int> dims = {grid.height,grid.width,n_actions};

  Lambda brain({.dims=dims, .n_features=prod(dims),
                .n_actions=n_actions,
                .alpha=0.1,
                .gamma=0.95,
                .epsilon=0.1,
                .lmbdaG=0.96, .lmbdaB=0.94, .lmbdaN=0.42});
  //cout << brain.bin[valarray<int>({2,2,3})] << endl;
  int time;
  state_t nextState;
  double reward;
  bool goal_achieved;
  for (int episoden(0); episoden<num_episodes; episoden++) {
    if ((episoden % 10) == 0) cout << ".";
    brain.reset();
    nextState = grid.reset();
    time = 0;
    while (true) {
      ++time;
      brain.sensoryState = nextState;
      brain.plasticUpdate();
      tie(nextState, reward, goal_achieved) = grid.step(brain.action);
      brain.reward = reward;
      if (goal_achieved or time == 200) break; // end episode
      else if (reward <= -10) { // fell in hole
        brain.sensoryState = nextState;
        brain.plasticUpdate();
        brain.reset();
        nextState = grid.reset();
      }
    }
    numsteps.push_back(time);
  }
  cout << endl;

  write("steps.csv",numsteps);
  cout << end(numsteps)[-1] << endl;
}

void testConvBelt() {
  ConvBelt belt({.goalRewards={0}, .randomize=false});

  belt.push_back({.features={{0,1},{0,1},{2,1},{0,0}},
                  .rewards={-1,-1,-1,0},
                  .solvable=true,
                  .startState=0,
                  .transitions=
                    {{0,0,2,3},
                     {0,0,0,0},
                     {2,0,2,3},
                     {3,3,3,3}} });

  belt.push_back({.features={{0,1},{0,1},{1,1},{0,0}},
                  .rewards={-1,-1,-1,0},
                  .solvable=true,
                  .startState=0,
                  .transitions=
                    {{0,0,2,3},
                     {0,0,0,0},
                     {2,0,2,3},
                     {3,3,3,3}} });

  belt.push_back({.features={{0,2},{0,2},{1,2},{0,0}},
                  .rewards={-1,-1,-1,-10},
                  .solvable=false,
                  .startState=0,
                  .transitions=
                    {{0,0,2,3},
                     {0,0,0,0},
                     {2,0,2,3},
                     {3,3,3,3}} });


  belt.push_back({.features={{0,2},{0,2},{2,1},{0,0}},
                  .rewards={-1,-1,-1,0},
                  .solvable=true,
                  .startState=0,
                  .transitions=
                    {{0,0,2,3},
                     {0,0,0,0},
                     {2,0,2,3},
                     {3,3,3,3}} });

  vector<state_t> finalpath;
  vector<int> numsteps;
  int num_episodes(6'000);
  int n_actions(4);
  valarray<int> dims = {3,3,n_actions};

  Lambda brain({.dims=dims, .n_features=prod(dims),
                .n_actions=n_actions,
                .alpha=0.1,
                .gamma=0.95,
                .epsilon=0.1,
                .lmbdaG=0.96, .lmbdaB=0.94, .lmbdaN=0.42});
  int time;
  state_t nextState;
  double reward;
  bool goal_achieved;
  for (int episoden(0); episoden<num_episodes; episoden++) {
    if ((episoden % 10) == 0) cout << ".";
    brain.reset();
    nextState = belt.reset();
    time = 0;
    while (true) {
      ++time;
      brain.sensoryState = nextState;
      brain.plasticUpdate();
      tie(nextState, reward, goal_achieved) = belt.step(brain.action);
      brain.reward = reward;
      if (belt.numPuzzlesToSolve == 0 or time == 600) break; // end episode
      //else if (reward <= -10) { // fell in hole
      //  brain.sensoryState = nextState;
      //  brain.plasticUpdate();
      //  brain.reset();
      //  nextState = belt.reset();
      //}
    }
    numsteps.push_back(time);
  }
  cout << endl;

  write("steps.csv",numsteps);
  cout << end(numsteps)[-1] << endl;
}

int main() {
  Random::getCommonGenerator().seed(std::chrono::system_clock::now().time_since_epoch().count());
  //testLambda();
  //testGrid();
  testConvBelt();
  //vector<vector<int>> vmatrix//(4,vector<int>(4))
  //{{0,0,2,3},
  // {0,0,0,0},
  // {2,0,2,3},
  // {3,3,3,3}};
  cout << "ran okay" << endl;
}

//int main() {
//  using namespace std;
//  shared_ptr<ParametersTable> PT = Parameters::root;
//  unordered_map<string, shared_ptr<AbstractGenome>> genomes;
//  genomes["root::"] = CircularGenome_genomeFactory(PT);
//  shared_ptr<AbstractBrain> templateBrain = TDLambdaBrain_brainFactory(2, 1, PT);
//  // initialize the genome
//  auto genomeHandler = genomes["root::"]->newHandler(genomes["root::"], true);
//  for (int i=1; i<=36; i++) {
//    genomeHandler->writeDouble(0.5, 0.0, 1.0);
//    genomeHandler->advanceIndex();
//  }
//  shared_ptr<AbstractBrain> brain = templateBrain->makeBrain(genomes);
//  brain->update();
//  cout << "ran okay" << endl;
//  return(0);
//}
