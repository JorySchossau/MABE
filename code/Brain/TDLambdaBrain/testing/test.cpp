#include <Utilities/gitversion.h>
#include <Global.h>
#include <module_factories.h>
#include <Genome/CircularGenome/CircularGenome.h>
#include <Brain/AbstractBrain.h>
#include <Brain/TDLambdaBrain/TDLambdaBrain.h>
#include <Brain/TDLambdaBrain/TDLambda.h>

#include <unordered_map>
#include <string>
#include "../TDUtils.h"

#include <valarray>

int main() {
  using namespace std;
  using namespace TD;
  TD::testTDLambda();
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
