//  MABE is a product of The Hintze Lab @ MSU
//     for general research information:
//         hintzelab.msu.edu
//     for MABE documentation:
//         github.com/Hintzelab/MABE/wiki
//
//  Copyright (c) 2015 Michigan State University. All rights reserved.
//     to view the full license, visit:
//         github.com/Hintzelab/MABE/wiki/License

#include "Parameters.h"

#include <regex>

extern const char *gitversion; // from gitversion.h autogenerated by mbuild.py

std::shared_ptr<ParametersTable> Parameters::root;

long long ParametersTable::nextTableID = 0;

template <> inline const bool ParametersEntry<bool>::getBool() { return get(); }

template <> inline const std::string ParametersEntry<std::string>::getString() {
  return get();
}

template <> inline const int ParametersEntry<int>::getInt() { return get(); }

template <> inline const double ParametersEntry<double>::getDouble() {
  return get();
}

std::shared_ptr<ParameterLink<bool>>
Parameters::getBoolLink(const std::string &name,
                        std::shared_ptr<ParametersTable> table) {
  auto entry = table->lookupBoolEntry(name);
  auto newLink = std::make_shared<ParameterLink<bool>>(name, entry, table);
  return newLink;
}

std::shared_ptr<ParameterLink<std::string>>
Parameters::getStringLink(const std::string &name,
                          std::shared_ptr<ParametersTable> table) {
  auto entry = table->lookupStringEntry(name);
  auto newLink =
      std::make_shared<ParameterLink<std::string>>(name, entry, table);
  return newLink;
}

std::shared_ptr<ParameterLink<int>>
Parameters::getIntLink(const std::string &name,
                       std::shared_ptr<ParametersTable> table) {
  auto entry = table->lookupIntEntry(name);
  auto newLink = std::make_shared<ParameterLink<int>>(name, entry, table);
  return newLink;
}

std::shared_ptr<ParameterLink<double>>
Parameters::getDoubleLink(const std::string &name,
                          std::shared_ptr<ParametersTable> table) {
  auto entry = table->lookupDoubleEntry(name);
  auto newLink = std::make_shared<ParameterLink<double>>(name, entry, table);
  return newLink;
}

void Parameters::parseFullParameterName(const std::string &full_name,
                                        std::string &name_space_name,
                                        std::string &category_name,
                                        std::string &parameter_name) {
  // no need for error checking, since this is internal??
  // Then it should be a private member at least.

  static const std::regex param(R"(^(.*::)?(\w+)-(\w+)$)");
  std::smatch m;
  if (std::regex_match(full_name, m, param)) {
    name_space_name = m[1].str();
    category_name = m[2].str();
    parameter_name = m[3].str();
  } else {
    // this doesn't distinguish between command line parameters and setting file
    // parameters
    std::cout << "  ERROR! :: found misformatted parameter \"" << full_name
              << "\"\n  Parameters must have format: [category]-[name] "
                 "or [name space][category]-[name]"
              << std::endl;
    exit(1);
  }
}

void Parameters::readCommandLine(
    int argc, const char **argv,
    std::unordered_map<std::string, std::string> &param_name_values,
    std::vector<std::string> &file_list, bool &save_files) {

  const std::string usage_message =
      R"( [-f <file1> <file2> ...] [-p <parameter name/value pairs>] [-s]
                                    
  -f : "load files" - list of settings files to be loaded.
       Parameters in later files overwrite parameters in earlier files.

  -p : "set parameters" - list of parameter/name pairs. 
        e.g. "-p GLOBAL-updates 100 GLOBAL-popSize 200" would set MABE to 
        run for 100 updates with a population size of 200. Parameters set 
        on the command line overwrite parameters from files.

  -s : "save" - save settings files.

  -l : "create population loading script"
        This creates a default file "population_loader.plf" that contains 
        the script for loading the initial population. See file or wiki
        for usage examples. Note: using -l will ignore all other command 
        line arguments.

  -v : "version" - show git commit hash from when compiled

)";
  const std::string default_plf_contents = R"(

MASTER = default 100 # by default :) 

# At the moment default and random mean the same thing. This will change
#MASTER = random 100

# an example loading a single file generated by default-archivist
#MASTER = 'snapshot_organisms_10.csv' 

# another example
#some_var = greatest 5 by ID from { '*.csv' }
#MASTER = collapse some_var
 
# an example with exact match
#MASTER = match ID where 3 from 'snapshot_organisms_0.csv' 

# a convoluted example :P
#another_var = greatest 5 by ID from { '*.csv' } 
#still_another_var = greatest 2 by ID from { '*.csv' : least 10 by score_AVE from { */LOD_*.csv : */SSWD_*.csv } : '*/*.csv' } 
#MASTER = collapse { any 3 from { another_var : still_another_var } }

)";

  static const std::regex command_line_argument_flag(R"(-([a-z]))");
  bool dont_run = false;
  for (int i = 1; i < argc; i++) {
    std::smatch m;
    std::string argument(argv[i]);
    if (!std::regex_match(argument, m, command_line_argument_flag)) {
      std::cout << "Error : out of context argument \"" << argument
                << "\" on command line" << std::endl;
      exit(1);
    }
    switch (m[1].str()[0]) {
    case 'h':
      std::cout << MABE_pretty_logo;
      std::cout << "Usage: " << argv[0] << usage_message << std::endl;
      dont_run = true;
      break;
    case 'v':
      std::cout << gitversion << std::endl;
      dont_run = true;
      break;
    case 'l': {
      std::ofstream plf_file("population_loader.plf");
      plf_file << default_plf_contents;
      plf_file.close();
      std::cout << "created population loader file "
                   "\"population_loader.plf\""
                << std::endl
                << "Change parameter GLOBAL-initPop to this file name (or "
                   "any other .plf file name) to load a specific population"
                << std::endl;
      dont_run = true;
      break;
    }
    case 's':
      save_files = true;
      break;
    case 'f':
      for (i++; i < argc; i++) {
        std::string filename(argv[i]);
        if (std::regex_match(filename, command_line_argument_flag)) {
          i--;
          break;
        }
        file_list.push_back(filename);
      }
      break;

    case 'p':
      for (i++; i < argc - 1; i += 2) {
        std::string param_name(argv[i]), param_value(argv[i + 1]);
        if (std::regex_match(param_name, command_line_argument_flag)) {
          i--;
          break;
        }
        if (param_name_values.find(param_name) != param_name_values.end()) {
          std::cout << "  ERROR :: Parameter \"" << param_name
                    << "\" is defined more then once on the command "
                       "line.\nExiting.\n";
          exit(1);
        }
        if (std::regex_match(param_value, command_line_argument_flag)) {
          std::cout << "  ERROR :: Parameter \"" << param_name
                    << "\" is defined on command line with out a "
                       "value.\nExiting.\n";
          exit(1);
        }
        param_name_values[param_name] = param_value;
      }
      break;

    default:
      std::cout << "  Error on command line. Unrecognized option -"
                << m[1].str()[0] << ". Exiting." << std::endl;
      exit(1);
    }
  }
  if (dont_run)
    exit(0);
} // end Parameters::readCommandLine()

std::unordered_map<std::string, std::string>
Parameters::readParametersFile(const std::string &file_name) {

  std::unordered_map<std::string, std::string> config_file_list;

  std::ifstream file(file_name); // open file named by file_name
  if (!file.is_open()) {
    std::cout << "  ERROR! unable to open file \"" << file_name
              << "\".\nExiting.\n";
    exit(1);
  }
  std::string dirty_line;
  std::string category_name;
  std::string name_space_name;

  while (std::getline(file, dirty_line)) {

    std::regex comments("#.*");
    std::string line = std::regex_replace(dirty_line, comments, "");

    std::regex empty_lines(R"(^\s*$)");
    if (std::regex_match(line, empty_lines))
      continue;

    {
      std::regex category(R"(^\s*%\s*(\w*)\s*$)");
      std::smatch m;
      if (std::regex_match(line, m, category)) {
        category_name = m[1].str();
        continue;
      }
    }

    {
      std::regex name_space_open(R"(^\s*\+\s*(\w+::)\s*$)");
      std::smatch m;
      if (std::regex_match(line, m, name_space_open)) {
        name_space_name += m[1].str();
        continue;
      }
    }

    {
      std::regex name_space_close(R"(^\s*-\s*$)");
      std::regex name_space_remove_last(R"(\w+::$)");
      std::smatch m;
      if (std::regex_match(line, m, name_space_close)) {
        if (name_space_name.empty()) {
          std::cout << " Error: no namespace to descend, already at root:: "
                       "namespace. "
                    << std::endl;
          exit(1);
        }
        name_space_name =
            std::regex_replace(name_space_name, name_space_remove_last, "");
        continue;
      }
    }

    {
      std::regex name_value_pair(R"(^\s*([\S]+)\s*=\s*(\S?.*\S)\s*$)");
      std::smatch m;
      if (std::regex_match(line, m, name_value_pair)) {
        auto name = name_space_name;
        name.append((category_name.empty() ? "" : (category_name + "-")))
            .append(m[1].str());
        if (config_file_list.find(name) != config_file_list.end()) {
          std::cout << "  Error: \"" << name
                    << "\" is defined more then once in file: \"" << file_name
                    << "\".\n exiting.\n";
          exit(1);
        }
        config_file_list[name] = m[2].str();
        continue;
      }
    }

    std::cout << " Error: unrecognised line\n" << dirty_line << " in file "
              << file_name << "\nSee "
                              "https://github.com/Hintzelab/MABE/wiki/"
                              "Parameters-Name-Space  for correct usage."
              << std::endl;
        exit(1);
  }

  return config_file_list;
} // end Parameters::readParametersFile

bool Parameters::initializeParameters(int argc, const char *argv[]) {

  if (root == nullptr) {
    root = ParametersTable::makeTable();
  }

  std::unordered_map<std::string, std::string> command_line_list;
  std::vector<std::string> fileList;

  bool saveFiles = false;
  Parameters::readCommandLine(argc, argv, command_line_list, fileList,
                              saveFiles);

  std::string workingNameSpace, workingCategory, workingParameterName;

  for (const auto &fileName : fileList) { // load all files in order - this
                                          // order is arbitrary if wildcarded?
    std::unordered_map<std::string, std::string> file_list =
        Parameters::readParametersFile(fileName);
    for (const auto &file : file_list) {
      parseFullParameterName(file.first, workingNameSpace, workingCategory,
                             workingParameterName);
      if (!root->getParameterTypeAndSetParameter(
              workingCategory + "-" + workingParameterName,
              file.second, workingNameSpace, true)) {
        std::cout << (saveFiles ? "   WARNING" : "  ERROR")
                  << " :: while reading file \"" << fileName << "\" found \""
                  << workingNameSpace << workingCategory << "-"
                  << workingParameterName << ".\n      But \""
                  << workingCategory << "-" << workingParameterName
                  << "\" is not a registered parameter!" << std::endl
                  << (saveFiles ? "      This parameter will not be saved "
                                  "to new files."
                                : "  Exiting.")
                  << std::endl;
        if (!saveFiles) {
          exit(1);
        }
      }
    }
  }
  for (const auto &command :
       command_line_list) { // load command line parameters last
    parseFullParameterName(command.first, workingNameSpace, workingCategory,
                           workingParameterName);
    if (!root->getParameterTypeAndSetParameter(
            workingCategory + "-" + workingParameterName,
            command.second, workingNameSpace, true)) {
      std::cout << (saveFiles ? "   WARNING" : "  ERROR")
                << " :: while reading command line found \"" << workingNameSpace
                << workingCategory << "-" << workingParameterName
                << ".\n      But \"" << workingCategory << "-"
                << workingParameterName << "\" is not a registered parameter!"
                << std::endl
                << (saveFiles ? "      This parameter will not be saved "
                                "to new files."
                              : "  Exiting.")
                << std::endl;
      if (!saveFiles) {
        exit(1);
      }
    }
  }
  return saveFiles;
} // end  Parameters::initializeParameters

void Parameters::saveSettingsFile(const std::string &name_space,
                                  std::stringstream &file,
                                  std::vector<std::string> category_list,
                                  int max_line_length, int comment_indent,
                                  bool also_children, int name_space_level) {

  if (root->lookupTable(name_space)->neverSave)
    return;

  std::map<std::string, std::vector<std::string>> sortedParameters;
  root->lookupTable(name_space)->parametersToSortedList(sortedParameters);

  std::string current_indent;

  /*   *** Will uncomment and fix when namespaces are actually needed
      auto name_space_parts = nameSpaceToNameParts(name_space);
          for (int i = 0; i < name_space_level; i++) {
        currentIndent += "  ";
        name_space_parts.erase(name_space_parts.begin());
      }

      if (name_space_parts.size() > 0) {
        for (auto p : name_space_parts) {
          file << currentIndent << "+ " << p.substr(0, p.size() - 2) << "\n";
          name_space_level++;
          currentIndent += "  ";
        }
      }
   */
  if (!category_list.empty()  && category_list[0] == "-") {
    if (sortedParameters.find("GLOBAL") != sortedParameters.end() &&
        !(find(category_list.begin(), category_list.end(), "GLOBAL") !=
          category_list.end())) {
      file << current_indent << "% GLOBAL"
           << "\n";
      for (auto const &parameter : sortedParameters["GLOBAL"]) {
        printParameterWithWraparound(file, current_indent + "  ", parameter,
                                     max_line_length, comment_indent);
        //					file <<
        // currentIndent << "  " << parameter << "\n";
      }
      file << "\n";
    }
  } else { // write parameters to file.
    if (sortedParameters.find("GLOBAL") != sortedParameters.end() &&
        find(category_list.begin(), category_list.end(), "GLOBAL") !=
            category_list.end()) {
      file << current_indent << "% GLOBAL"
           << "\n";
      for (auto const &parameter : sortedParameters["GLOBAL"]) {
        printParameterWithWraparound(file, current_indent + "  ", parameter,
                                     max_line_length, comment_indent);
        //					file <<
        // currentIndent << "  " << parameter << "\n";
      }
      file << "\n";
    }
  }
  sortedParameters.erase("GLOBAL");

  for (auto group : sortedParameters) {
    bool saveThis = false;
    if (!category_list.empty()  && category_list[0] != "-") {
      for (auto cat : category_list) {
        if (static_cast<int>(group.first.size()) >=
            (static_cast<int>(cat.size())) - 1) {
          if (group.first == cat) {
            saveThis = true;
          } else {
            if (static_cast<int>(cat.size()) > 0 &&
                cat[(static_cast<int>(cat.size())) - 1] == '*') {
              if (group.first.substr(0, cat.size() - 1) ==
                  cat.substr(0, cat.size() - 1)) {
                saveThis = true;
              }
            }
          }
        }
      }
    } else {
      saveThis = true;
      for (auto cat : category_list) {
        if (static_cast<int>(group.first.size()) >=
            (static_cast<int>(cat.size())) - 1) {
          if (group.first == cat) {
            saveThis = false;
          } else {
            if (static_cast<int>(cat.size()) > 0 &&
                cat[(static_cast<int>(cat.size())) - 1] == '*') {
              if (group.first.substr(0, cat.size() - 1) ==
                  cat.substr(0, cat.size() - 1)) {
                saveThis = false;
              }
            }
          }
        }
      }
    }
    if (saveThis) {
      file << current_indent << "% " << group.first << "\n";
      for (auto const &parameter : group.second) {
        printParameterWithWraparound(file, current_indent + "  ", parameter,
                                     max_line_length, comment_indent);
        //					file << currentIndent << "  " <<
        //parameter <<
        //"\n";
      }
      file << "\n";
    }
  }

  if (also_children) {
    std::vector<std::shared_ptr<ParametersTable>> checklist =
        root->lookupTable(name_space)->getChildren();
    sort(checklist.begin(), checklist.end());
    for (auto const &c : checklist) {
      saveSettingsFile(c->getTableNameSpace(), file, category_list,
                       max_line_length, comment_indent, true, name_space_level);
    }
  }
  /*   *** Will uncomment and fix when namespaces are actually needed
    while (name_space_parts.size() > 0) {
      currentIndent = currentIndent.substr(2, currentIndent.size());
      file << currentIndent << "- ("
           << name_space_parts[name_space_parts.size() - 1].substr(
                  0, name_space_parts[name_space_parts.size() - 1].size() - 2)
           << ")\n";
      name_space_parts.pop_back();
    }
*/ // cout <<
                                                                      // "  -
                                                                      // \"" <<
                                                                      // fileName
                                                                      // << "\"
                                                                      // has
                                                                      // been
                                                                      // created.\n";

} // end  Parameters::saveSettingsFile

void Parameters::printParameterWithWraparound(
    std::stringstream &file, const std::string &current_indent,
    const std::string &entire_parameter, int max_line_length,
    int comment_indent) {

  auto pos_of_comment =
      entire_parameter.find_first_of("@@@#"); // must be cleaned
  if (pos_of_comment == std::string::npos) {
    std::cout << " Error : parameter has no comment";
    exit(1); // which makes type conversion to int safe after this??
  }
  if (int(pos_of_comment) > max_line_length - 9) {
    std::cout
        << " Warning: parameter name and value too large to fit on single "
           "line. Ignoring column width for this line\n";
  }

  std::string line;
  line += current_indent;
  line += entire_parameter.substr(0, pos_of_comment); // write name-value

  std::string sub_line(comment_indent, ' ');
  if (int(line.length()) < comment_indent)
    line +=
        sub_line.substr(0, comment_indent - line.length()); // pad with spaces

  auto comment =
      entire_parameter.substr(pos_of_comment + 3); // + 3 must be cleaned

  std::regex new_line(R"(\n)");
  comment = std::regex_replace(comment, new_line, "\n ");

  // add as much of the comment as possible to the line
  static const std::regex as_much_of_comment(
      R"(.{1,)" + std::to_string(max_line_length - line.length() - 2) +
      R"(}[^\s]*)");
  std::smatch a_m_c;
  std::regex_search(comment, a_m_c, as_much_of_comment);
  auto first_comment = a_m_c.str();
  line += "#" + first_comment.substr(2);
  file << line << '\n';

  comment = a_m_c.suffix();
  // write rest of the comments right-aligned with slight padding
  static const std::regex aligned_comments(
      R"(.{1,)" + std::to_string(max_line_length - comment_indent - 3) +
      R"(}[^\s]*)");
  for (auto &m : forEachRegexMatch(comment, aligned_comments)) {
    auto comment_piece = m.str();
    file << sub_line << "# " << comment_piece
         << (comment_piece.back() == '\n' ? "" : "\n");
  }
} // end  Parameters::printParameterWithWraparound

void Parameters::saveSettingsFiles(
    int max_line_length, int comment_indent,
    std::vector<std::string> name_space_list,
    std::vector<std::pair<std::string, std::vector<std::string>>>
        category_lists) {
  bool also_children;
  std::string file_name;
  std::vector<std::string> other_category_list;
  for (auto name_space : name_space_list) {
    for (auto clist : category_lists) {
      other_category_list.insert(other_category_list.end(),
                                 clist.second.begin(), clist.second.end());
      if (!name_space.empty() && name_space.back() == '*') {
        name_space.pop_back();
        also_children = true;
      } else {
        also_children = false;
      }
      // why bother at all?
      static const std::regex colon(R"(::)");
      file_name = std::regex_replace(name_space, colon, "_");
      if (!file_name.empty()) {
        file_name.pop_back();
        file_name += "_";
      }

      std::stringstream ss;
      if (clist.second.size() == 1 && clist.second[0].empty()) {
        other_category_list.insert(other_category_list.begin(), "-");
        saveSettingsFile(name_space, ss, other_category_list, max_line_length,
                         comment_indent, also_children);
      } else {
        saveSettingsFile(name_space, ss, clist.second, max_line_length,
                         comment_indent, also_children);
      }
      std::string workingString = ss.str();
      workingString.erase(
          std::remove_if(workingString.begin(), workingString.end(),
                         [](char c) { return c == ' ' || c == 11; }),
          workingString.end());
      bool lastCharEnter = false;
      bool fileEmpty = true;
      for (auto c : workingString) {
        if (c == 10) {
          lastCharEnter = true;
        } else {
          if (lastCharEnter) {
            if (!(c == '+' || c == '-' || c == 10)) {
              fileEmpty = false;
            }
          }
          lastCharEnter = false;
        }
      }
      if (!fileEmpty) {
        std::ofstream file(file_name + clist.first);
        file << ss.str();
        file.close();
      }
    }
  }
} // end Parameters::saveSettingsFiles
