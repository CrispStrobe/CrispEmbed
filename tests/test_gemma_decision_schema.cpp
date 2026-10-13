#include "core/decision_schema.h"
#include "core/clean_exit.h"
#include <fstream>
#include <iostream>
#include <sstream>

static int crispembed_test_main(int argc, char ** argv) {
    if (argc != 2) return 2;
    try {
        std::ifstream file(argv[1]);
        std::ostringstream data;
        data << file.rdbuf();
        using namespace core_decision;
        const auto root = Parser(data.str()).parse();
        const auto questions = unpack(root.get("questions"));
        std::vector<std::string> bank;
        auto add = [&](const std::string & text) {
            if (std::find(bank.begin(), bank.end(), text) == bank.end()) bank.push_back(text);
        };
        add(state_text(root.get("workflow").kind == Value::null_value ? "" : python_str(root.get("workflow")),
                       root.get("state")));
        for (auto & item : questions.fields) {
            auto q = question(item.first, item.second);
            add(question_text(q));
            for (size_t i = 0; i < q.options.size(); ++i) add(option_text(q, i));
        }
        std::cout << "[";
        for (size_t i = 0; i < bank.size(); ++i) std::cout << (i ? "," : "") << quote(bank[i]);
        std::cout << "]\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
