#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/rnn.h"
#include "dynet/gru.h"
#include "dynet/lstm.h"
#include "dynet/dict.h"
#include "dynet/expr.h"
#include "dynet/globals.h"

#include "easylogging++.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <tuple>

using namespace std;
using namespace dynet;

INITIALIZE_EASYLOGGINGPP

unsigned LAYERS = 2;
unsigned INPUT_DIM = 8;  //256
unsigned HIDDEN_DIM = 24;  // 1024
unsigned VOCAB_SIZE = 0;

dynet::Dict d;
int kSOS;
int kEOS;

// <word-id, sentence#, word#>
typedef tuple<int,int,int> TokenSkip;
enum { WORD=0, SIDX, WIDX };
typedef vector<TokenSkip> Sentence;
typedef vector<Sentence> Document;
typedef vector<Document> Corpus;


struct RNNSkipLM {
    LookupParameter p_c;
    Parameter p_R;
    Parameter p_bias;
    SimpleRNNBuilder builder;
    explicit RNNSkipLM(ParameterCollection& model) : builder(LAYERS, INPUT_DIM, HIDDEN_DIM, model, true) {
        p_c = model.add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM}); 
        p_R = model.add_parameters({VOCAB_SIZE, HIDDEN_DIM});
        p_bias = model.add_parameters({VOCAB_SIZE});
    }

    // return Expression of total loss
    Expression BuildLMGraph(const Document& doc, ComputationGraph& cg) {
        Expression i_R = parameter(cg, p_R); // hidden -> word rep parameter
        Expression i_bias = parameter(cg, p_bias);  // word bias
        vector<Expression> errs;
        vector<vector<Expression>> hidden;
        uint64_t si = 0;
        for (auto &sent: doc) {
            builder.new_graph(cg);  // reset RNN builder for new graph
            builder.start_new_sequence();
            hidden.push_back({});
            for (unsigned t = 0; t < sent.size()-1; ++t) {
                auto &token = sent[t];
                Expression i_x_t = lookup(cg, p_c, std::get<WORD>(token));
                // y_t = RNN(x_t)

                Expression i_y_t;
                if (std::get<SIDX>(token) >= 0) {
                    i_y_t = builder.add_auxiliary_input(i_x_t, 
                            hidden[std::get<SIDX>(token)][std::get<WIDX>(token)]);
                } else {
                    i_y_t = builder.add_input(i_x_t);
                }
                Expression i_r_t = i_bias + i_R * i_y_t;

                Expression i_err = pickneglogsoftmax(i_r_t, std::get<WORD>(sent[t+1]));
                errs.push_back(i_err);

                // only use top-most layer for skipping
                hidden.back().push_back(builder.back());
            }
            ++si;
        }
        Expression i_nerr = sum(errs);
        return i_nerr;
    }
};


void read_documents(const std::string &filename, Corpus &text);

int main(int argc, char** argv) {
    START_EASYLOGGINGPP(argc, argv);
    el::Configurations defaultConf;
    defaultConf.setToDefault();
    defaultConf.set(el::Level::Info, el::ConfigurationType::Format, "%datetime{%h:%m:%s} %level %msg");
    el::Loggers::reconfigureLogger("default", defaultConf);

    dynet::initialize(argc, argv);
    if (argc != 3 && argc != 4) {
        LOG(INFO) << "Usage: " << argv[0] << " corpus.txt dev.txt [model.file]\n";
        return 1;
    }
    kSOS = d.convert("<s>");
    kEOS = d.convert("</s>");

    // load the corpora
    Corpus training, dev;
    LOG(INFO) << "Reading training data from " << argv[1] << "...\n";
    read_documents(argv[1], training);
    d.freeze(); // no new word types allowed
    VOCAB_SIZE = d.size();
    LOG(INFO) << "Reading dev data from " << argv[2] << "...\n";
    read_documents(argv[2], dev);

    ostringstream os;
    os << "lm"
        << '_' << LAYERS
        << '_' << INPUT_DIM
        << '_' << HIDDEN_DIM
        << "-pid" << getpid() << ".params";
    const string fname = os.str();
    LOG(INFO) << "Parameters will be written to: " << fname << endl;
    double best = 9e+99;

    ParameterCollection model;
    bool use_momentum = false;
    Trainer* sgd = nullptr;
    if (use_momentum)
        sgd = new MomentumSGDTrainer(model);
    else
        sgd = new SimpleSGDTrainer(model);

    RNNSkipLM lm(model);
    if (argc == 4) {
        TextfileLoader loader(argv[3]);
        loader.populate(model);'
    }

    unsigned report_every_i = 50;
    unsigned dev_every_i_reports = 500;
    unsigned si = training.size();
    vector<unsigned> order(training.size());
    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
    bool first = true;
    int report = 0;
    unsigned lines = 0;
    while(1) {
        Timer iteration("completed in");
        double loss = 0;
        unsigned chars = 0;
        for (unsigned i = 0; i < report_every_i; ++i) {
            if (si == training.size()) {
                si = 0;
                if (first) { first = false; } else { sgd->update_epoch(); }
                LOG(INFO) << "**SHUFFLE\n";
                shuffle(order.begin(), order.end(), *rndeng);
            }

            // build graph for this instance
            ComputationGraph cg;
            auto& doc = training[order[si]];
            for (auto &sent: doc)
                chars += sent.size() - 1;
            ++si;
            //LOG(INFO) << "sent length " << sent.size();
            Expression loss_expr = lm.BuildLMGraph(doc, cg);
            loss += as_scalar(cg.forward(loss_expr));
            cg.backward(loss_expr);
            sgd->update();
            ++lines;
        }
        sgd->status();
        // FIXME: is chars incorrect?
        LOG(INFO) << " E = " << (loss / chars) << " ppl=" << exp(loss / chars) << ' ';
        //lm.RandomSample(); // why???

        // show score on dev data?
        report++;
        if (report % dev_every_i_reports == 0) {
            double dloss = 0;
            int dchars = 0;
            for (int i = 0; i < dev.size(); ++i) {
                const auto& doc = dev[i];
                ComputationGraph cg;
                Expression loss_expr = lm.BuildLMGraph(doc, cg);
                dloss += as_scalar(cg.forward(loss_expr));
                for (auto &sent: doc)
                    dchars += sent.size() - 1;
            }
            if (dloss < best) {
                best = dloss;
                TextFileSaver saver(fname);
                saver.save(model);
            }
            LOG(INFO) << "\n***DEV [epoch=" << (lines / (double)training.size()) << "] E = " << (dloss / dchars) << " ppl=" << exp(dloss / dchars) << ' ';
        }
    }
    delete sgd;
}

void read_documents(const std::string &filename, Corpus &corpus) {
    ifstream in(filename);
    assert(in);
    int toks = 0, lno = 0;
    string line;
    Document doc;
    while(std::getline(in, line)) {
        ++lno;
        auto sentence = read_sentence(line, &d);
        if (sentence.empty()) {
            // empty lines separate documents
            corpus.push_back(doc);
            doc.clear();
        } else {
            if (sentence.front() != kSOS && sentence.back() != kEOS) {
                LOG(INFO) << "Sentence in " << filename << ":" << lno << " didn't start or end with <s>, </s>\n";
                abort();
            }
            // re-package sentence
            Sentence s;
            for (auto tok: sentence)
                s.push_back(TokenSkip(tok,-1,-1));
            doc.push_back(s);
            toks += sentence.size();
        }
    }

    assert(doc.empty());

    // add in 'skip links' -- for now, based on identical tokens
    int num_skips = 0;
    for (auto &doc: corpus) {
        map<int, std::pair<int,int>> mrocc;
        for (int i = 0; i < doc.size(); ++i) {
            auto &sentence = doc[i];
            for (int j = 0; j < sentence.size(); ++j) {
                auto &token = sentence[j];
                if (std::get<WORD>(token) != kEOS && std::get<WORD>(token) != kSOS) {
                    const auto &found = mrocc.find(std::get<WORD>(token));
                    if (found != mrocc.end()) {
                        std::get<SIDX>(token) = found->second.first;
                        std::get<WIDX>(token) = found->second.second;
                        num_skips++;
                    }
                    mrocc[std::get<WORD>(token)] = make_pair(i, j);
                }
            }
        }
    }
    LOG(INFO) << corpus.size() << " documents, " << (lno-corpus.size()) << " sentences, " << num_skips << " skips, " << toks << " tokens, " << d.size() << " types\n";
}
