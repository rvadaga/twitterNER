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
#include "../utils/getpid.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace std;
using namespace dynet;

unsigned LAYERS = 2;
unsigned INPUT_DIM = 8;  //256
unsigned HIDDEN_DIM = 24;  // 1024
unsigned BATCH_SIZE = 4;
unsigned VOCAB_SIZE = 0;

dynet::Dict d;
int kSOS;
int kEOS;

template <class Builder>
struct RNNLanguageModel {
  LookupParameter p_c;
  Parameter p_R;
  Parameter p_bias;
  Parameter p_last;
  Parameter p_last_bias;
  Builder builder;
  explicit RNNLanguageModel(ParameterCollection& model) : builder(LAYERS, INPUT_DIM, HIDDEN_DIM, model) {
    p_c = model.add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM});
    p_R = model.add_parameters({VOCAB_SIZE, HIDDEN_DIM});
    p_bias = model.add_parameters({VOCAB_SIZE});
    p_last = model.add_parameters({1, LAYERS * HIDDEN_DIM});
    p_last_bias = model.add_parameters({1});
  }

  // return Expression of total loss
  Expression BuildLMGraphs(const vector<vector<int> >& sents,
                           unsigned id,
                           unsigned bsize,
                           unsigned & chars,
                           ComputationGraph& cg) {
    const unsigned slen = sents[id].size();
    builder.new_graph(cg);  // reset RNN builder for new graph
    builder.start_new_sequence();
    Expression i_R = parameter(cg, p_R); // hidden -> word rep parameter
    Expression i_bias = parameter(cg, p_bias);  // word bias
    vector<Expression> errs;
    vector<unsigned> last_arr(bsize, sents[0][0]), next_arr(bsize);
    for (unsigned t = 1; t < slen; ++t) {
      for (unsigned i = 0; i < bsize; ++i) {
        next_arr[i] = sents[id + i][t];
        if (next_arr[i] != *sents[id].rbegin()) chars++; // add non-EOS
      }
      // y_t = RNN(x_t)
      Expression i_x_t = lookup(cg, p_c, last_arr);
      Expression i_y_t = builder.add_input(i_x_t);
      // Expression i_r_t = i_bias + i_R * i_y_t;
      // Expression i_err = pickneglogsoftmax(i_r_t, next_arr);
      // errs.push_back(i_err);
      last_arr = next_arr;
    }
    Expression i_last = parameter(cg, p_last);
    Expression i_last_bias = parameter(cg, p_last_bias);

    Expression last_h = builder.final_h();
    Expression err = i_last_bias + i_last * last_h;
    Expression i_nerr = sum_batches(err);
    return i_nerr;
  }

  // return Expression for total loss
  void RandomSample(int max_len = 150) {
    cerr << endl;
    ComputationGraph cg;
    builder.new_graph(cg);  // reset RNN builder for new graph
    builder.start_new_sequence();

    Expression i_R = parameter(cg, p_R);
    Expression i_bias = parameter(cg, p_bias);
    vector<Expression> errs;
    int len = 0;
    int cur = kSOS;
    while (len < max_len && cur != kEOS) {
      ++len;
      Expression i_x_t = lookup(cg, p_c, cur);
      // y_t = RNN(x_t)
      Expression i_y_t = builder.add_input(i_x_t);
      Expression i_r_t = i_bias + i_R * i_y_t;

      Expression ydist = softmax(i_r_t);

      unsigned w = 0;
      while (w == 0 || (int)w == kSOS) {
        auto dist = as_vector(cg.incremental_forward(ydist));
        double p = rand01();
        for (; w < dist.size(); ++w) {
          p -= dist[w];
          if (p < 0.0) { break; }
        }
        if (w == dist.size()) w = kEOS;
      }
      cerr << (len == 1 ? "" : " ") << d.convert(w);
      cur = w;
    }
    cerr << endl;
  }
};

// Sort in descending order of length
struct CompareLen {
  bool operator()(const std::vector<int>& first, const std::vector<int>& second) {
    return first.size() > second.size();
  }
};

int main(int argc, char** argv) {
  dynet::initialize(argc, argv);
  if (argc != 3 && argc != 4) {
    cerr << "Usage: " << argv[0] << " corpus.txt dev.txt [model.file]\n";
    return 1;
  }
  kSOS = d.convert("<s>");
  kEOS = d.convert("</s>");
  vector<vector<int>> training, dev;
  string line;
  int tlc = 0;
  int ttoks = 0;
  cerr << "Reading training data from " << argv[1] << "...\n";
  {
    ifstream in(argv[1]);
    assert(in);
    while (getline(in, line)) {
      ++tlc;
      training.push_back(read_sentence(line, &d));
      ttoks += training.back().size();
      if (training.back().front() != kSOS && training.back().back() != kEOS) {
        cerr << "Training sentence in " << argv[1] << ":" << tlc << " didn't start or end with <s>, </s>\n";
        abort();
      }
    }
    cerr << tlc << " lines, " << ttoks << " tokens, " << d.size() << " types\n";
  }
  d.freeze(); // no new word types allowed
  VOCAB_SIZE = d.size();

  // Sort the training sentences in descending order of length
  CompareLen comp;
  sort(training.begin(), training.end(), comp);
  // Pad the sentences in the same batch with EOS so they are the same length
  // This modifies the training objective a bit by making it necessary to
  // predict EOS multiple times, but it's easy and not so harmful
  for (size_t i = 0; i < training.size(); i += BATCH_SIZE)
    for (size_t j = 1; j < BATCH_SIZE; ++j)
      while (training[i + j].size() < training[i].size())
        training[i + j].push_back(kEOS);

  int dlc = 0;
  int dtoks = 0;
  cerr << "Reading dev data from " << argv[2] << "...\n";
  {
    ifstream in(argv[2]);
    assert(in);
    while (getline(in, line)) {
      ++dlc;
      dev.push_back(read_sentence(line, &d));
      dtoks += dev.back().size();
      if (dev.back().front() != kSOS && dev.back().back() != kEOS) {
        cerr << "Dev sentence in " << argv[2] << ":" << tlc << " didn't start or end with <s>, </s>\n";
        abort();
      }
    }
    cerr << dlc << " lines, " << dtoks << " tokens\n";
  }

  ostringstream os;
  os << "lm"
     << '_' << LAYERS
     << '_' << INPUT_DIM
     << '_' << HIDDEN_DIM
     << "-pid" << getpid() << ".params";
  const string fname = os.str();
  cerr << "Parameters will be written to: " << fname << endl;
  double best = 9e+99;

  ParameterCollection model;
  Trainer* sgd = nullptr;
  sgd = new SimpleSGDTrainer(model);
  sgd->clip_threshold *= BATCH_SIZE;

  RNNLanguageModel<LSTMBuilder> lm(model);
  if (argc == 4) {
    TextFileLoader loader(argv[3]);
    loader.populate(model);
  }

  unsigned report_every_i = 50;
  unsigned dev_every_i_reports = 500;
  vector<unsigned> order((training.size() + BATCH_SIZE - 1) / BATCH_SIZE);
  unsigned si = order.size();
  for (unsigned i = 0; i < order.size(); ++i) order[i] = i * BATCH_SIZE;
  bool first = true;
  int report = 0;
  unsigned lines = 0;
  while (1) {
    Timer iteration("completed in");
    double loss = 0;
    unsigned chars = 0;
    for (unsigned i = 0; i < report_every_i; ++i, ++si) {
      if (si == order.size()) {
        si = 0;
        if (first) { first = false; } else { sgd->update_epoch(); }
        cerr << "**SHUFFLE\n";
        shuffle(order.begin(), order.end(), *rndeng);
      }
      // build graph for this instance
      ComputationGraph cg;
      unsigned bsize = std::min((unsigned)training.size() - order[si], BATCH_SIZE); // Batch size
      Expression loss_expr = lm.BuildLMGraphs(training, order[si], bsize, chars, cg);
      loss += as_scalar(cg.forward(loss_expr));
      cg.backward(loss_expr);
      sgd->update();
      lines += bsize;
    }
    sgd->status();
    cerr << " E = " << (loss / chars) << " ppl=" << exp(loss / chars) << ' ';
    lm.RandomSample();

    // show score on dev data?
    report++;
    if (report % dev_every_i_reports == 0) {
      double dloss = 0;
      unsigned dchars = 0;
      for (unsigned i = 0; i < dev.size(); ++i) {
        ComputationGraph cg;
        Expression loss_expr = lm.BuildLMGraphs(dev, i, 1, dchars, cg);
        dloss += as_scalar(cg.forward(loss_expr));
      }
      if (dloss < best) {
        best = dloss;
        TextFileSaver saver(fname);
        saver.save(model);
      }
      cerr << "\n***DEV [epoch=" << (lines / (double)training.size()) << "] E = " << (dloss / dchars) << " ppl=" << exp(dloss / dchars) << ' ';
    }
  }
  delete sgd;
}
