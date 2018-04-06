#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/expr.h"
#include "dynet/io.h"

#include <iostream>
#include <fstream>

using namespace std;
using namespace dynet;

int main(int argc, char** argv) {
  dynet::initialize(argc, argv);

  // parameters
  const unsigned HIDDEN_SIZE = 8;
  ParameterCollection m;
  SimpleSGDTrainer sgd(m);
  //MomentumSGDTrainer sgd(m);

  ComputationGraph cg;

  Parameter p_W, p_b, p_V, p_a;
  p_W = m.add_parameters({HIDDEN_SIZE, 2});
  p_b = m.add_parameters({HIDDEN_SIZE});
  p_V = m.add_parameters({1, HIDDEN_SIZE});
  p_a = m.add_parameters({1});
  if (argc == 2) {
    TextFileLoader loader(argv[1]);
    loader.populate(m);
  }

  Expression W = parameter(cg, p_W);
  Expression b = parameter(cg, p_b);
  Expression V = parameter(cg, p_V);
  Expression a = parameter(cg, p_a);

  vector<float> x_values(2);  // set x_values to change the inputs to the network
  Expression x = input(cg, {2}, &x_values);
  dynet::real y_value;  // set y_value to change the target output
  Expression y = input(cg, &y_value);

  Expression h = tanh(W*x + b);
  Expression y_pred = logistic(V*h + a);
  Expression loss_expr = binary_log_loss(y_pred, y);

  cg.print_graphviz();

  // train the parameters
  for (unsigned iter = 0; iter < 2000; ++iter) {
    double loss = 0;
    for (unsigned mi = 0; mi < 4; ++mi) {
      bool x1 = mi % 2;
      bool x2 = (mi / 2) % 2;
      x_values[0] = x1 ? 1 : 0;
      x_values[1] = x2 ? 1 : 0;
      y_value = (x1 != x2) ? 1 : 0;
      loss += as_scalar(cg.forward(loss_expr));
      cg.backward(loss_expr);
      sgd.update();
    }
    sgd.update_epoch();
    loss /= 4;
    cerr << "E = " << loss << endl;
  }

  // Output the model and parameter objects
  // to a file.
  TextFileSaver saver("/tmp/xor-xent.model");
  saver.save(m);
}
