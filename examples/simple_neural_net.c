#include "cocograd.h"

#include <stdio.h>

int main() {
    init_net();

    int layers[3] = {4, 4, 1};
    struct MLP *mlp = init_mlp(3, layers, 3);

    double xs[4*3] = {2.0, 3.0, -1.0,
                      3.0, -1.0, 0.5,
                      0.5, 1.0, 1.0,
                      1.0, 1.0, 1.0};
    
    double ys[4] = {1.0, -1.0, -1.0, 1.0};

    struct ValueList **inputs = value_list_arr(xs, 4, 3);
    struct ValueList *truths = init_value_list(ys, 4);

    train_network_with_iteration(mlp, inputs, truths, 4, .02, 200000);

    struct ValueList *outs = init_value_list(0,0);
    for (int i = 0; i < 4; i++) {
        add_value_lists(outs, eval_mlp(mlp, inputs[i]));
    }

    print_value_list(outs);
}