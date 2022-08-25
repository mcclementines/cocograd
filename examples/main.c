#include <stdio.h>

#include "grad.h"

int main()
{
    struct Value x1 = new_value(2.0);
    struct Value x2 = new_value(0.0);

    struct Value w1 = new_value(-3.0);
    struct Value w2 = new_value(1.0);

    struct Value b = new_value(6.8813735870195432);

    struct Value x1w1 = mul_values(&x1, &w1);

    struct Value x2w2 = mul_values(&x2, &w2);

    struct Value x1w1x2w2 = add_values(&x1w1, &x2w2);

    struct Value n = add_values(&x1w1x2w2, &b);

    // tan h
    // struct Value o = tanh_value(&n);

    struct Value q = mul_value_double(&n, 2.0);
    struct Value e = exp_value(&q);
    struct Value r = add_value_double(&e, -1);
    struct Value s = add_value_double(&e, 1);
    struct Value o = div_values(&r, &s);
    
    //

    // struct Value o = tanh_value(&n);

    backwards(&o);

    print_value_tree(&o);
}
