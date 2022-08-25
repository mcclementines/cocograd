#ifndef COCOGRAD
#define COCOGRAD

struct Value {
    double data;
    double grad;
    double opt;
    struct Value *lhs;
    struct Value *rhs;
    void (*backward)(struct Value *self);
};

struct ValueList {
    struct Value **list;
    int size;
};

struct Value new_value(double data);

struct Value add_values(struct Value *lhs, struct Value *rhs);

struct Value add_value_double(struct Value *lhs, double rhs);

struct Value sub_value_double(struct Value *lhs, double rhs);

struct Value mul_values(struct Value *lhs, struct Value *rhs);

struct Value mul_value_double(struct Value *lhs, double rhs);

struct Value tanh_value(struct Value *self);

struct Value exp_value(struct Value *self);

struct Value pow_value_double(struct Value *base, double exp);

struct Value div_values(struct Value *numer, struct Value *denom);

bool is_value_in_list(struct Value *val, struct ValueList *list);

void add_value_to_list(struct Value *val, struct ValueList *list);

struct ValueList build_topo(struct Value *val);

void backwards(struct Value *val);

void print_value(struct Value *val);

void print_value_tree(struct Value *val);

#endif
