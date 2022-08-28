# Cocograd

Cocograd is a C clone or rather, poor copy, of Micrograd by Andrej Karpathy. It is based on a video he published about building Micrograd, and the project exists purely for my personal education. 

It does not have all the features of Micrograd, rather it has all the features laid out by Andrej in his video. It *could* have all the features, but at the moment I think I learned what I needed to learn and will put implementing anything else on hold, maybe here or there I'll add bits and pieces.

The biggest drawback to this project is using C. Python certainly has more useful features for building out Micrograd in the way Micrograd is designed. A better clone (although, maybe not a *true* clone) would find a way to design Micrograd in a C way. If that was my goal, this project would have taken infinitely longer and I would have learned more about C than about neural networks, which was my real goal.

I think the biggest drawback to my implementation is that I have pointers, everywhere, and many do not and cannot easily be free'd. This is bad programming practice, I know. But OSs are just so good and cleaning up stray memory nowadays...

Anyways, it works as micrograd does, except that there is no overloading of operators and every mathemical expression is built from specific functions. I tried to name the functions as literal as possible to avoid confusion. The result is a very wordy and lengthy function list, which I actually don't mind. Using just the provided neuralnet functions, most of them are hidden away, leaving a pretty clean usage when implementing a network. See *simple_neural_net.c* for what I mean.

## Examples

For now, I only have one real example. It can be run by running `make` in the src directory and then `make run prog=simple_neural_net` in the examples directory. This example is a C clone, albeit a better copy than the library itself, of Andrej's example in his video. Cool stuff.

I think the first thing I will do when I want to come back to this project is to implement Micrograd's main example in Cocograd. The library should be able to handle it. Will find out, someday.